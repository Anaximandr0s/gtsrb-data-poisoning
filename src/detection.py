"""Detection of poisoned training samples from the model's last hidden layer.

Every method groups samples by their training label, the only labels the defender
has. None of them sees the true poison indices; ``evaluate_detection`` is the only
function that does, and it only scores a finished detection.
"""

import numpy as np
import torch
from art.defences.detector.poison import ActivationDefence
from art.estimators.classification import PyTorchClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from torch import nn

from src.evaluate import make_art_classifier
from src.utils import set_seed

FEATURE_LAYER = "fc"


def get_activations(model: nn.Module, x: np.ndarray, batch_size: int) -> np.ndarray:
    """Last-hidden-layer activations ``model.features(x)``, shape (N, fc_dim)."""
    model.eval()
    device = next(model.parameters()).device
    batches = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start : start + batch_size]).to(device)
            batches.append(model.features(batch).cpu().numpy())
    return np.concatenate(batches)


def _reduce(activations: np.ndarray, n_dims: int) -> np.ndarray:
    """PCA projection to at most ``n_dims`` dimensions (fewer if the class is tiny)."""
    n_components = min(n_dims, len(activations), activations.shape[1])
    return PCA(n_components=n_components, svd_solver="full").fit_transform(activations)


def cluster_activations(
    activations: np.ndarray, labels: np.ndarray, cfg: dict, seed: int
) -> tuple[np.ndarray, dict]:
    """Activation clustering (Chen et al. 2018): per label, PCA -> 2-means; flag a small cluster.

    The smaller cluster of a class is flagged only if its share of the class is
    below ``size_threshold``. Returns the boolean suspect array and per-class info.
    """
    params = cfg["detection"]["clustering"]
    suspect = np.zeros(len(labels), dtype=bool)
    info = {}
    for class_id in np.unique(labels):
        members = np.flatnonzero(labels == class_id)
        reduced = _reduce(activations[members], cfg["detection"]["pca_dims"])
        kmeans = KMeans(n_clusters=params["n_clusters"], n_init=10, random_state=seed)
        clusters = kmeans.fit_predict(reduced)
        sizes = np.bincount(clusters, minlength=params["n_clusters"])
        smaller = int(sizes.argmin())
        share = float(sizes[smaller] / len(members))
        flagged = share < params["size_threshold"]
        if flagged:
            suspect[members[clusters == smaller]] = True
        info[int(class_id)] = {
            "cluster_sizes": sizes.tolist(),
            "smaller_share": share,
            "silhouette": float(silhouette_score(reduced, clusters)),
            "flagged": bool(flagged),
        }
    return suspect, info


class _HiddenLayerActivationDefence(ActivationDefence):
    """ART's ActivationDefence reading the last hidden layer.

    ART takes the last entry of ``classifier.layer_names``, which for our model is
    the 43-logit output layer, not the 256-unit hidden layer of the paper.
    """

    def _get_activations(self, x_train: np.ndarray | None = None) -> np.ndarray:
        """Activations of ``FEATURE_LAYER`` for the training data, flattened to (N, D)."""
        x = self.x_train if x_train is None else x_train
        activations = self.classifier.get_activations(x, layer=FEATURE_LAYER, batch_size=128)
        return activations.reshape(len(x), -1)


def art_activation_defence(
    classifier: PyTorchClassifier,
    x: np.ndarray,
    labels: np.ndarray,
    cfg: dict,
    cluster_analysis: str | None = None,
) -> np.ndarray:
    """ART ActivationDefence (2 clusters, PCA to ``pca_dims``) on the hidden layer; True = suspect.

    ART's PCA and KMeans take no seed: call ``set_seed`` before for reproducible results.
    """
    one_hot = np.eye(classifier.nb_classes, dtype=np.float32)[labels]
    defence = _HiddenLayerActivationDefence(classifier, x, one_hot)
    _, is_clean = defence.detect_poison(
        nb_clusters=2,
        reduce="PCA",
        nb_dims=cfg["detection"]["pca_dims"],
        cluster_analysis=cluster_analysis or cfg["detection"]["art"]["cluster_analysis"],
    )
    return np.array(is_clean) == 0


def centroid_outliers(
    activations: np.ndarray, labels: np.ndarray, cfg: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Per label, PCA -> distance to the class centroid -> z-score; flag z above ``z_threshold``.

    Returns the boolean suspect array and the z-scores.
    """
    z_scores = np.zeros(len(labels))
    for class_id in np.unique(labels):
        members = np.flatnonzero(labels == class_id)
        reduced = _reduce(activations[members], cfg["detection"]["pca_dims"])
        distances = np.linalg.norm(reduced - reduced.mean(axis=0), axis=1)
        z_scores[members] = (distances - distances.mean()) / distances.std()
    return z_scores > cfg["detection"]["centroid"]["z_threshold"], z_scores


def calibrated_centroid_outliers(
    train_activations: np.ndarray,
    train_labels: np.ndarray,
    val_activations: np.ndarray,
    val_labels: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Distance to the clean validation centroid of each class, thresholded on clean validation data.

    Per class: Euclidean distance (all dimensions) to the centroid of the clean
    validation activations, divided by the median distance of those validation
    images. The threshold is the ``percentile`` of these normalised scores pooled over
    all validation images, so about (100 - percentile)% of clean images are flagged.
    Pooling avoids per-class percentiles estimated from as few as 30 images.
    Returns (suspect, training scores, threshold).
    """
    params = cfg["detection"]["calibrated"]
    if params["space"] != "full":
        raise ValueError(f"only the full activation space is implemented, got {params['space']!r}")
    train_scores = np.zeros(len(train_labels))
    val_scores = np.zeros(len(val_labels))
    for class_id in np.unique(val_labels):
        val_members = val_labels == class_id
        centroid = val_activations[val_members].mean(axis=0)
        val_distances = np.linalg.norm(val_activations[val_members] - centroid, axis=1)
        scale = np.median(val_distances)
        val_scores[val_members] = val_distances / scale
        train_members = train_labels == class_id
        train_scores[train_members] = (
            np.linalg.norm(train_activations[train_members] - centroid, axis=1) / scale
        )
    threshold = float(np.percentile(val_scores, params["percentile"]))
    return train_scores > threshold, train_scores, threshold


def run_detection(
    model: nn.Module,
    x: np.ndarray,
    labels: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: dict,
) -> dict:
    """Run the four methods on a training set with its (possibly poisoned) labels; no ground truth used.

    The calibrated method also uses the trusted clean validation set ``x_val, y_val``.
    Returns the activations, a suspect mask per method, the clustering info, the
    centroid z-scores, and the calibrated scores and threshold.
    """
    batch_size = cfg["train"]["batch_size"]
    activations = get_activations(model, x, batch_size)
    clustering_suspect, clustering_info = cluster_activations(activations, labels, cfg, cfg["seed"])
    centroid_suspect, z_scores = centroid_outliers(activations, labels, cfg)
    set_seed(cfg["seed"])
    art_suspect = art_activation_defence(make_art_classifier(model, cfg), x, labels, cfg)
    calibrated_suspect, calibrated_scores, threshold = calibrated_centroid_outliers(
        activations, labels, get_activations(model, x_val, batch_size), y_val, cfg
    )
    return {
        "activations": activations,
        "suspects": {
            "clustering": clustering_suspect,
            "art": art_suspect,
            "centroid": centroid_suspect,
            "calibrated": calibrated_suspect,
        },
        "clustering_info": clustering_info,
        "z_scores": z_scores,
        "calibrated_scores": calibrated_scores,
        "calibrated_threshold": threshold,
    }


def evaluate_detection(
    suspect: np.ndarray, poison_indices: np.ndarray, y_train: np.ndarray, target_class: int
) -> dict:
    """Score a detection against the true poison indices: counts, precision, recall, F1, per class."""
    is_poison = np.zeros(len(suspect), dtype=bool)
    is_poison[np.asarray(poison_indices, dtype=int)] = True
    tp = int((suspect & is_poison).sum())
    fp = int((suspect & ~is_poison).sum())
    fn = int((~suspect & is_poison).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    classes, counts = np.unique(y_train[suspect], return_counts=True)
    return {
        "n_flagged": int(suspect.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "flagged_per_class": {int(c): int(n) for c, n in zip(classes, counts)},
        "flagged_in_target_class": int((suspect & (y_train == target_class)).sum()),
        "false_positive_indices": np.flatnonzero(suspect & ~is_poison).tolist(),
        "false_negative_indices": np.flatnonzero(~suspect & is_poison).tolist(),
    }
