"""Generic training loop shared by the clean, poisoned and repaired models.

Only the data given to these functions changes between the three models.
"""

import copy
import logging
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import resolve_path
from src.model import GtsrbCNN, build_model
from src.utils import get_device, load_json, save_json

logger = logging.getLogger(__name__)

# config sections whose change makes a saved checkpoint stale
_TRAINING_SECTIONS = ("seed", "data", "model", "train")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    desc: str = "",
) -> tuple[float, float]:
    """One pass over ``loader``: trains if an optimizer is given, else evaluates. Returns (loss, accuracy)."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in tqdm(loader, desc=desc, leave=False, disable=not desc):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)
    return total_loss / seen, correct / seen


def make_scheduler(optimizer: torch.optim.Optimizer, cfg: dict) -> ReduceLROnPlateau | None:
    """ReduceLROnPlateau on validation accuracy, or None if ``train.scheduler`` is null."""
    params = cfg["train"]["scheduler"]
    if params is None:
        return None
    # mode="max" because the monitored metric is an accuracy: higher is better
    return ReduceLROnPlateau(
        optimizer, mode="max", factor=params["factor"], patience=params["patience"]
    )


def _checkpoint_path(cfg: dict, run_name: str) -> Path:
    """Path of the checkpoint file for ``run_name``."""
    return resolve_path(cfg, "models_dir") / f"{run_name}.pt"


def _history_path(cfg: dict, run_name: str) -> Path:
    """Path of the training-history JSON for ``run_name``."""
    return resolve_path(cfg, "metrics_dir") / f"{run_name}_history.json"


def train_model(
    model: GtsrbCNN, train_loader: DataLoader, val_loader: DataLoader, cfg: dict, run_name: str
) -> dict:
    """Train with Adam and early stopping on val accuracy; keep and save the best epoch's weights.

    Saves ``results/models/<run_name>.pt`` and ``results/metrics/<run_name>_history.json``
    and returns the history (per-epoch losses, accuracies, learning rate and time).
    """
    train_cfg = cfg["train"]
    if train_cfg["augmentation"]:
        raise NotImplementedError("augmentation is disabled on purpose: it would move the trigger")
    device = get_device()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"]
    )
    scheduler = make_scheduler(optimizer, cfg)

    history = {
        key: [] for key in ("train_loss", "train_acc", "val_loss", "val_acc", "lr", "epoch_time_s")
    }
    best_acc, best_epoch, best_state = -1.0, 0, None
    start = time.perf_counter()
    for epoch in range(1, train_cfg["epochs"] + 1):
        epoch_start = time.perf_counter()
        lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer, desc=f"{run_name} epoch {epoch}"
        )
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step(val_acc)
        record = {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": lr,
            "epoch_time_s": time.perf_counter() - epoch_start,
        }
        for key, value in record.items():
            history[key].append(value)
        logger.info(
            "%s epoch %d: train loss %.4f acc %.4f | val loss %.4f acc %.4f | lr %.1e | %.1fs",
            run_name, epoch, train_loss, train_acc, val_loss, val_acc, lr, record["epoch_time_s"],
        )
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= train_cfg["patience"]:
            logger.info("Early stopping: no val accuracy gain for %d epochs", train_cfg["patience"])
            break

    model.load_state_dict(best_state)
    model.eval()
    history.update(
        best_epoch=best_epoch, best_val_acc=best_acc, total_time_s=time.perf_counter() - start
    )
    logger.info("Best val accuracy %.4f at epoch %d", best_acc, best_epoch)

    path = _checkpoint_path(cfg, run_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": cfg,
            "norm": {
                "mean": model.normalize.mean.flatten().tolist(),
                "std": model.normalize.std.flatten().tolist(),
            },
            "best_epoch": best_epoch,
            "best_val_acc": best_acc,
        },
        path,
    )
    logger.info("Saved checkpoint %s", path)
    save_json(history, _history_path(cfg, run_name))
    return history


def load_model(path: Path, cfg: dict) -> GtsrbCNN:
    """Rebuild the model from ``cfg`` and a checkpoint (norm stats included), in eval mode."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(cfg, checkpoint["norm"]["mean"], checkpoint["norm"]["std"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(get_device()).eval()


def _training_config(cfg: dict) -> dict:
    """The config sections that determine the trained weights."""
    return {key: cfg.get(key) for key in _TRAINING_SECTIONS}


def train_or_load(
    model: GtsrbCNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict,
    run_name: str,
    force_retrain: bool = False,
) -> tuple[GtsrbCNN, dict]:
    """Reuse the saved ``run_name`` checkpoint if its training config matches ``cfg``, else train.

    Returns the model (eval mode) and its training history.
    """
    path, history_path = _checkpoint_path(cfg, run_name), _history_path(cfg, run_name)
    if force_retrain:
        logger.info("Training %s: retraining forced", run_name)
    elif not (path.exists() and history_path.exists()):
        logger.info("Training %s: no saved checkpoint or history", run_name)
    else:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if _training_config(checkpoint["config"]) == _training_config(cfg):
            logger.info("Loaded %s from %s (training config unchanged)", run_name, path)
            model.load_state_dict(checkpoint["state_dict"])
            return model.to(get_device()).eval(), load_json(history_path)
        logger.info("Training %s: training config differs from the saved checkpoint", run_name)
    return model, train_model(model, train_loader, val_loader, cfg, run_name)
