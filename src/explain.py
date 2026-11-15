"""Visual explanations: Grad-CAM heatmaps, attention on the trigger corner, t-SNE of activations.

For display only: nothing here is used to detect poison or to choose a parameter.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.attack import _patch_slices


def grad_cam(model: nn.Module, x: np.ndarray, batch_size: int, class_idx: int | None = None) -> np.ndarray:
    """Grad-CAM heatmaps (N, H, W) in [0, 1] at ``model.last_conv`` for the predicted class (or ``class_idx``).

    Channel weights are the mean gradients of the class score over each feature map; the
    weighted sum goes through ReLU, is upsampled to the image size and divided by its max.
    """
    model.eval()
    device = next(model.parameters()).device
    saved = {}

    def keep_output(_module, _inputs, output):
        saved["maps"] = output
        output.register_hook(lambda grad: saved.__setitem__("grads", grad))

    handle = model.last_conv.register_forward_hook(keep_output)
    heatmaps = []
    try:
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start : start + batch_size]).to(device)
            logits = model(batch)
            target = logits.argmax(dim=1) if class_idx is None else torch.full((len(batch),), class_idx, device=device)
            model.zero_grad()
            logits.gather(1, target[:, None]).sum().backward()
            weights = saved["grads"].mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * saved["maps"]).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
            peak = cam.flatten(1).max(dim=1).values.clamp_min(1e-12)
            heatmaps.append((cam / peak[:, None, None]).detach().cpu().numpy())
    finally:
        handle.remove()
    return np.concatenate(heatmaps)


def trigger_attention(heatmaps: np.ndarray, cfg: dict) -> np.ndarray:
    """Share of each heatmap's mass inside the corner square of side ``explain.attention_region`` around the trigger."""
    size = cfg["explain"]["attention_region"]
    rows, cols = _patch_slices(heatmaps.shape[-2], heatmaps.shape[-1], size, cfg["backdoor"]["trigger"]["position"])
    total = heatmaps.sum(axis=(-2, -1))
    inside = heatmaps[..., rows, cols].sum(axis=(-2, -1))
    return np.divide(inside, total, out=np.zeros_like(total), where=total > 0)

