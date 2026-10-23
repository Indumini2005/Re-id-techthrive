"""Losses and image-quality metrics for denoiser training."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PSNRLoss(nn.Module):
    """The loss NAFNet is trained with: -PSNR, which optimizes the metric
    directly instead of a proxy. ``loss_weight`` is negative by construction,
    so minimizing it maximizes PSNR."""

    def __init__(self, loss_weight: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.loss_weight = loss_weight
        self.scale = 10.0 / math.log(10)
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = ((pred - target) ** 2).mean(dim=(1, 2, 3)) + self.eps
        return self.loss_weight * self.scale * torch.log(mse).mean()


class CharbonnierLoss(nn.Module):
    """Smooth L1 variant; more tolerant of outliers than plain L2, which
    matters when the degradation chain occasionally produces extreme noise."""

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps2 = eps * eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((pred - target) ** 2 + self.eps2).mean()


class EdgeLoss(nn.Module):
    """Charbonnier on a Laplacian pyramid level. Keeps the denoiser from
    smoothing away the facial detail that stages 3-4 depend on."""

    def __init__(self):
        super().__init__()
        k = torch.tensor([[0.05, 0.25, 0.4, 0.25, 0.05]])
        kernel = (k.T @ k).unsqueeze(0).unsqueeze(0)
        self.register_buffer("kernel", kernel.repeat(3, 1, 1, 1))
        self.criterion = CharbonnierLoss()

    def _blur(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), self.kernel, groups=x.shape[1])

    def _laplacian(self, x: torch.Tensor) -> torch.Tensor:
        blurred = self._blur(x)
        down = blurred[:, :, ::2, ::2]
        up = torch.zeros_like(x)
        up[:, :, ::2, ::2] = down * 4
        return x - self._blur(up)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.criterion(self._laplacian(pred), self._laplacian(target))


class CombinedLoss(nn.Module):
    """Default training objective: PSNR plus a small edge term.

    Set ``edge_weight=0`` to reproduce the stock NAFNet recipe exactly.
    """

    def __init__(self, psnr_weight: float = 1.0, edge_weight: float = 0.05,
                 charbonnier_weight: float = 0.0):
        super().__init__()
        self.psnr = PSNRLoss()
        self.edge = EdgeLoss() if edge_weight > 0 else None
        self.charbonnier = CharbonnierLoss() if charbonnier_weight > 0 else None
        self.psnr_weight = psnr_weight
        self.edge_weight = edge_weight
        self.charbonnier_weight = charbonnier_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        parts = {}
        total = pred.new_zeros(())

        if self.psnr_weight:
            parts["psnr"] = self.psnr(pred, target)
            total = total + self.psnr_weight * parts["psnr"]
        if self.edge is not None:
            parts["edge"] = self.edge(pred, target)
            total = total + self.edge_weight * parts["edge"]
        if self.charbonnier is not None:
            parts["charb"] = self.charbonnier(pred, target)
            total = total + self.charbonnier_weight * parts["charb"]

        parts["total"] = total
        return total, {k: float(v.detach()) for k, v in parts.items()}


# --- metrics ---------------------------------------------------------------
@torch.no_grad()
def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """Mean PSNR (dB) over the batch."""
    pred = pred.clamp(0, max_val)
    mse = ((pred - target) ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-10)
    return float((10 * torch.log10(max_val**2 / mse)).mean())


@torch.no_grad()
def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11,
         max_val: float = 1.0) -> float:
    """Gaussian-window SSIM, averaged over the batch and channels."""
    pred = pred.clamp(0, max_val)
    channels = pred.shape[1]
    device, dtype = pred.device, pred.dtype

    sigma = 1.5
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = (g / g.sum()).unsqueeze(0)
    window = (g.T @ g).expand(channels, 1, window_size, window_size).contiguous()

    pad = window_size // 2
    mu1 = F.conv2d(pred, window, padding=pad, groups=channels)
    mu2 = F.conv2d(target, window, padding=pad, groups=channels)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=pad, groups=channels) - mu1_mu2

    c1, c2 = (0.01 * max_val) ** 2, (0.03 * max_val) ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return float(ssim_map.mean())


def build_loss(name: str = "combined", **kwargs) -> nn.Module:
    name = name.lower()
    if name == "psnr":
        return CombinedLoss(psnr_weight=1.0, edge_weight=0.0)
    if name == "charbonnier":
        return CombinedLoss(psnr_weight=0.0, edge_weight=0.0, charbonnier_weight=1.0)
    if name == "combined":
        return CombinedLoss(**kwargs)
    raise ValueError(f"unknown loss '{name}' (psnr | charbonnier | combined)")
