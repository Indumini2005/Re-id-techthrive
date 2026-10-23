"""Datasets for denoiser training.

Two modes:

* **paired**   - you already have ``noisy/`` and ``clean/`` directories with
  matching filenames (a ``_noise`` suffix on the noisy side is tolerated).
* **synthetic** - you only have clean images; the CCTV degradation chain in
  ``degradations.py`` runs on the fly, so every epoch sees fresh corruption.

Synthetic is the right default: you can point it at any sharp image folder
(COCO, your own 4K stills, scraped frames) and get unlimited pairs.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .degradations import PRESETS, DegradationConfig, degrade

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(root: Path, recursive: bool = True) -> List[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"image directory not found: {root}")
    it = root.rglob("*") if recursive else root.glob("*")
    return sorted(p for p in it if p.suffix.lower() in IMG_EXTS)


def _strip_noise_suffix(stem: str, suffix: str = "_noise") -> str:
    return stem[: -len(suffix)] if stem.lower().endswith(suffix) else stem


def to_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    """BGR HWC uint8 -> RGB CHW float32 in [0,1]."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float().div_(255.0)


def to_image(tensor: torch.Tensor) -> np.ndarray:
    """RGB CHW float32 -> BGR HWC uint8."""
    arr = tensor.detach().clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
    return cv2.cvtColor((arr * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)


def _augment(noisy: np.ndarray, clean: np.ndarray, rng: random.Random):
    if rng.random() < 0.5:
        noisy, clean = noisy[:, ::-1], clean[:, ::-1]
    if rng.random() < 0.2:
        noisy, clean = noisy[::-1], clean[::-1]
    k = rng.randint(0, 3)
    if k:
        noisy, clean = np.rot90(noisy, k), np.rot90(clean, k)
    return np.ascontiguousarray(noisy), np.ascontiguousarray(clean)


def _paired_crop(noisy: np.ndarray, clean: np.ndarray, size: int, rng: random.Random):
    h, w = clean.shape[:2]
    if h < size or w < size:
        pad_h, pad_w = max(0, size - h), max(0, size - w)
        noisy = np.pad(noisy, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        clean = np.pad(clean, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        h, w = clean.shape[:2]
    y, x = rng.randint(0, h - size), rng.randint(0, w - size)
    return noisy[y : y + size, x : x + size], clean[y : y + size, x : x + size]


class DenoiseDataset(Dataset):
    """Yields ``(noisy, clean)`` tensor pairs, both RGB CHW float in [0,1]."""

    def __init__(
        self,
        clean_dir: Path,
        noisy_dir: Optional[Path] = None,
        patch_size: int = 256,
        preset: str = "cctv",
        augment: bool = True,
        repeat: int = 1,
        is_train: bool = True,
        max_images: Optional[int] = None,
        degradation: Optional[DegradationConfig] = None,
    ):
        self.clean_paths = list_images(Path(clean_dir))
        if max_images:
            self.clean_paths = self.clean_paths[:max_images]
        if not self.clean_paths:
            raise RuntimeError(f"no images found under {clean_dir}")

        self.noisy_dir = Path(noisy_dir) if noisy_dir else None
        self.noisy_lookup = {}
        if self.noisy_dir:
            for p in list_images(self.noisy_dir):
                self.noisy_lookup[_strip_noise_suffix(p.stem).lower()] = p
            missing = [p for p in self.clean_paths if p.stem.lower() not in self.noisy_lookup]
            if missing:
                print(
                    f"[dataset] {len(missing)} clean images have no noisy counterpart; "
                    "they will be degraded synthetically as a fallback."
                )

        self.patch_size = patch_size
        self.augment = augment and is_train
        self.is_train = is_train
        self.repeat = max(1, repeat)
        self.base_degradation = degradation or PRESETS.get(preset, PRESETS["cctv"])
        self.preset = preset

    def __len__(self) -> int:
        return len(self.clean_paths) * self.repeat

    def _degradation_for(self, index: int) -> DegradationConfig:
        """Train: a fresh random chain each time. Val: seeded by index, so the
        validation score is comparable across epochs."""
        import copy

        cfg = copy.copy(self.base_degradation)
        cfg._rng = random.Random(None if self.is_train else 1234 + index)
        return cfg

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        real_index = index % len(self.clean_paths)
        clean_path = self.clean_paths[real_index]

        clean = cv2.imread(str(clean_path), cv2.IMREAD_COLOR)
        if clean is None:                                   # skip unreadable files
            return self[(index + 1) % len(self)]

        noisy = None
        noisy_path = self.noisy_lookup.get(clean_path.stem.lower())
        if noisy_path is not None:
            noisy = cv2.imread(str(noisy_path), cv2.IMREAD_COLOR)
            if noisy is not None and noisy.shape != clean.shape:
                noisy = cv2.resize(noisy, (clean.shape[1], clean.shape[0]))
        if noisy is None:
            noisy = degrade(clean, self._degradation_for(index))

        rng = random.Random(None if self.is_train else 999 + index)
        if self.patch_size:
            noisy, clean = _paired_crop(noisy, clean, self.patch_size, rng)
        if self.augment:
            noisy, clean = _augment(noisy, clean, rng)

        return to_tensor(noisy), to_tensor(clean)


class FullImageDataset(Dataset):
    """Whole images, no cropping. For final evaluation; batch size must be 1
    unless every image is the same size."""

    def __init__(
        self,
        clean_dir: Path,
        noisy_dir: Optional[Path] = None,
        preset: str = "cctv",
        max_side: Optional[int] = 1024,
        max_images: Optional[int] = None,
    ):
        self.clean_paths = list_images(Path(clean_dir))
        if max_images:
            self.clean_paths = self.clean_paths[:max_images]
        self.noisy_lookup = {}
        if noisy_dir:
            for p in list_images(Path(noisy_dir)):
                self.noisy_lookup[_strip_noise_suffix(p.stem).lower()] = p
        self.base_degradation = PRESETS.get(preset, PRESETS["cctv"])
        self.max_side = max_side

    def __len__(self) -> int:
        return len(self.clean_paths)

    def __getitem__(self, index: int):
        import copy

        path = self.clean_paths[index]
        clean = cv2.imread(str(path), cv2.IMREAD_COLOR)

        if self.max_side:
            h, w = clean.shape[:2]
            if max(h, w) > self.max_side:
                s = self.max_side / max(h, w)
                clean = cv2.resize(clean, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

        noisy_path = self.noisy_lookup.get(path.stem.lower())
        noisy = cv2.imread(str(noisy_path), cv2.IMREAD_COLOR) if noisy_path else None
        if noisy is not None and noisy.shape != clean.shape:
            noisy = cv2.resize(noisy, (clean.shape[1], clean.shape[0]))
        if noisy is None:
            cfg = copy.copy(self.base_degradation)
            cfg._rng = random.Random(1234 + index)
            noisy = degrade(clean, cfg)

        return to_tensor(noisy), to_tensor(clean), path.name


def build_loaders(
    train_dir: Path,
    val_dir: Optional[Path] = None,
    noisy_train_dir: Optional[Path] = None,
    noisy_val_dir: Optional[Path] = None,
    patch_size: int = 256,
    batch_size: int = 8,
    workers: int = 4,
    preset: str = "cctv",
    val_split: float = 0.05,
    repeat: int = 1,
):
    """Returns ``(train_loader, val_loader)``. If ``val_dir`` is omitted a
    deterministic slice of the training images is held out."""
    from torch.utils.data import DataLoader, Subset

    train_ds = DenoiseDataset(
        train_dir, noisy_train_dir, patch_size=patch_size, preset=preset,
        augment=True, is_train=True, repeat=repeat,
    )

    if val_dir:
        val_ds = DenoiseDataset(
            val_dir, noisy_val_dir, patch_size=patch_size, preset=preset,
            augment=False, is_train=False,
        )
    else:
        n = len(train_ds.clean_paths)
        n_val = max(1, int(n * val_split))
        indices = list(range(n))
        random.Random(0).shuffle(indices)
        val_indices, train_indices = indices[:n_val], indices[n_val:]

        full_val = DenoiseDataset(
            train_dir, noisy_train_dir, patch_size=patch_size, preset=preset,
            augment=False, is_train=False,
        )
        val_ds = Subset(full_val, val_indices)
        train_ds = Subset(train_ds, [i + k * n for k in range(train_ds.repeat) for i in train_indices])

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=workers,
        pin_memory=pin, drop_last=True, persistent_workers=workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=max(1, batch_size // 2), shuffle=False, num_workers=max(0, workers // 2),
        pin_memory=pin, persistent_workers=workers > 1,
    )
    return train_loader, val_loader
