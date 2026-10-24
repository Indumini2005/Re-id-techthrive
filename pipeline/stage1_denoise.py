"""Stage 1 - crop restoration.

Two backends behind one interface:

* **classical** (default) - median plus bilateral, in denoise_classical.py.
  No model, no training, a few milliseconds per crop.
* **neural** - an exported NAFNet graph under ONNX Runtime or TensorRT.
  Kept for machines that can afford it; it is not viable on a Jetson Nano
  at any useful rate.

The pipeline denoises *person crops*, not whole frames. Detection tolerates
noise; recognition does not. Restricting the work to the region that feeds
ArcFace cuts the pixel count by one to two orders of magnitude and is what
makes the whole system fit on a 128-core Maxwell GPU. The tiling machinery
below exists for the neural path, which still needs it on larger inputs.
"""

import threading
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

import config


class Denoiser:
    """Wraps the exported NAFNet graph.

    Args:
        model_path: .onnx file. Defaults to ``config.NAFNET_ONNX``.
        tile_size / overlap: 0 overlap disables blending; ``tile_size=0``
            disables tiling and runs the whole frame at once (fastest when
            the input is already small, e.g. a YOLO crop).
        batch_size: tiles pushed through the graph per ``run`` call.
        provider: "tensorrt" | "cuda" | "cpu".
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        tile_size: int = config.TILE_SIZE,
        overlap: int = config.TILE_OVERLAP,
        batch_size: int = 4,
        provider: Optional[str] = None,
        intra_op_threads: int = 0,
    ):
        import onnxruntime as ort

        from .utils import ImageTiler

        self.model_path = Path(model_path or config.NAFNET_ONNX)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"NAFNet weights not found at {self.model_path}. "
                "Train one with training/train_denoiser.py, then run "
                "export/export_onnx.py."
            )

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if intra_op_threads:
            opts.intra_op_num_threads = intra_op_threads

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=opts,
            providers=config.provider_list(provider),
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # A fixed batch dimension in the exported graph overrides batch_size.
        static_batch = self.session.get_inputs()[0].shape[0]
        self.batch_size = static_batch if isinstance(static_batch, int) else batch_size

        self.tile_size = tile_size
        self.tiler = ImageTiler(tile_size, overlap) if tile_size else None
        self._lock = threading.Lock()

    # -- inference ---------------------------------------------------------
    def _run_batch(self, batch_nchw: np.ndarray) -> np.ndarray:
        with self._lock:
            return self.session.run([self.output_name], {self.input_name: batch_nchw})[0]

    def _infer_tiles(self, tiles_rgb: List[np.ndarray]) -> List[np.ndarray]:
        """tiles_rgb: list of HWC uint8 RGB -> list of HWC float32 RGB in [0,1]."""
        out: List[np.ndarray] = []
        for i in range(0, len(tiles_rgb), self.batch_size):
            chunk = tiles_rgb[i : i + self.batch_size]
            arr = np.stack(chunk).astype(np.float32) / 255.0      # N,H,W,C
            arr = np.transpose(arr, (0, 3, 1, 2))                 # N,C,H,W

            # Pad a short final chunk so a statically-shaped graph still runs.
            short = self.batch_size - arr.shape[0]
            if short > 0 and isinstance(self.session.get_inputs()[0].shape[0], int):
                arr = np.concatenate([arr, np.repeat(arr[-1:], short, axis=0)], axis=0)

            pred = self._run_batch(np.ascontiguousarray(arr))
            pred = np.transpose(pred[: len(chunk)], (0, 2, 3, 1))  # N,H,W,C
            out.extend(pred)
        return out

    def __call__(self, img_bgr: np.ndarray) -> np.ndarray:
        return self.denoise(img_bgr)

    def denoise(self, img_bgr: np.ndarray) -> np.ndarray:
        """BGR uint8 in -> BGR uint8 out, same resolution."""
        if img_bgr is None or img_bgr.size == 0:
            return img_bgr
        if img_bgr.ndim == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

        if self.tiler is None:
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pred = self._infer_tiles([rgb])[0]
            bgr = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
            return np.clip(bgr * 255.0, 0, 255).astype(np.uint8)

        tiles, coords, padded_shape = self.tiler.split(img_bgr)
        tiles_rgb = [cv2.cvtColor(t, cv2.COLOR_BGR2RGB) for t in tiles]
        preds = self._infer_tiles(tiles_rgb)
        preds_bgr = [cv2.cvtColor(p, cv2.COLOR_RGB2BGR) for p in preds]
        return self.tiler.merge(preds_bgr, coords, padded_shape, img_bgr.shape[:2])


# -- module-level singleton, so the pipeline loop can just call a function ---
_denoiser = None
_denoiser_lock = threading.Lock()


def get_denoiser(**kwargs):
    """Stage-1 singleton, classical or neural per ``config.DENOISE_BACKEND``.

    Both expose ``.denoise(bgr) -> bgr``, so nothing downstream changes when
    the backend does. The default is classical: on a 128-core Maxwell it
    recovers most of the available quality for a few milliseconds per crop,
    against hundreds for any network worth running.
    """
    global _denoiser
    if _denoiser is not None:
        return _denoiser

    with _denoiser_lock:
        if _denoiser is not None:
            return _denoiser

        backend = kwargs.pop("backend", None) or config.DENOISE_BACKEND
        if backend == "classical":
            from .denoise_classical import ClassicalDenoiser
            _denoiser = ClassicalDenoiser(
                profile=kwargs.pop("profile", config.DENOISE_PROFILE),
                adaptive=kwargs.pop("adaptive", True),
                clahe=kwargs.pop("clahe", None),
                sharpen=kwargs.pop("sharpen", None),
            )
        elif backend == "none":
            from .denoise_classical import ClassicalDenoiser
            _denoiser = ClassicalDenoiser(profile="off")
        else:
            _denoiser = Denoiser(**kwargs)
    return _denoiser


def reset_denoiser():
    """Drop the singleton, so a different backend can be selected."""
    global _denoiser
    with _denoiser_lock:
        _denoiser = None


def denoise_frame(img_bgr: np.ndarray) -> np.ndarray:
    """Stage-1 entry point used by ``main_pipeline``."""
    return get_denoiser().denoise(img_bgr)


# Kept so the original flat-script call site still resolves.
process_single_image = denoise_frame
