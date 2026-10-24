"""Stage 3 - GFPGAN facial reconstruction. Optional, and often unavailable.

GFPGAN detects and aligns the face inside the YOLO crop (via facexlib), then
hallucinates high-frequency detail the CCTV sensor never recorded. The
output is a 512x512 aligned face, which is what ArcFace wants.

Two practical constraints, both of which usually decide the matter before
quality does:

**It will not install on a Jetson Nano.** gfpgan and basicsr both require
Python 3.7+, and JetPack 4.6 - the final release for that board - ships
3.6.9. Short of building a newer Python from source or exporting the model
to ONNX on another machine, stage 3 simply cannot run there.

**Its cost does not scale with the crop.** The face is aligned to a fixed
512x512 before the StyleGAN2 decoder runs, so a 40-pixel face costs the same
as a 400-pixel one. That is the opposite of stage 1, where restricting work
to the crop is what made the budget. There is no equivalent saving here: the
only lever is how often it runs, which is what ``--restore_mode escalate``
controls.

The pipeline degrades cleanly without it - ``prepare_face`` in
pipeline/stage1_denoise.py pads and Lanczos-resamples the crop to 112x112,
which is what ArcFace consumes either way. Measure recognition with and
without before assuming stage 3 is required.
"""

import threading
from pathlib import Path
from typing import List, Optional

import numpy as np

import config


class FaceRestorer:
    def __init__(
        self,
        model_path: Optional[Path] = None,
        upscale: int = config.GFPGAN_UPSCALE,
        arch: str = config.GFPGAN_ARCH,
        channel_multiplier: int = config.GFPGAN_CHANNEL_MULTIPLIER,
        device: Optional[str] = None,
    ):
        import sys

        try:
            from gfpgan import GFPGANer
        except ImportError as exc:
            if sys.version_info < (3, 7):
                raise ImportError(
                    "gfpgan requires Python 3.7+ and this is %d.%d. On JetPack "
                    "4.6 that cannot be fixed with pip - stage 3 is unavailable "
                    "on this board. Run with --restore_mode never; the pipeline "
                    "feeds ArcFace the Lanczos-resampled crop instead."
                    % sys.version_info[:2]
                )
            raise ImportError(
                "gfpgan is not installed (%s). Stage 3 is optional; "
                "--restore_mode never skips it." % exc
            )

        model_path = Path(model_path or config.GFPGAN_WEIGHTS)
        if not model_path.exists():
            raise FileNotFoundError(
                f"GFPGAN weights not found at {model_path}. Download "
                "GFPGANv1.3.pth into the models/ folder."
            )

        # bg_upsampler stays None: only the face crop matters, and
        # Real-ESRGAN on the background would blow the power budget.
        self.restorer = GFPGANer(
            model_path=str(model_path),
            upscale=upscale,
            arch=arch,
            channel_multiplier=channel_multiplier,
            bg_upsampler=None,
            device=device,
        )
        self._lock = threading.Lock()

    def restore(self, crop_bgr: np.ndarray, only_center_face: bool = True) -> Optional[np.ndarray]:
        """YOLO crop in -> single aligned, restored face out (or None)."""
        faces = self.restore_all(crop_bgr, only_center_face=only_center_face)
        return faces[0] if faces else None

    def restore_all(self, crop_bgr: np.ndarray, only_center_face: bool = False) -> List[np.ndarray]:
        if crop_bgr is None or crop_bgr.size == 0:
            return []
        with self._lock:
            _, restored_faces, _ = self.restorer.enhance(
                crop_bgr,
                has_aligned=False,
                only_center_face=only_center_face,
                paste_back=False,
            )
        return list(restored_faces) if restored_faces else []


_restorer: Optional[FaceRestorer] = None
_restorer_lock = threading.Lock()


def get_restorer(**kwargs) -> FaceRestorer:
    global _restorer
    if _restorer is None:
        with _restorer_lock:
            if _restorer is None:
                _restorer = FaceRestorer(**kwargs)
    return _restorer


def restore_yolo_crop(yolo_cropped_img: np.ndarray) -> Optional[np.ndarray]:
    """Stage-3 entry point. Matches the original API."""
    return get_restorer().restore(yolo_cropped_img)
