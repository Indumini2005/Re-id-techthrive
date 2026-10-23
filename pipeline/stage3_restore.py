"""Stage 3 - GFPGAN facial reconstruction.

GFPGAN detects and aligns the face inside the YOLO crop (via facexlib), then
hallucinates high-frequency detail that the CCTV sensor never recorded. The
output is a 512x512 aligned face, which is exactly what ArcFace wants.
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
        from gfpgan import GFPGANer

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
