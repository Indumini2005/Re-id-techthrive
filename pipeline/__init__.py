"""Five-stage localized inference pipeline, multi-camera.

    CAM_A ─┐
           ├─ pair in time ─► denoise ─► crop ─► restore ─► embed ─► fuse ─► match
    CAM_B ─┘

Each stage is a lazily-constructed singleton so importing this package costs
nothing; a model only loads when its stage is first used.
"""

from .fusion import CrossViewFusion, FusedTrack, ViewObservation, fuse_embeddings
from .sources import CameraStream, Frame, FramePair, SyncedCapture, cameras_from_args
from .stage1_denoise import ClassicalDenoiser, denoise_frame, get_denoiser, prepare_face
from .stage2_detect import PersonDetector, get_person_crop, get_person_crops
from .stage3_restore import FaceRestorer, restore_yolo_crop
from .stage4_embed import ArcFaceEmbedder, extract_embedding
from .stage5_match import ThreatIndex, query_cctv_crop, register_known_subject

__all__ = [
    # capture
    "CameraStream",
    "Frame",
    "FramePair",
    "SyncedCapture",
    "cameras_from_args",
    # stages
    "ClassicalDenoiser",
    "denoise_frame",
    "get_denoiser",
    "prepare_face",
    "PersonDetector",
    "get_person_crop",
    "get_person_crops",
    "FaceRestorer",
    "restore_yolo_crop",
    "ArcFaceEmbedder",
    "extract_embedding",
    "ThreatIndex",
    "query_cctv_crop",
    "register_known_subject",
    # fusion
    "CrossViewFusion",
    "FusedTrack",
    "ViewObservation",
    "fuse_embeddings",
]
