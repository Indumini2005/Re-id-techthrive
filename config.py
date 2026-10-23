"""Central configuration for the edge surveillance pipeline.

Every path is resolved relative to this file so the project can be copied
onto the Jetson without editing anything but the values below.

Hardware profiles live at the bottom of this file. The target is the
**original Jetson Nano** (Maxwell, 128 CUDA cores, 4 GB shared memory,
compute capability 5.3), which differs from an Orin in ways that change the
design, not just the numbers:

* no INT8 - compute 5.3 has no DP4A instruction, so FP16 is the floor;
* ~470 GFLOPS FP16 against the Orin Nano's ~20 TFLOPS, roughly 40x less;
* 4 GB shared between CPU and GPU, so not every model fits at once;
* 5W / 10W power modes, not 7-25W;
* JetPack 4.6 is the final release, pinning the device to Python 3.6.

Run ``python3 check_device.py`` on the board to see what it actually has.
Everything under ``pipeline/`` is written to import on Python 3.6; the
``training/`` package targets the workstation and does not have to be.
"""

import os
from pathlib import Path

ROOT = Path(__file__).parent.resolve()

# --- Directories -----------------------------------------------------------
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"          # training checkpoints / logs
DB_DIR = ROOT / "vector_db"       # FAISS index + label store
CAPTURE_DIR = ROOT / "captures"   # frames saved on a positive match

for _d in (MODELS_DIR, DATA_DIR, RUNS_DIR, DB_DIR, CAPTURE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Stage 1: NAFNet denoiser ---------------------------------------------
NAFNET_ONNX = MODELS_DIR / "nafnet.onnx"
NAFNET_ENGINE = MODELS_DIR / "nafnet.engine"
NAFNET_CKPT = MODELS_DIR / "nafnet.pth"

# Tiling. Overridden by the hardware profile below: a Nano wants small tiles
# and little overlap, because every extra tile is another full forward pass.
TILE_SIZE = 256
TILE_OVERLAP = 128

# Model width/depth. Must match whatever you trained and exported.
# The "nano" profile trains a width-16, single-block variant instead; see
# NAFNET_ARCHS and training/train_denoiser.py --arch.
NAFNET_WIDTH = 32
NAFNET_ENC_BLKS = [1, 1, 1, 8]
NAFNET_MIDDLE_BLKS = 1
NAFNET_DEC_BLKS = [1, 1, 1, 1]

# Named architectures, so training, export and inference cannot drift apart.
NAFNET_ARCHS = {
    # Paper-ish baseline. ~17M params. For an Orin or a workstation.
    "base": {"width": 32, "enc": [1, 1, 1, 8], "middle": 1, "dec": [1, 1, 1, 1]},
    # Half width, shallow bottleneck. ~1.1M params, roughly 10x cheaper.
    "small": {"width": 16, "enc": [1, 1, 1, 2], "middle": 1, "dec": [1, 1, 1, 1]},
    # What the original Nano can actually sustain. ~0.3M params.
    "nano": {"width": 16, "enc": [1, 1, 1], "middle": 1, "dec": [1, 1, 1]},
}

# --- Stage 2: YOLOv8 person detection -------------------------------------
# Three backends, picked by file extension:
#   .engine -> TensorRT via ultralytics   (needs Python 3.8+)
#   .pt     -> PyTorch via ultralytics    (needs Python 3.8+)
#   .onnx   -> ONNX Runtime, self-contained postprocessing
# On JetPack 4.6 (Python 3.6) ultralytics will not install, so the ONNX
# backend is the only one that works on the Nano. Export the .onnx on your
# workstation with export/export_yolo.py --format onnx.
YOLO_WEIGHTS = MODELS_DIR / "yolov8n.onnx"
YOLO_IMGSZ = 640
YOLO_CONF = 0.5
YOLO_IOU = 0.45
YOLO_PERSON_CLASS = 0          # COCO 'person'
YOLO_CROP_MARGIN = 0.05        # 5% context padding around the box
YOLO_MAX_SUBJECTS = 3          # subjects tracked per frame

# --- Stage 3: GFPGAN face restoration -------------------------------------
GFPGAN_WEIGHTS = MODELS_DIR / "GFPGANv1.3.pth"
GFPGAN_UPSCALE = 2
GFPGAN_ARCH = "clean"
GFPGAN_CHANNEL_MULTIPLIER = 2

# When to spend time on GFPGAN:
#   always   - restore every detected subject (the architecture diagram)
#   escalate - embed the cheap denoised crop first; only restore when that
#              preliminary score is close enough to matter, then re-embed
#   never    - skip stage 3 entirely, feed the crop straight to ArcFace
# GFPGAN is a StyleGAN2 decoder and costs seconds per face on Maxwell, so
# "escalate" is the only setting that keeps the Nano responsive. Most frames
# contain nobody enrolled, and those are rejected for the price of one
# ArcFace pass.
RESTORE_MODE = os.environ.get("EDGE_RESTORE_MODE", "escalate")

# Preliminary cosine (crop vs gallery) above which restoration is worth it.
# Set it well below MATCH_THRESHOLD: the un-restored crop scores lower than
# the restored face would, so this is a deliberately generous pre-filter.
ESCALATE_ABOVE = 0.40

# --- Stage 4: ArcFace embedding -------------------------------------------
ARCFACE_ONNX = MODELS_DIR / "arcface_w600k_r50.onnx"
ARCFACE_INPUT = 112
EMBEDDING_DIM = 512

# --- Stage 5: FAISS vector DB ---------------------------------------------
FAISS_INDEX = DB_DIR / "threats.index"
FAISS_LABELS = DB_DIR / "threats_labels.json"
MATCH_THRESHOLD = 0.65         # cosine similarity for a positive ID
MATCH_TOP_K = 3

# --- Multi-camera fusion ---------------------------------------------------
# Two cameras on the same area. The pipeline pairs their frames in time, then
# compares the two views of each subject against each other before querying
# the gallery.
CAMERAS = [
    {
        "name": "CAM_A",
        "url": os.environ.get("EDGE_RTSP_A", "rtsp://192.168.1.101:554/stream"),
        "codec": "h264",
    },
    {
        "name": "CAM_B",
        "url": os.environ.get("EDGE_RTSP_B", "rtsp://192.168.1.102:554/stream"),
        "codec": "h264",
    },
]

CAPTURE_FPS = 30               # what the cameras deliver (and file replay rate)

# Frames further apart than this are flagged as skewed rather than fused. At
# 30 fps the inter-frame interval is 33ms, so 50ms tolerates roughly one
# frame of drift between two free-running cameras.
SYNC_WINDOW_MS = 50.0

# Cosine between two LIVE views of the same person. Deliberately lower than
# MATCH_THRESHOLD: both vectors come from degraded CCTV faces, whereas a
# gallery comparison has one clean side.
CROSS_VIEW_THRESHOLD = 0.45

FUSION_MODE = "mean"           # mean | max | first
REQUIRE_VIEW_AGREEMENT = False # both views must independently name the same ID
REQUIRE_BOTH_VIEWS = False     # suppress alerts seen by only one camera

# --- Runtime ---------------------------------------------------------------
# "tensorrt" | "cuda" | "cpu".  Picked up by every ONNX stage.
EXEC_PROVIDER = os.environ.get("EDGE_PROVIDER", "cuda")

RTSP_URL = os.environ.get("EDGE_RTSP", "rtsp://192.168.1.100:554/stream")
RTSP_LATENCY_MS = 200

# Run the heavy stages every Nth frame pair; the rest are passthrough display
# frames.
DENOISE_EVERY_N = 1
DETECT_EVERY_N = 2

# "frame" denoises the whole frame (matches the architecture diagram);
# "crop" detects first and denoises only the person crop. Detection tolerates
# noise, recognition does not, so "crop" costs little accuracy and saves most
# of the work - mandatory on a first-generation Nano.
DENOISE_SCOPE = os.environ.get("EDGE_DENOISE_SCOPE", "crop")

# Downscale each frame before any inference. The Nano cannot process 1080p
# at a useful rate; 640x360 keeps faces large enough for ArcFace after the
# GFPGAN upscale. 0 disables resizing.
PROCESS_WIDTH = 0

HEADLESS = bool(int(os.environ.get("EDGE_HEADLESS", "0")))
SAVE_MATCHES = True


# --- Hardware profiles -----------------------------------------------------
# Applied at import time, overriding the module-level defaults above. Choose
# with EDGE_PROFILE, or let detect_profile() read /proc/device-tree/model.
PROFILES = {
    # Original Jetson Nano: Maxwell, 128 cores, 4 GB shared, no INT8.
    "nano": {
        "TILE_SIZE": 128,
        "TILE_OVERLAP": 32,
        "NAFNET_ARCH": "nano",
        "DENOISE_SCOPE": "crop",
        "PROCESS_WIDTH": 640,
        "RESTORE_MODE": "escalate",
        "YOLO_IMGSZ": 416,
        "YOLO_MAX_SUBJECTS": 2,
        "DETECT_EVERY_N": 1,
        "CAPTURE_FPS": 30,
        "TRT_PRECISION": "fp16",     # Maxwell has no INT8 path
        "POWER_BUDGET_W": 10.0,      # MAXN mode on a 5V/4A barrel supply
    },
    # Xavier NX / AGX: INT8 and DLA available.
    "xavier": {
        "TILE_SIZE": 256,
        "TILE_OVERLAP": 64,
        "NAFNET_ARCH": "small",
        "DENOISE_SCOPE": "crop",
        "PROCESS_WIDTH": 1280,
        "RESTORE_MODE": "escalate",
        "YOLO_IMGSZ": 640,
        "YOLO_MAX_SUBJECTS": 3,
        "DETECT_EVERY_N": 1,
        "CAPTURE_FPS": 30,
        "TRT_PRECISION": "int8",
        "POWER_BUDGET_W": 15.0,
    },
    # Orin Nano / NX.
    "orin": {
        "TILE_SIZE": 256,
        "TILE_OVERLAP": 128,
        "NAFNET_ARCH": "base",
        "DENOISE_SCOPE": "frame",
        "PROCESS_WIDTH": 0,
        "RESTORE_MODE": "always",
        "YOLO_IMGSZ": 640,
        "YOLO_MAX_SUBJECTS": 3,
        "DETECT_EVERY_N": 1,
        "CAPTURE_FPS": 30,
        "TRT_PRECISION": "int8",
        "POWER_BUDGET_W": 15.0,
    },
    # Development machine: no constraints worth modelling.
    "workstation": {
        "TILE_SIZE": 256,
        "TILE_OVERLAP": 128,
        "NAFNET_ARCH": "base",
        "DENOISE_SCOPE": "frame",
        "PROCESS_WIDTH": 0,
        "RESTORE_MODE": "always",
        "YOLO_IMGSZ": 640,
        "YOLO_MAX_SUBJECTS": 3,
        "DETECT_EVERY_N": 1,
        "CAPTURE_FPS": 30,
        "TRT_PRECISION": "fp16",
        "POWER_BUDGET_W": 0.0,
    },
}


def detect_profile():
    """Identify the board from the device tree, defaulting to workstation."""
    override = os.environ.get("EDGE_PROFILE")
    if override:
        return override if override in PROFILES else "workstation"
    try:
        with open("/proc/device-tree/model", "rb") as fh:
            model = fh.read().decode("utf-8", "ignore").lower()
    except Exception:
        return "workstation"
    if "orin" in model:
        return "orin"
    if "xavier" in model:
        return "xavier"
    if "nano" in model or "tx1" in model or "tx2" in model:
        return "nano"
    return "workstation"


PROFILE = detect_profile()


def apply_profile(name=None):
    """Overwrite the module-level settings with a profile's values."""
    global PROFILE
    name = name or PROFILE
    settings = PROFILES.get(name)
    if not settings:
        return {}
    PROFILE = name
    globals().update(settings)

    arch = NAFNET_ARCHS.get(settings.get("NAFNET_ARCH", "base"), NAFNET_ARCHS["base"])
    globals().update({
        "NAFNET_WIDTH": arch["width"],
        "NAFNET_ENC_BLKS": arch["enc"],
        "NAFNET_MIDDLE_BLKS": arch["middle"],
        "NAFNET_DEC_BLKS": arch["dec"],
    })

    # Environment variables still win, so a profile never traps you.
    if "EDGE_DENOISE_SCOPE" in os.environ:
        globals()["DENOISE_SCOPE"] = os.environ["EDGE_DENOISE_SCOPE"]
    if "EDGE_RESTORE_MODE" in os.environ:
        globals()["RESTORE_MODE"] = os.environ["EDGE_RESTORE_MODE"]
    return settings


apply_profile(PROFILE)

# Compute capability 6.1 is the floor for TensorRT INT8 (the DP4A integer
# dot-product instruction). Maxwell is 5.3, so the Nano is FP16-only.
INT8_MIN_COMPUTE = 6.1
INT8_AVAILABLE = PROFILE not in ("nano",)


def provider_list(provider: str = None):
    """ONNX Runtime provider list for the requested backend, with fallbacks."""
    provider = provider or EXEC_PROVIDER
    if provider == "tensorrt":
        return [
            (
                "TensorrtExecutionProvider",
                {
                    "trt_fp16_enable": True,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": str(MODELS_DIR / "trt_cache"),
                },
            ),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
    if provider == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]
