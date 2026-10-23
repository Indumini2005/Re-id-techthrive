# Privacy-First Edge Surveillance Pipeline

A fully localized neural pipeline for CCTV threat matching on an **NVIDIA
Jetson Nano (original, Maxwell, 4 GB)**. **Two RTSP cameras at 30 fps**,
paired in time and fused for multi-angle identification. No cloud, no
transmission, no stored raw video.

> **Hardware note.** This targets the first-generation Jetson Nano
> (B01, 945-13450), *not* an Orin Nano. That is a 40× difference in compute
> and it changes the design, not just the numbers. See
> [Hardware reality](#hardware-reality) before quoting any spec, and run
> `python3 check_device.py` on the board.

```
CAM_A (30fps) ─┐
               ├─ pair in time ─► 1. denoise ─► 2. crop ─► 3. restore ─► 4. embed ─┐
CAM_B (30fps) ─┘                                                                    │
                                                        cross-view association ◄────┤
                                                             fuse both angles ──────┤
                                                        5. FAISS cosine match ◄─────┘
                                                                    │
                                                            operator alert
```

Both cameras are read on their own threads at native rate. Each inference
cycle takes the newest frame from the lead camera and the temporally closest
frame from the other. The two embeddings are first compared **against each
other** to establish they show the same person, then averaged into one
gallery query.

---

## Hardware reality

Run `python3 check_device.py` on the board. Summary of what the original
Nano is, against what the architecture slides claim:

| | Slide claims (Orin Nano) | Your board (Jetson Nano B01) |
|---|---|---|
| GPU | Ampere, 1024 cores | **Maxwell, 128 cores** |
| Compute | 40 TOPS | **~0.47 TFLOPS FP16** (~80× less) |
| Memory | 8 GB | **4 GB, shared CPU+GPU** |
| Power | 10–15 W | **5 W / 10 W (MAXN)** |
| INT8 | yes | **no — compute 5.3 has no DP4A** |
| JetPack | 5/6, Python 3.8+ | **4.6 (final), Python 3.6** |

Three of these change the code, not just the expectations:

**INT8 is unavailable.** TensorRT INT8 needs compute capability 6.1+ for the
DP4A integer dot-product instruction. Maxwell is 5.3. `export_tensorrt.py`
now refuses INT8 with a diagnostic rather than letting TensorRT silently fall
back to FP16 while you believe you are measuring INT8. **FP16 is the fastest
path on this board, not a fallback** — fix the slide that says "FP32 to INT8".

**Python 3.6 on the device.** JetPack 4.6 is the last release for this board
and ships Python 3.6.9. So:

- Everything under `pipeline/`, plus `config.py`, `main_pipeline.py`,
  `enroll.py`, `benchmark.py` and `denoise.py`, is written to import on 3.6.
  No `from __future__ import annotations`, no f-string `=`, no walrus.
  `training/` targets your workstation and is not constrained.
- **`ultralytics` requires Python 3.8+ and will not install on the Nano.**
  Stage 2 therefore has a second backend: export YOLOv8 to ONNX on your
  workstation and run it under ONNX Runtime, with the letterbox and NMS
  implemented in `pipeline/stage2_detect.py`. This is the default.
- `gfpgan`/`basicsr` also require 3.7+. Stage 3 is optional at runtime and
  degrades to a passthrough; see the restoration policy below.
- `pip install dataclasses` — a 3.6 backport is needed.

**4 GB shared memory.** All four models resident at once will not fit
alongside two decoded 1080p streams. Run headless (`--headless`) to free the
desktop's ~500 MB, add swap, and keep GFPGAN out of the hot path.

### The restoration policy

GFPGAN is a StyleGAN2 decoder — seconds per face on Maxwell. Running it on
every detected person is not viable, so `--restore_mode` controls when it is
worth paying for:

| Mode | Behaviour |
|---|---|
| `always` | Restore every subject. The architecture diagram. Orin-class hardware. |
| `escalate` | **Default on Nano.** Embed the cheap denoised crop, query the gallery, and only restore when that preliminary score clears `--escalate_above` (0.40). Then re-embed the restored face. |
| `never` | Skip stage 3; feed the crop straight to ArcFace. |

`escalate` works because almost every frame contains nobody enrolled. Those
are rejected for the price of one ArcFace pass, and the expensive model only
runs on plausible candidates. Alerts mark which embeddings were escalated.

### Profiles

`config.py` carries per-board profiles (`nano`, `xavier`, `orin`,
`workstation`), auto-selected from `/proc/device-tree/model` and overridable
with `EDGE_PROFILE`. The `nano` profile sets 128px tiles, 32px overlap, the
`nano` model architecture, `--denoise_scope crop`, 640px processing width,
`--restore_mode escalate`, YOLO at 416px and FP16-only TensorRT.

---

## Layout

| Path | What it does |
|---|---|
| `config.py` | Every path, threshold and runtime switch, plus hardware profiles. Start here. |
| `check_device.py` | Probe the board: GPU, memory, INT8/DLA support, what will actually run. |
| `hub/server.py` | Hub: pairs nodes, fuses, matches, serves the dashboard. |
| `hub/static/dashboard.html` | The operator UI. |
| `hub/protocol.py` | Node↔hub wire format and clock-offset correction. |
| `node_agent.py` | Runs on each capture laptop; sends embeddings, not video. |
| `pipeline/vectors.py` | Pure-numpy vector helpers (keeps the hub OpenCV-free). |
| `main_pipeline.py` | The real-time dual-camera loop. |
| `denoise.py` | Batch denoising CLI for offline folders. |
| `enroll.py` | Add / capture / compare / list / remove identities. |
| `benchmark.py` | Per-stage latency, FPS and Jetson power draw. |
| `pipeline/sources.py` | Threaded multi-camera RTSP capture and frame pairing. |
| `pipeline/fusion.py` | Cross-view association and multi-angle identification. |
| `pipeline/stage1_denoise.py` | NAFNet inference with overlap-blended tiling. |
| `pipeline/stage2_detect.py` | YOLOv8 person detection and cropping. |
| `pipeline/stage3_restore.py` | GFPGAN face reconstruction. |
| `pipeline/stage4_embed.py` | ArcFace 512-D embeddings. |
| `pipeline/stage5_match.py` | FAISS index, persistence, cosine matching. |
| `pipeline/utils.py` | Tiler, stage timer, HUD drawing. |
| `training/nafnet.py` | NAFNet architecture (PyTorch). |
| `training/degradations.py` | Synthetic CCTV degradation chain. |
| `training/dataset.py` | Paired and synthetic datasets. |
| `training/losses.py` | PSNR / Charbonnier / edge losses, PSNR + SSIM metrics. |
| `training/train_denoiser.py` | The training loop. |
| `training/evaluate.py` | PSNR / SSIM scoring of a checkpoint or ONNX graph. |
| `training/make_dataset.py` | Build pairs from image folders or video clips. |
| `export/export_onnx.py` | Checkpoint → ONNX. |
| `export/export_tensorrt.py` | ONNX → TensorRT engine (FP16; refuses INT8 on Maxwell). |
| `export/export_yolo.py` | YOLOv8 → ONNX (for the Nano) or TensorRT. |

`yolocropping.py`, `gpfgan.py` and `arcface_faiss.py` are thin shims kept so
the original imports still resolve.

---

## Install

The work splits across two machines. **Train and export on a workstation;
the Nano only runs inference.**

### Workstation (training + export)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Use Python 3.10 or 3.11. The default interpreter in this workspace is 3.14,
and neither `basicsr` nor `gfpgan` builds there.

`basicsr` is unmaintained and breaks in two ways: it needs `numpy<2`, and it
imports `torchvision.transforms.functional_tensor`, removed in torchvision
0.17. Either install `torchvision<0.17`, or patch the one line in
`basicsr/data/degradations.py`:

```python
# from torchvision.transforms.functional_tensor import rgb_to_grayscale
from torchvision.transforms.functional import rgb_to_grayscale
```

### Jetson Nano (inference only)

JetPack 4.6, Python 3.6. Install the **NVIDIA JetPack wheels** for
`onnxruntime-gpu` — the PyPI version has no CUDA support for aarch64.
TensorRT and `pycuda` come from JetPack itself.

```bash
sudo apt install python3-opencv libopenblas-dev
pip3 install -r requirements-jetson.txt
# onnxruntime-gpu: NVIDIA's JetPack 4.6 wheel, not PyPI
python3 check_device.py
```

Do **not** try to install `ultralytics` or `gfpgan` here; both require
Python 3.8+. Stage 2 runs from ONNX instead, and stage 3 is optional.

### Models to copy into `models/`

| File | Source | Needed on |
|---|---|---|
| `nafnet.onnx` | **you train this** — see below | Nano |
| `yolov8n.onnx` | `export/export_yolo.py --format onnx --imgsz 416` | Nano |
| `arcface_w600k_r50.onnx` | insightface `buffalo_l` pack | Nano |
| `GFPGANv1.3.pth` | TencentARC/GFPGAN releases | optional |

---

## Training the denoiser

This is the one model you train yourself; the other three are pretrained.

### 1. Get data

You need **sharp** images. The degradation chain synthesizes the noisy side,
so quality of the clean reference is what matters — never feed it footage
that is already soft.

```powershell
# From your own video clips (sharp frames only, 2 per second)
python -m training.make_dataset video --src clips\ --out data\train --fps 2

# Or from a folder of stills
python -m training.make_dataset images --src data\raw --out data\train
```

That writes `data/train/clean/` and `data/train/noisy/`. You can also skip
this step entirely and let the degradation run on the fly, which gives the
model fresh corruption every epoch:

```powershell
python -m training.train_denoiser --train_dir data\clean --preset cctv
```

### 2. Train

```powershell
python -m training.train_denoiser `
    --train_dir data\train\clean `
    --noisy_train_dir data\train\noisy `
    --epochs 200 --batch_size 8 --patch_size 256 `
    --width 32 --lr 1e-3 --preset cctv
```

Output lands in `runs/nafnet/`: `best.pth`, `last.pth`, a `log.csv`, and
sample strips (`noisy | denoised | clean`) under `samples/` every 5 epochs.
Look at those strips — the loss curve will not tell you that the model has
started smearing faces.

**On a Jetson Nano, train `--arch nano`.** The width-32 baseline is ~17M
parameters and will not run usefully on 128 Maxwell cores:

```powershell
python -m training.train_denoiser --train_dir data\train\clean `
    --noisy_train_dir data\train\noisy --arch nano --epochs 300 --patch_size 128
```

| `--arch` | Params | For |
|---|---|---|
| `base` | ~17M | Orin / workstation |
| `small` | ~1.1M | Xavier |
| `nano` | ~0.3M | **this board** |

The architecture is recorded in the checkpoint, so export and inference pick
it up automatically — a `nano` checkpoint cannot be silently loaded into a
`base` graph.

Useful knobs:

- `--arch small` — if `nano` leaves visible noise and you can afford the
  frame time. Measure with `benchmark.py` before deciding.
- `--preset harsh` — heavier noise, low light, aggressive compression. Train
  on this if your cameras are genuinely bad; it costs PSNR on easy footage.
- `--loss psnr --edge_weight 0` — the stock NAFNet recipe. The default adds a
  small edge term because stages 3–5 need facial detail preserved, and pure
  PSNR optimization is happy to blur it away.
- `--repeat 4` — four random patches per image per epoch, for small datasets.
- `--resume runs\nafnet\last.pth` — pick training back up.

### 3. Evaluate

```powershell
python -m training.evaluate --ckpt runs\nafnet\best.pth --data_dir data\val --save_dir runs\cmp
```

Reports PSNR/SSIM before and after, per image and in aggregate. Anything
below about +3 dB gain means the degradation preset does not match your real
cameras — retune `training/degradations.py` rather than training longer.

### 4. Export

```powershell
# Workstation: NAFNet -> ONNX, and YOLOv8 -> ONNX for the device
python export\export_onnx.py --ckpt runs\nafnet\best.pth --tile 128 --opset 12 --simplify
python export\export_yolo.py --format onnx --imgsz 416
```

Copy both `.onnx` files plus the ArcFace model to the Nano's `models/`, then
optionally build TensorRT engines **on the device**:

```bash
python3 export/export_tensorrt.py --onnx models/nafnet.onnx --precision fp16 --benchmark
```

**No INT8 on this board.** Compute capability 5.3 has no DP4A instruction, so
`--precision int8` exits with a diagnostic rather than letting TensorRT
quietly fall back. FP16 is the fastest available path here. Keep `--opset 12`
— JetPack 4.6 ships ONNX Runtime 1.11, which does not understand newer opsets.

Then verify the exported graph did not lose anything:

Then verify the exported graph did not lose anything:

```powershell
python -m training.evaluate --onnx models\nafnet.onnx --data_dir data\val
```

---

## Running the two cameras

Set the URLs once, in `config.CAMERAS` or via the environment:

```powershell
$env:EDGE_RTSP_A = "rtsp://192.168.1.101:554/stream"
$env:EDGE_RTSP_B = "rtsp://192.168.1.102:554/stream"
$env:EDGE_PROVIDER = "tensorrt"     # or "cuda", "cpu"
```

```powershell
# Enroll from both angles at once - stand the subject in front of both cameras
python enroll.py capture --identity Subject_A --seconds 10 --save_faces
python enroll.py list

# Live, two cameras
python main_pipeline.py

# Explicit sources, optionally named
python main_pipeline.py --source LOBBY=rtsp://.../a --source GATE=rtsp://.../b

# Off-device development against two recordings
python main_pipeline.py --source a.mp4 --source b.mp4 --no_gstreamer --capture_fps 30

# Benchmark for the report
python benchmark.py --resolution 1280x720 --cameras 2 --iters 30 --power
```

Any stage whose weights are missing logs a warning and becomes a passthrough,
so you can bring the pipeline up one model at a time. A single camera still
works — pass one `--source` and fusion falls through to direct matching.

### Keys

| Key | Action |
|---|---|
| `q` | quit |
| `s` | snapshot the composed dual view to `captures/` |
| `d` | toggle between the raw and denoised display |

---

## Distributed mode: two laptops + a dashboard

An alternative to running both cameras off one board. Each capture laptop
processes its own camera and sends **only the embedding**; the hub pairs the
two nodes, fuses them, matches, and serves a live UI.

```
Laptop A ── CAM_A ──┐  ~2 KB per observation
                    ├──► HUB (UI laptop) ──► browser dashboard
Laptop B ── CAM_B ──┘      pair · fuse · match
```

Raw video never leaves the capture machine, which keeps the privacy claim
intact across the network rather than only on one box. It also survives a
weak link — 2 KB per observation instead of two video streams.

### Run it

On the **UI laptop** (stdlib only, no new dependencies):

```powershell
python -m hub.server --open
```

It prints its LAN address. Then on **each capture laptop**:

```powershell
python node_agent.py --name CAM_A --source 0 --hub http://192.168.1.50:8765
python node_agent.py --name CAM_B --source 0 --hub http://192.168.1.50:8765
```

`--source` takes a webcam index, a video file, or an `rtsp://` URL. Add
`--preview` to also show a local window on the capture machine.

Open `http://<ui-laptop-ip>:8765/` from anywhere on the LAN — phone, tablet,
projector. The page loads no external resources, so it works air-gapped.

### What the dashboard shows

- **The cross-view similarity**, large and centre: cosine between CAM_A's and
  CAM_B's live embeddings, on a ring with the threshold marked.
- Both camera previews with detection boxes, plus the restored face crops.
- Gallery verdict: identity, fused score, best single view, and **fusion gain**
  — the number that shows whether using two cameras actually helped.
- Live threshold sliders (both thresholds, and the fusion mode) that take
  effect immediately — useful for tuning during a demo.
- Per-stage latency, node health, clock offset, sync skew, alert log.

### Two things worth knowing

**Clock skew.** Two laptops don't share a clock and NTP often isn't running on
a demo network. Each node stamps its own capture time; the hub estimates each
node's offset as the *minimum* observed `arrival − captured` (minimum, because
transit delay is always ≥ 0, so the smallest difference is dominated by clock
offset) and corrects before pairing. The dashboard shows the estimate per node.

**Pairing window.** The hub defaults to a 150 ms sync window, much wider than
the 50 ms used for two cameras on one board. Nodes report a few times per
second, so the best achievable pairing is about half the send interval — 83 ms
at 6 obs/s. A partner older than `max_pair_age_ms` (1 s) is not used at all,
so a node dropping out degrades to single-view instead of fusing against a
stale frame.

### Degraded operation

Each node loads whatever it has and reports the rest as off in its startup
banner, so you can bring the demo up one piece at a time. With no gallery
enrolled, the hub still shows live cross-view similarity — which is the
headline number anyway. With no detector, the whole frame is treated as one
subject. Only stage 4 (ArcFace) is genuinely required; without it there are
no vectors to compare.

---

## How the two views are compared

Two different thresholds do two different jobs. Conflating them is the usual
way this kind of system goes wrong:

| Threshold | Default | Compares | Why |
|---|---|---|---|
| `--cross_view_threshold` | 0.45 | live CAM_A ↔ live CAM_B | Both sides are degraded CCTV faces, so similarity runs lower. This only decides *association* — same person or not. |
| `--threshold` | 0.65 | fused vector ↔ gallery | One side is a clean enrolled vector, so the usual ArcFace operating point applies. |

Association uses greedy highest-similarity-first matching, and refuses to put
two boxes from the same camera into one track. Fusion then averages the
member embeddings and renormalizes (`--fusion mean`), which is the
maximum-likelihood direction for vectors scattered around a shared identity —
extra angles strictly help.

Every alert prints the fused score, the cross-view similarity, and each
camera's independent verdict, so you can see whether fusion actually bought
anything or one view is carrying the result:

```
[!] 14:22:07  THREAT MATCH: Subject_A (fused cosine 0.845, views CAM_A+CAM_B)
    cross-view similarity 0.769 | views agree: True
      CAM_A: Subject_A 0.801
      CAM_B: Subject_A 0.772
    note: fusion +0.044 vs best single view
```

Two policy switches for higher-stakes deployments:

- `--require_agreement` — both cameras must independently name the same
  identity, not just the fused vector. Cuts false positives, costs recall.
- `--require_both_views` — suppress anyone only one camera can see.

To check a pair of frames offline, without running the live loop:

```powershell
python enroll.py compare --image_a cam_a.jpg --image_b cam_b.jpg
```

---

## About the 30 fps target

The cameras are read at 30 fps and every frame is timestamped, but **the
inference loop will not run at 30 fps**, and the design does not pretend
otherwise. On a 128-core Maxwell GPU shared between two camera streams, the
gap is large: expect on the order of **1–3 frame pairs per second** with the
`nano` profile, and well under 1 if you insist on whole-frame denoising at
1080p. Benchmark your own board rather than trusting any figure here.

What the pipeline does instead is drop cleanly. Capture threads keep a short
ring buffer and always hand back the newest frame; skipped frames are counted
and reported rather than queued, so the view never drifts into the past. The
HUD shows capture fps and inference fps separately, and `benchmark.py` tells
you the stride directly:

```
30 fps needs 33.3 ms/pair; this is 7.2x over budget.
The loop will process roughly every 8th frame pair (~88% of frames dropped).
Denoising is 84% of the per-camera cost - try --denoise_scope crop, INT8, or a narrower model.
```

If you need the gap closed, in order of what to try (the `nano` profile
already applies 1–4):

1. **`--process_width 640`** — downscale before any inference. Cost scales
   with pixel count, so 1080p → 640×360 is a 9× cut for no recognition loss
   once GFPGAN upscales the face crop anyway.
2. **`--denoise_scope crop`** — detect on the noisy frame, then denoise only
   the person box. Another 10–30× cut, because a person crop is a tiny
   fraction of the frame. Detection tolerates noise; recognition does not.
   This inverts stages 1 and 2 relative to the architecture slide, so measure
   it before committing to it in the writeup.
3. **`--restore_mode escalate`** — keep GFPGAN off the hot path.
4. **`--arch nano`** — retrain the denoiser at width 16.
5. **`--detect_every 2`** — run the stack on every second frame pair.
6. **`--headless`** — frees ~500 MB and the compositor's GPU time, which
   matters on a 4 GB shared-memory board.

INT8 is *not* on this list: Maxwell cannot do it.

Identification does not need 30 fps. A subject walking through frame is
visible for a second or more, which is dozens of chances at any usable rate;
what matters is that the two views are *paired*, not that every frame is
processed.

---

## Tuning for the power budget

The Nano has two power modes, not a 10–15 W range:

```bash
sudo nvpmodel -m 0      # MAXN, ~10 W - needs a 5V/4A barrel jack supply
sudo nvpmodel -m 1      # 5 W, two CPU cores disabled
sudo jetson_clocks      # pin clocks to maximum for benchmarking
```

Running MAXN from micro-USB will brown out under sustained inference; use the
barrel jack and fit the J48 jumper. The denoiser dominates — it is the only
stage touching every pixel, and with two cameras it runs twice per cycle. The
ordered list above is also the right order for power. `benchmark.py --power`
samples `tegrastats` and reports mean and peak watts against the budget in
the active profile.

---

## Privacy properties

- No network egress. Nothing in this repo opens an outbound socket except the
  RTSP read from the local camera network.
- `captures/` and `vector_db/` are the only places anything is written, both
  local. `.gitignore` excludes them.
- The vector DB stores 512-D embeddings, not images, unless you pass
  `--save_faces` during enrollment.
