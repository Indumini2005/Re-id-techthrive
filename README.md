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

On top of identification sits a scene layer: every person is **tracked**
with a Kalman filter, people who were never enrolled are **re-identified as
"Visitor #N"**, each person's **journey** across the cameras is recorded,
the system **predicts which camera they will appear at next** and scores
itself, and **dwell time / loitering** is measured. A **web dashboard**
shows all of it. See [Exhibition features](#exhibition-features).

---

## Stage 1 is classical, and crop-only

No model, no training, no weights file. A 3×3 median followed by a bilateral
filter, applied to the **person crop** rather than the frame.

Both halves of that decision are what make the system fit on this board:

**Crop, not frame.** Detection tolerates noise; recognition does not. Only
the region that feeds ArcFace needs restoring, which cuts stage-1 work by
one to two orders of magnitude.

**Classical, not neural.** Measured on 96×192 crops against a simulated
CCTV degradation (`python bench_denoise.py`):

| | gain | ms/crop (Nano est.) |
|---|---|---|
| `fast` — median3 | +2.5 dB | **0.8 ms** |
| `balanced` — median3 + bilateral d7 | +3.2 dB | 7.8 ms |
| `quality` — median3 + bilateral d9 | +3.2 dB | 12.1 ms |
| non-local means (best config) | +2.5 dB | 126 ms |
| *a neural denoiser, 128px tile* | *—* | *57 ms* |

The default is `balanced`: ~31 ms per inference cycle for two cameras and two
subjects each, leaving the frame budget to detection.

The absolute gains move with how harsh you make the simulated degradation;
the ordering and the cost ratios are what hold. Nano figures scale laptop
timings by 6× — **re-measure on the board**, which the script does
automatically when run there.

Two results worth knowing, because both contradict the obvious design:

**A 3×3 median does most of the work** — it recovers roughly 80% of the
available gain for under a millisecond. That isn't luck: the degradation
ends in JPEG, and impulse-like block artifacts are exactly what a median
removes and what a Gaussian-assuming filter smears. It runs first so those
artifacts don't corrupt the bilateral filter's range weights.

**Non-local means is not worth it here.** It's the textbook answer and it
lost: less gain than a median for 140× the cost.

### CLAHE and sharpening are off

Both are implemented and both are disabled. Measured against the clean
reference, CLAHE loses several dB raw and still ~2 dB after matching
first-order statistics — so it isn't just the intended tone change, it
degrades structure. Unsharp masking costs a further ~1 dB by re-amplifying
residual noise. `bench_denoise.py` prints both as an A/B against the default.

They're kept because illumination normalisation is standard for face
embedding under uneven lighting, and enrolment runs the same pipeline so
gallery and query stay consistent. That argument is plausible but
**unverified** — confirming it needs ArcFace match rates on real faces, not
PSNR. Until someone measures that, the default is the setting that
demonstrably preserves the image:

```powershell
python bench_denoise.py          # shipped profiles, including the +clahe A/B
python bench_denoise.py --sweep  # the filter comparison above
```

Run it **on the Jetson** for real numbers; elsewhere it prints a scaled
estimate and says so.

### No model, no training, no PyTorch

Stage 1 is OpenCV and nothing else — there is no neural denoiser in this
project, no code path that would load one, and no training step anywhere.
Stages 2 and 4 use published pretrained ONNX graphs you download once.

The device therefore installs no deep learning framework at all.

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

- Every file that runs on the device imports on 3.6: no
  `from __future__ import annotations`, no f-string `=`, no walrus.
- **`ultralytics` requires Python 3.8+ and will not install on the Nano.**
  Stage 2 therefore runs from ONNX, with the letterbox and NMS implemented
  in `pipeline/stage2_detect.py`. Export the graph on a workstation once.
- `gfpgan`/`basicsr` also require 3.7+. Stage 3 is optional at runtime and
  degrades to a passthrough; see the restoration policy below.
- **No PyTorch on the device.** Stage 1 is OpenCV, stages 2 and 4 are ONNX
  Runtime. That was a large part of the reason to drop the neural denoiser.
- `pip install dataclasses` — a 3.6 backport is needed.

**4 GB shared memory.** Stage 1 holding no model at all helps here: the
resident set is YOLOv8 plus ArcFace, not those two and a denoiser as well.
Run headless (`--headless`) to free the desktop's ~500 MB, add swap, and
keep GFPGAN out of the hot path.

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
with `EDGE_PROFILE`. The `nano` profile sets the `balanced` denoise profile,
`--denoise_scope crop`, 640px processing width, `--restore_mode escalate`,
YOLO at 416px, at most 2 subjects per frame, and FP16-only TensorRT.

### Stage 3b is sized to the crop, not to the model's default

SCRFD is distributed for 640×640 — finding every face in a full frame. Here
it is handed a single person crop that YOLO has already localised, so 640
would mean upscaling a ~96×192 crop by 3.3× and paying for the area. Cost
goes as the square of the input, and `det_10g.onnx` takes a dynamic one, so
the detector picks its size per image instead (rounded up to a multiple of
32, so the stride-32 level divides evenly).

Measured on one laptop CPU, `det_10g.onnx` at each size:

| Input | Anchors | Cost | vs 640 |
|---|---|---|---|
| 640 | 16800 | 56.0 ms | 1.0× |
| 320 | 4200 | 14.0 ms | 4.0× |
| **192** | **1512** | **5.5 ms** | **10.2×** |
| 128 | 672 | 2.8 ms | 20.0× |

192 is where a 96×192 crop needs no resampling at all, and it leaves the head
region about 54 px — comfortably above the ~16 px SCRFD needs at its
stride-8 level. Varying the shape per call is free: mixed sizes measured
5.64 ms/call against 5.71 ms for a fixed 192, and a never-before-seen shape
cost nothing extra.

### Which frame gets processed

The cameras deliver 30 fps; the Nano processes a handful. So every cycle
several frames arrive and one is used — and simply taking the newest is
arbitrary, since that may be the one smeared by motion or the encoder.

**Frame selection** (`FRAME_SELECT = "sharpest"`, `--frame_select`). Each
camera thread scores every frame as it arrives: variance of the Laplacian
on a 160 px-wide copy of the green channel, ~4 ms for a 1080p frame on the
Nano itself, off the
main loop. The pairer then takes the best of the frames that arrived since
its last decision:

- a pair is ranked by its **weaker** view — it is only as useful as its
  worse frame;
- scores are compared **within** each camera, never across, because the raw
  number depends on what the camera is pointed at;
- a pair inside the sync window always beats a sharper one outside it;
- frames passed over are never reconsidered, and nothing older than
  `FRAME_SELECT_MAX_AGE_MS` (250) is a candidate — so the choice never
  reaches back further than one inference cycle.

The downscale is load-bearing, not just for speed: sensor noise has high
Laplacian energy too. Measured on a test scene, a noisy *blurred* frame
scores 828 against 199 for the clean sharp one at full resolution — noise
wins — and 285 against 645 after the downscale.

**Face quality gate** (`FACE_MIN_EYE_DIST`, `FACE_MAX_YAW`). After SCRFD
finds a face, its landmarks give two numbers for free: how far apart the
eyes are, and how far the nose sits off the eye midpoint (how turned the
head is — measured along the eye line, so a tilted head is not mistaken for
a turned one). A face that is too small or too side-on is skipped for that
frame, which saves the ArcFace pass and, more importantly, avoids an
embedding that resembles every other poor face. The session summary counts
what was embedded and why the rest were refused.

Both thresholds are reasoned starting points, **not measured on real
faces**. Tune them once `enroll.py compare` can be run on real photos.

`FACE_DETECT_SIZE` (640) is now the ceiling, used for full frames during
enrolment; `FACE_DETECT_MIN` (128) is the floor, because below it small faces
start to disappear. On the same laptop this took the whole per-person path
from 116.6 ms to 64.6 ms for one subject, and 433 ms to 225 ms for four.

---

## Layout

| Path | What it does |
|---|---|
| `config.py` | Every path, threshold and runtime switch, plus hardware profiles. Start here. |
| `check_device.py` | Probe the board: GPU, memory, INT8/DLA support, what will actually run. |
| `check_py36.py` | Fails loudly if device-side code uses post-3.6 syntax. |
| `check_models.py` | Loads the three ONNX graphs, checks them against what the code assumes, times them. |
| `setup_nano.sh` | One-time Nano setup on the stock Python 3.6: apt OpenCV, pip, numpy fix, onnxruntime-gpu, swap. |
| `calibrate.py` | Measures what a similarity score means on these cameras; enables confidence percentages. |
| `sync_to_jetson.ps1` | Copies uncommitted code (and, with `-Models`, the models) from the laptop to the Nano. |
| `tests/run_all.py` | Runs the eleven suites below. No model files, no camera needed. |
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
| `pipeline/stage1_denoise.py` | Classical crop restoration: median + bilateral, OpenCV only. |
| `pipeline/stage2_detect.py` | YOLOv8 person detection and cropping. |
| `pipeline/face_align.py` | Stage 3b: SCRFD face detection and the similarity warp ArcFace needs. |
| `pipeline/analytics.py` | Scene layer: ties tracking, visitors, journeys, predictions and loitering together. |
| `pipeline/tracker.py` | Per-camera Kalman tracker: stable IDs, velocity, where each person will be next. |
| `pipeline/visitors.py` | In-memory re-identification of people never enrolled ("Visitor #N"). |
| `pipeline/journey.py` | Visits per camera, learned walking times, scored next-camera predictions, loitering. |
| `pipeline/calibration.py` | Score-to-probability fit behind the confidence percentages. |
| `pipeline/stage3_restore.py` | GFPGAN face reconstruction. |
| `pipeline/stage4_embed.py` | ArcFace 512-D embeddings. |
| `pipeline/stage5_match.py` | FAISS index, persistence, cosine matching. |
| `pipeline/utils.py` | Stage timer, HUD drawing, frame downscale. |
| `bench_denoise.py` | Measures stage 1; every default came out of it. |
| `export/export_yolo.py` | YOLOv8 → ONNX (needed for the Nano) or TensorRT. |
| `export/export_tensorrt.py` | Any ONNX → TensorRT engine (FP16; refuses INT8 on Maxwell). |

`yolocropping.py`, `gpfgan.py` and `arcface_faiss.py` are thin shims kept so
the original imports still resolve.

---

## Tests

```powershell
python tests/run_all.py
```

No model files, no camera, no GPU — so these run on the Nano as soon as
Python and OpenCV are up, which makes them the way to tell a broken install
from a broken camera. They cover the places where a mistake produces
plausible-looking output instead of an error:

| Suite | What would otherwise pass silently |
|---|---|
| `test_align.py` | A warp that is subtly wrong yields a normal-looking face and a useless embedding. Asserts landmarks land on the reference to 0.00 px, that the transform has no shear, and that SCRFD box/landmark decoding is exact. |
| `test_yolo_onnx.py` | Letterbox padding not undone — boxes land near the subject, so the crop looks fine and is wrong. |
| `test_sync.py` | Frame pairing drift: a stale anchor can make one camera permanently consume the other's frames. Also frame selection: weaker-view ranking, per-camera scaling, no reuse, the age limit. |
| `test_quality.py` | Noise passing for sharpness; a tilted head read as a turned one; the quality gate refusing the wrong faces. |
| `test_tracker.py` | A 1-second prediction that misses the person; IDs that swap when two people cross, or reset after a missed frame. |
| `test_analytics.py` | A stranger renamed on the second camera; a prediction scored wrongly; a loitering alert raised twice or never; a percentage shown without calibration; face work that starves people waiting for it. |
| `test_fusion.py` | Two people merged into one track, or one person split across two. |
| `test_hub.py` | A fresh observation fused with a minutes-old frame from a dead node. |
| `test_index.py` | Gallery persistence and cosine matching across save/reload. |
| `test_escalate.py` | GFPGAN invoked when it should not be, or skipped when it should not be. |
| `test_imports.py` | Every module imports — catches a rename that only breaks at runtime. |

Each suite is a standalone script that exits non-zero on failure, so
`run_all.py` runs them as subprocesses: several stub `cv2` or `onnxruntime`
to stay runnable without them, and those stubs must not leak into the next
suite.

`check_models.py` is the complement — the suites pass on a machine where
nothing can actually run, and that check is what tells you the graphs are
present, loadable and shaped as the code assumes.

---

## Install

There is no training step. The only thing a workstation is needed for is
exporting YOLOv8 to ONNX, because `ultralytics` cannot be installed on the
Nano's Python 3.6.

### Workstation (one-off: YOLOv8 → ONNX)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install ultralytics onnx onnxsim
python export\export_yolo.py --format onnx --imgsz 416
```

Copy the resulting `models/yolov8n.onnx` to the Nano. That's the whole
workstation role.

### Jetson Nano

**Use the Python 3.6.9 that JetPack 4.6 already ships. Do not install a
newer one.** It looks like an upgrade and is not:

| Installing 3.8+ loses | Why it matters |
|---|---|
| `python3-opencv` from apt | Built for 3.6 with CUDA **and GStreamer**. The RTSP capture needs GStreamer. Rebuilding OpenCV from source on a Nano takes hours. |
| NVIDIA's `onnxruntime-gpu` wheel | Published as `cp36` for JetPack 4.6. A cp38 build means community wheels and more risk. |

Every device-side file in this repo is written to import on 3.6 —
`python check_py36.py` verifies it, and is worth running before you copy
anything across. The only thing 3.6 costs you is GFPGAN (stage 3), which
this board cannot run at a useful speed regardless.

```bash
sudo apt install python3-opencv libopenblas-dev
pip3 install -r requirements-jetson.txt

# onnxruntime-gpu: NVIDIA's JetPack 4.6 cp36 wheel from the Jetson Zoo,
# not PyPI — the PyPI build has no CUDA for aarch64.
pip3 install onnxruntime_gpu-*-cp36-cp36m-linux_aarch64.whl

python3 check_device.py
```

Do **not** try to install `ultralytics` or `gfpgan` here; both require
Python 3.7+. Stage 2 runs from ONNX instead, and stage 3 is optional.

### Models to copy into `models/`

| File in `models/` | Source | Name in the download | Needed for |
|---|---|---|---|
| `yolov8n.onnx` | `export/export_yolo.py --format onnx --imgsz 416` | same | **stage 2, required** |
| `det_10g.onnx` | insightface `buffalo_l` pack | same | **stage 3b, required** |
| `arcface_w600k_r50.onnx` | insightface `buffalo_l` pack | `w600k_r50.onnx` | **stage 4, required** |
| `GFPGANv1.3.pth` | TencentARC/GFPGAN releases | same | stage 3, optional — won't install on the Nano |

`det_10g.onnx` and the ArcFace model come from the **same** archive, so that
is one download, not two. Note the third row: the pack calls that file
`w600k_r50.onnx`, and it has to be **renamed** on the way in.

```powershell
Copy-Item buffalo_l\det_10g.onnx   models\det_10g.onnx
Copy-Item buffalo_l\w600k_r50.onnx models\arcface_w600k_r50.onnx
```

The pack also contains `1k3d68.onnx`, `2d106det.onnx` and `genderage.onnx`
— 3D landmarks, dense landmarks and attribute estimation. This pipeline uses
none of them, and `1k3d68.onnx` alone is 137 MB, so leave them out of
`models/` rather than loading three graphs that are never called.

Then confirm all three are actually usable:

```powershell
python check_models.py
```

That checks more than existence: that ONNX Runtime will build a session,
which provider it really gets (CUDA or a silent fall back to CPU), that
SCRFD emits the 9 outputs / 3 feature levels / 2 anchors-per-stride that
`pipeline/face_align.py` hard-codes, that ArcFace is 512-D, and what each
stage costs on that machine.

**Face alignment is not optional, and it fails in a way worth
understanding.** ArcFace does not give a low score for an input that is not
a properly aligned face — it maps such inputs into one region of its
embedding space. Measured on `arcface_w600k_r50.onnx`, six unrelated
non-face inputs had pairwise cosine similarity **0.85–0.94**, against a
gallery threshold of 0.65 and a cross-view threshold of 0.45. So an
unaligned crop does not fail to match; it matches other unaligned crops,
confidently and wrongly.

That is why `config.FACE_ALLOW_FALLBACK` defaults to `False`. Without
`det_10g.onnx` the pipeline refuses to embed and makes no identifications,
rather than making bad ones. Set it `True` only to demonstrate the rest of
the pipeline, never to judge accuracy.

**Stage 1 needs no model file.** That is the point of the classical
pipeline: two of the four stages ship as code, not weights.

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

The same dashboard is served in both setups — by the hub here, or by the
Nano itself with `main_pipeline.py --dashboard 8765`.

- **Both camera feeds with a live overlay**: every person's box and ID,
  their name (`Visitor #3`, or an enrolled name in red), confidence, dwell
  time, the path they have walked, and a dashed box with an arrow for where
  the Kalman filter predicts they will be in one second. Loitering people
  are outlined in red.
- **The cross-view similarity**, large and centre: cosine between CAM_A's and
  CAM_B's live embeddings, on a ring with the threshold marked.
- Watchlist verdict: identity, confidence, best single view, and **fusion
  gain** — the number that shows whether using two cameras actually helped.
- **People**: a card per person seen recently — face, where they are or
  when they left, dwell time, and their next-camera prediction with a
  countdown bar over the predicted window.
- **Live events**: new visitors, returning visitors, walks between cameras,
  predictions and their outcomes, loitering.
- **Journey timeline**: the last five minutes, one row per person, one lane
  per camera, with walking times between cameras.
- **Next-camera prediction**: the routes learned so far and the measured hit
  rate.
- Alerts (watchlist matches and loitering), live threshold sliders (cross-
  view, watchlist, visitor, loitering time, fusion mode), edge efficiency,
  per-stage latency.

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

## Exhibition features

Everything here runs on top of identification, in numpy, at about a
millisecond per cycle, and works in both setups. Code:
`pipeline/analytics.py`, which ties together `tracker.py`, `visitors.py`,
`journey.py` and `calibration.py`.

On the Jetson, with the web UI:

```bash
python3 main_pipeline.py --dashboard 8765 --headless
```

then open `http://<jetson-ip>:8765/` on any laptop or phone on the same
network. `--headless` skips the board's own window and frees memory; drop it
if a monitor is plugged into the Nano, which then shows the same overlay.

### Tracking, and prediction between frames

Every person in view gets a constant-velocity Kalman filter per camera and
a stable ID (`A3`, `B1`). The filter estimates velocity as well as position,
so the system knows where someone is heading before the next detection; the
overlay draws that as an arrow and a dashed box one second ahead. In the
tests a steady walker's 1-second prediction lands within 3 px. With
`--detect_every 2`, the frames in between are drawn from prediction alone,
which halves the detector's cost while the boxes keep following people.

Everyone is tracked — up to `TRACK_MAX_PEOPLE` (20) — but face recognition
is rationed to `--max_subjects` people per camera per cycle (2 on the Nano),
nearest and never-recognised first. An earlier version capped *detection*
at that number instead; on real footage with more people than that in view,
people dropped in and out of the results, their IDs kept resetting, and
loitering never fired. Tracking everyone fixed it: same video, people kept
one ID for 25+ seconds.

### Track-then-recognise — the speed-up

A person who is tracked and identified keeps that identity; their face is
re-checked only every `--embed_refresh` seconds (default 2), or at once if
they cross another person, which is where a tracker can swap two people.
After a crop gives no usable face, the next attempt waits a second.

ArcFace is the most expensive stage on the Nano (122 ms per face on the
GPU), so this matters. On the pedestrian test video, tracking skipped 83%
of face checks — mostly deferred retries, since that video's faces are too
small to use. With real, visible faces the effect is different but larger
per person: ArcFace about once per two seconds per person instead of every
cycle. **`--embed_refresh 0` restores the old every-cycle behaviour**, which
makes a clean before/after comparison for the judges. The Nano figures have
still to be measured.

### Visitors — re-identifying people nobody enrolled

The first time a face is seen it gets a temporary name, `Visitor #N`, and
it is recognised again when it reappears on either camera: *"Visitor #3
recognised again on CAM_B — last seen on CAM_A 8 s ago."* Each visitor is
compared by the average of their recent looks, which is steadier than any
single frame. If the two cameras see someone at the same moment under two
different names, the names are merged; if a visitor turns out to be
enrolled, their journey continues under their real name.

Visitors exist **only in memory** and are forgotten 10 minutes after they
were last seen (`VISITOR_TTL_S`). The threshold (0.45, a dashboard slider)
is a live-vs-live comparison like the cross-view one. Lowering it to "catch
more" is the wrong fix: ArcFace places poor faces close together (see
[Face alignment is not optional](#models-to-copy-into-models)), so a low
threshold merges strangers.

### Journeys and next-camera prediction

Each sighting extends a person's visit to a camera; a gap of 3 s ends it.
When a visit ends and the same person's next visit starts on the other
camera, that walking time is learned for that route. When someone then
leaves a camera, the system predicts where they will appear next and in
what window — *"expected at CAM_B in ~8 s (4–14 s)"* — and when they arrive
it scores itself: a hit if the camera and the time are right. The running
hit rate is on the dashboard.

Be precise about this when presenting: it is a statistic learned from real
walks during the session, not a trained model. Until someone has walked a
route once it says "learning". With five or more walks the window is the
10th–90th percentile band of the walking times plus a margin; with fewer
it is a wide margin around what was seen. Coming back to the same camera
before anyone could have reached the other one withdraws the prediction
rather than counting it as a miss.

**Camera placement decides what you can show.** Two cameras on the same
spot give multi-angle fusion, but nobody ever *walks* between them, so
there are no routes to learn. Two cameras on different spots give walks and
predictions, but never two angles at once. Partly overlapping views give
some of both; plan the stand around which story matters more.

### Dwell time and loitering

Each person's continuous time in view at a camera is shown on their box
and their card. Past `--loiter_seconds` (30 by default, also a slider) a
loitering alert fires once per visit. It works without a face — the track
alone is enough — so it also covers people facing away. On the pedestrian
test video it raised 11 alerts on people who stayed in view over 30 s.

### Confidence percentages

A cosine score is not a probability, so the dashboard shows **raw scores
labelled "uncalibrated"** until the relationship has been measured on these
cameras:

```bash
# calib/<person>/<5-10 photos each>, ideally taken with the actual cameras
python3 calibrate.py --dir calib
python3 calibrate.py --show
```

It fits P(same person | score) from every same-person and different-person
pair of photos, refuses data too thin to support a percentage, never shows
0% or 100%, and prints the equal error rate — a check on whether the
thresholds suit these cameras. Percentages assume a same-person and a
different-person pair are equally likely beforehand; the dashboard says
so. `--from_gallery` reuses enrolled vectors instead, which is convenient
but optimistic.

### What has not been measured yet

- **None of this has run on the Nano yet** — it was built and checked on a
  laptop (11 test suites, Python 3.6 compatibility, the real pipeline with
  real models on a pedestrian video). Run `tests/run_all.py` and a short
  session on the board before relying on its speed.
- **Visitors, journeys and predictions need real faces**, and the test video
  has none usable. Their logic is tested on synthetic people; the thresholds
  have not been tested on real faces.

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

With stage 1 classical and crop-scoped, **detection is now the bottleneck**,
not denoising — the restorer costs ~32 ms per cycle against YOLOv8's much
larger share. Tune in this order (the `nano` profile already applies 1–3):

1. **`--process_width 640`** — downscale before any inference. Cost scales
   with pixel count, so 1080p → 640×360 is a 9× cut.
2. **`--yolo_imgsz 416`** — the detector is the dominant cost now; this is
   where the remaining time is.
3. **`--restore_mode escalate`** — keep GFPGAN off the hot path.
4. **`--denoise_profile fast`** — median only. +4.7 dB instead of +6.8 dB,
   but 3.6 ms per cycle instead of 32 ms.
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
- **Matched people are photographed by default.** `SAVE_MATCHES = True`
  writes the frames and face of every watchlist match to `captures/`. Run
  with `--no_save_matches` if only embeddings may be kept.
- Visitors (people never enrolled) live in memory only and are forgotten
  10 minutes after they were last seen; nothing about them is written to
  disk.
- **The dashboard has no login.** Anyone on the same network who opens its
  address sees the camera frames and faces. Use a network you control - not
  a venue's open WiFi - and stop the pipeline when the demo ends.
