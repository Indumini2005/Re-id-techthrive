"""Are the three ONNX graphs present, loadable, and shaped as the code assumes?

    python check_models.py

Run this on the Jetson before main_pipeline.py. The test suites in tests/
need no model files, so they pass on a machine where nothing can actually
run; this is the complement. It checks what tests/ cannot:

  - each file exists and ONNX Runtime will build a session from it
  - the provider actually in use (CUDA vs a silent fall back to CPU)
  - SCRFD's output count, feature levels and anchors-per-stride, which
    pipeline/face_align.py hard-codes as 9 / 3 / 2
  - ArcFace emits 512 dimensions
  - measured forward-pass cost per stage on THIS machine

Written for Python 3.6, like the rest of the device-side code.
"""

from __future__ import print_function

import argparse
import sys
import time

import numpy as np

import config

STAGES = [
    ("stage 2  person detect", "PERSON_ONNX", "yolov8n.onnx", True),
    ("stage 3b face detect", "FACE_DETECT_ONNX", "det_10g.onnx", True),
    ("stage 4  embedding", "ARCFACE_ONNX", "arcface_w600k_r50.onnx", True),
]

problems = []
notes = []


def head(title):
    print("")
    print("=" * 70)
    print(title)
    print("=" * 70)


def fail(msg):
    problems.append(msg)
    print("  FAIL  " + msg)


def ok(msg):
    print("  ok    " + msg)


def warn(msg):
    notes.append(msg)
    print("  note  " + msg)


def best_of(session, feed, runs):
    session.run(None, feed)                     # warm up / allocate
    best = None
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run(None, feed)
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    return best * 1000.0


def resolve(attr, fallback):
    path = getattr(config, attr, None)
    if path is None:
        path = config.MODELS_DIR / fallback
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=10,
                    help="timed passes per graph (default 10)")
    ap.add_argument("--no-bench", action="store_true",
                    help="check loading and shapes only, skip timing")
    args = ap.parse_args()

    head("ENVIRONMENT")
    print("  python   %s" % sys.version.split()[0])
    print("  numpy    %s" % np.__version__)
    try:
        import onnxruntime as ort
    except ImportError as exc:
        print("  FAIL  onnxruntime is not installed (%s)" % exc)
        print("\nOn JetPack 4.6 install the NVIDIA-built wheel; the PyPI one has "
              "no aarch64 CUDA build. See README, 'Installing on the Nano'.")
        return 1
    print("  ort      %s" % ort.__version__)
    available = ort.get_available_providers()
    print("  providers %s" % ", ".join(available))
    print("  profile  %s" % config.PROFILE)

    wanted = config.provider_list()
    if "CUDAExecutionProvider" in wanted and "CUDAExecutionProvider" not in available:
        warn("config asks for CUDA but this ONNX Runtime has no CUDA provider; "
             "everything below runs on CPU and will be far slower")

    ort.set_default_logger_severity(3)

    timings = []

    for label, attr, fallback, required in STAGES:
        path = resolve(attr, fallback)
        head("%s  -  %s" % (label.upper(), path.name))

        if not path.exists():
            if required:
                fail("%s is missing. Expected at %s" % (path.name, path))
            else:
                warn("%s is absent (optional)" % path.name)
            continue

        size_mb = path.stat().st_size / (1024.0 * 1024.0)
        print("  file     %.1f MB" % size_mb)

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 3
        try:
            sess = ort.InferenceSession(str(path), sess_options=opts,
                                        providers=config.provider_list())
        except Exception as exc:                                   # noqa: BLE001
            fail("%s will not load: %s" % (path.name, exc))
            continue

        actual = sess.get_providers()
        print("  running on %s" % actual[0])
        # Only remark on a fallback that is actually surprising. If this ORT
        # build has no CUDA provider at all, the environment note above
        # already said so once and repeating it per model adds nothing.
        if ("CUDAExecutionProvider" in available
                and actual[0] != "CUDAExecutionProvider"):
            warn("%s fell back to %s although this build HAS CUDA"
                 % (path.name, actual[0]))
        ok("session created")

        inp = sess.get_inputs()[0]
        outs = sess.get_outputs()
        print("  input    %s %s" % (inp.name, inp.shape))
        print("  outputs  %d" % len(outs))

        # Shape the feed the way the pipeline will.
        if attr == "FACE_DETECT_ONNX":
            dims = [1, 3, config.FACE_DETECT_SIZE, config.FACE_DETECT_SIZE]
        elif attr == "ARCFACE_ONNX":
            dims = [1, 3, config.ARCFACE_INPUT, config.ARCFACE_INPUT]
        else:
            dims = [d if isinstance(d, int) else 1 for d in inp.shape]
        feed = {inp.name: np.zeros(dims, np.float32)}

        try:
            result = sess.run(None, feed)
        except Exception as exc:                                   # noqa: BLE001
            fail("%s loads but will not run at %s: %s" % (path.name, dims, exc))
            continue
        ok("forward pass at %s" % "x".join(str(d) for d in dims))

        # --- assumptions the code depends on ---------------------------
        if attr == "FACE_DETECT_ONNX":
            if len(outs) < 9:
                fail("%s has %d outputs. Alignment needs landmarks, which "
                     "means 9. Use det_10g.onnx or det_2.5g.onnx, not a "
                     "box-only SCRFD variant." % (path.name, len(outs)))
            else:
                ok("9+ outputs: the landmark branch is present")
                fmc = len(outs) // 3
                if fmc != 3:
                    fail("%d feature levels; face_align.py assumes 3" % fmc)
                else:
                    ok("3 feature levels, as face_align.py assumes")
                    size = dims[2]
                    bad = []
                    for i, stride in enumerate((8, 16, 32)):
                        n = result[i].shape[0]
                        expect = (size // stride) ** 2 * 2
                        if n != expect:
                            bad.append((stride, n, expect))
                    if bad:
                        for stride, n, expect in bad:
                            fail("stride %d gave %d anchors, num_anchors=2 "
                                 "predicts %d" % (stride, n, expect))
                    else:
                        ok("2 anchors per cell at strides 8/16/32")

        if attr == "ARCFACE_ONNX":
            dim = result[0].shape[-1]
            if dim != config.EMBEDDING_DIM:
                fail("embedding is %d-D but config.EMBEDDING_DIM is %d"
                     % (dim, config.EMBEDDING_DIM))
            else:
                ok("%d-D embedding, matching config" % dim)

        if not args.no_bench:
            ms = best_of(sess, feed, args.runs)
            timings.append((label, "x".join(str(d) for d in dims), ms))
            print("  forward  %.1f ms  (best of %d)" % (ms, args.runs))

            # The face detector is the one stage whose input size the
            # pipeline varies, and the difference is an order of magnitude,
            # so time what it will really be handed as well as the ceiling.
            if attr == "FACE_DETECT_ONNX" and str(inp.shape[2]) not in "0123456789":
                crop = config.FACE_DETECT_MIN
                for cand in (192, 224):
                    if config.FACE_DETECT_MIN <= cand <= config.FACE_DETECT_SIZE:
                        crop = cand
                        break
                cdims = [1, 3, crop, crop]
                try:
                    cms = best_of(sess, {inp.name: np.zeros(cdims, np.float32)},
                                  args.runs)
                except Exception as exc:                           # noqa: BLE001
                    warn("dynamic resize to %d failed: %s" % (crop, exc))
                else:
                    print("  forward  %.1f ms at %dx%d, the size a person crop "
                          "gets  (%.1fx cheaper)" % (cms, crop, crop, ms / cms))
                    timings.append(("stage 3b on a crop", "%dx%d" % (crop, crop), cms))

        del sess

    # --- the gallery -----------------------------------------------------
    head("GALLERY")
    try:
        from pipeline.stage5_match import ThreatIndex
        idx = ThreatIndex()
        stats = idx.stats()
        print("  %s" % stats)
        if not stats.get("vectors"):
            warn("the gallery is empty, so nothing can be identified yet. "
                 "Enrol with: python enroll.py add --name NAME --image FILE")
        else:
            ok("%d vectors across %d identities"
               % (stats["vectors"], stats["identities"]))
    except Exception as exc:                                       # noqa: BLE001
        warn("could not open the gallery: %s" % exc)

    # --- budget ----------------------------------------------------------
    if timings:
        head("MEASURED COST")
        for label, shape, ms in timings:
            print("  %-24s %-16s %7.1f ms" % (label, shape, ms))
        print("")
        print("  Per detected person per camera, the pipeline pays stage 3b")
        print("  plus stage 4 once - stage 2 is paid once per frame. With")
        print("  N people in view that is:")
        per_frame = dict((l, m) for l, s, m in timings)
        det = per_frame.get("stage 2  person detect", 0.0)
        emb = per_frame.get("stage 4  embedding", 0.0)
        # Use the crop-sized figure when it was measured: that is what the
        # pipeline actually pays. The 640 number is the enrolment path.
        face = per_frame.get("stage 3b on a crop")
        if face is None:
            face = per_frame.get("stage 3b face detect", 0.0)
            basis = "stage 3b at %d (the full-frame ceiling)" % config.FACE_DETECT_SIZE
        else:
            basis = "stage 3b at the size a person crop gets"
        print("")
        print("     people   total ms   max fps")
        for n in (1, 2, 4):
            total = det + n * (face + emb)
            print("     %-8d %8.1f   %6.1f" % (n, total, 1000.0 / total if total else 0))
        print("")
        print("  Using %s." % basis)
        print("  Enrolment pays the 640 figure instead, once per photo, off")
        print("  the live path. Classical stage 1 is not counted here - run")
        print("  bench_denoise.py for that. Nor is decoding two RTSP streams.")

    head("VERDICT")
    if problems:
        print("  %d blocking problem(s):" % len(problems))
        for p in problems:
            print("    - %s" % p)
    else:
        print("  All three graphs load and match what the code assumes.")
    if notes:
        print("")
        print("  %d thing(s) to be aware of:" % len(notes))
        for n in notes:
            print("    - %s" % n)
    print("")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
