"""Measure what a similarity score means on YOUR cameras.

The dashboard shows "94% likely the same person" only after this has been
run on real photos. Until then it shows the raw score and says
"uncalibrated" - a percentage nobody measured would be the one number on
screen a judge could rightly call made up.

Collect photos, ideally taken with the actual cameras in the actual room:

    calib/
        Dulagi/     5-10 photos
        Indumini/   5-10 photos
        Kaveesha/   ...

then:

    python calibrate.py --dir calib          # fit from the folders
    python calibrate.py --from_gallery       # or from enrolled vectors
    python calibrate.py --show               # print the current calibration
    python calibrate.py --clear              # go back to raw scores

Every pair of photos of the same person is a "same" example; every pair of
photos of two different people is a "different" example. With 4 people x
6 photos that is 60 same pairs and 216 different pairs. The fit refuses
data too thin to support a percentage (see pipeline/calibration.py).

It also prints the equal error rate and the score where false accepts fall
to 1%, which say whether the thresholds in config.py suit these cameras.

--from_gallery reuses vectors already enrolled (two or more per person,
e.g. from `enroll.py capture`). Convenient, but enrolment photos are better
than live CCTV frames, so the percentages come out optimistic; say so if
you use it.
"""

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

import config
from pipeline.calibration import ScoreCalibrator, build_calibration

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def pair_scores(groups):
    """{person: [unit vectors]} -> (same-person scores, different-person scores)."""
    names = [n for n in groups if len(groups[n])]
    genuine, impostor = [], []
    for n in names:
        for a, b in combinations(groups[n], 2):
            genuine.append(float(np.dot(a, b)))
    for n1, n2 in combinations(names, 2):
        for a in groups[n1]:
            for b in groups[n2]:
                impostor.append(float(np.dot(a, b)))
    return genuine, impostor


def embed_folders(root, denoise=True):
    import cv2
    from enroll import _prepare_face
    from pipeline.stage4_embed import get_embedder
    from pipeline.vectors import l2_normalize

    embedder = get_embedder()
    groups = {}
    for person in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        vecs = []
        for path in sorted(person.rglob("*")):
            if path.suffix.lower() not in IMG_EXTS:
                continue
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                print("  %s/%s: unreadable, skipped" % (person.name, path.name))
                continue
            # The same chain as live frames: denoise, detect, align, embed.
            face = _prepare_face(img, denoise=denoise, restore=False, quiet=True)
            if face is None:
                print("  %s/%s: no usable face, skipped" % (person.name, path.name))
                continue
            emb = embedder.embed(face)
            if emb is not None:
                vecs.append(l2_normalize(emb))
        print("  %-20s %d usable photo(s)" % (person.name, len(vecs)))
        if len(vecs) >= 2:
            groups[person.name] = vecs
        elif vecs:
            print("    (needs at least 2 to form a same-person pair; left out)")
    return groups


def gallery_groups():
    from pipeline.stage5_match import get_index

    index = get_index()
    if index.index.ntotal == 0:
        raise SystemExit("the gallery is empty - enrol people first, or use --dir")
    vectors = np.asarray(index.index.reconstruct_n(0, index.index.ntotal), dtype=np.float32)
    groups = {}
    for vec, label in zip(vectors, index.labels):
        groups.setdefault(label, []).append(vec / max(np.linalg.norm(vec), 1e-9))
    for name, vecs in sorted(groups.items()):
        print("  %-20s %d vector(s)" % (name, len(vecs)))
    return {n: v for n, v in groups.items() if len(v) >= 2}


def show(cal):
    if not cal.calibrated:
        print("Not calibrated: the dashboard shows raw similarity scores.")
        return
    d = cal.data
    print("Calibrated %s from %s" % (d.get("fitted_at"), d.get("source")))
    print("  people              : %s" % d.get("n_identities"))
    print("  same-person pairs   : %s   mean score %.3f +- %.3f"
          % (d["n_genuine"], d["genuine_mean"], d["genuine_std"]))
    print("  diff-person pairs   : %s   mean score %.3f +- %.3f"
          % (d["n_impostor"], d["impostor_mean"], d["impostor_std"]))
    print("  equal error rate    : %.1f%% at score %.3f" % (100 * d["eer"], d["eer_threshold"]))
    print("  1%% false-accept at  : score %.3f  (catches %.1f%% of same-person pairs)"
          % (d["far_1pct_threshold"], 100 * d["tar_at_far_1pct"]))
    print("  50%% confidence at   : score %.3f" % d["score_at_50pct"])
    print("")
    print("  score -> shown as")
    for s in (0.2, 0.3, 0.4, 0.45, 0.5, 0.6, 0.65, 0.7, 0.8):
        print("   %.2f  ->  %3.0f%%" % (s, 100 * cal.probability(s)))
    print("")
    print("  Current thresholds: cross-view %.2f, visitor %.2f, gallery %.2f"
          % (config.CROSS_VIEW_THRESHOLD, config.VISITOR_THRESHOLD, config.MATCH_THRESHOLD))
    print("  The equal-error score above is a reasonable starting point for them.")
    print("  Percentages assume a same-person and a different-person pair are")
    print("  equally likely beforehand (stated in the dashboard).")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--dir", type=Path, help="folder with one subfolder of photos per person")
    src.add_argument("--from_gallery", action="store_true", help="use enrolled vectors")
    ap.add_argument("--show", action="store_true", help="print the current calibration")
    ap.add_argument("--clear", action="store_true", help="delete it; show raw scores")
    ap.add_argument("--no_denoise", action="store_true", help="skip stage 1 on the photos")
    args = ap.parse_args()

    cal = ScoreCalibrator()
    if args.clear:
        cal.clear()
        print("calibration removed (%s)" % cal.path)
        return 0
    if args.show or not (args.dir or args.from_gallery):
        show(cal)
        return 0

    print("Embedding faces..." if args.dir else "Reading the gallery...")
    groups = embed_folders(args.dir, denoise=not args.no_denoise) if args.dir else gallery_groups()
    genuine, impostor = pair_scores(groups)
    print("\n%d people, %d same-person pairs, %d different-person pairs"
          % (len(groups), len(genuine), len(impostor)))

    try:
        data = build_calibration(genuine, impostor, n_identities=len(groups),
                                 source=("photos in %s" % args.dir) if args.dir
                                 else "enrolled gallery (optimistic)")
    except ValueError as exc:
        print("\nNot calibrated: %s" % exc)
        print("The dashboard keeps showing raw scores.")
        return 1

    cal.save(data)
    print("\nsaved %s\n" % cal.path)
    show(cal)
    return 0


if __name__ == "__main__":
    sys.exit(main())
