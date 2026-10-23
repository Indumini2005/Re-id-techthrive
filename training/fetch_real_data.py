"""Download a set of real photographs to train the denoiser on.

    python -m training.fetch_real_data --out data/real --dataset div2k_valid

Why this exists: the degradation chain is the CCTV-specific half of the
problem and it is simulated faithfully. The *clean* half has to carry the
statistics of real photographs - texture at every scale, dense edges, fine
detail. Procedural scenes do not, and a denoiser trained only on them
learns that smooth is always the right answer, then erases facial detail on
real footage. That is the failure the downstream face stages cannot absorb.

Denoising is a low-level task, so the content does not have to be
surveillance footage. A few hundred sharp photographs of anything give the
model the right notion of what detail looks like. Mix them with synthetic
CCTV frames for the geometry:

    python -m training.train_denoiser \
        --train_dir data/synth/clean data/real/clean --arch nano
"""

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# All freely downloadable, standard in the image-restoration literature.
DATASETS = {
    "div2k_valid": {
        "url": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
        "size_mb": 430,
        "images": 100,
        "note": "100 2K photographs. The usual starting point; plenty for "
                "patch-based denoiser training.",
    },
    "div2k_train": {
        "url": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
        "size_mb": 3500,
        "images": 800,
        "note": "800 2K photographs. Better, but a 3.5 GB download.",
    },
}


def report_progress(count, block_size, total_size):
    done = count * block_size
    if total_size > 0:
        pct = min(100.0, done * 100.0 / total_size)
        sys.stdout.write("\r  %5.1f%%  %6.1f / %.1f MB"
                         % (pct, done / 1e6, total_size / 1e6))
    else:
        sys.stdout.write("\r  %6.1f MB" % (done / 1e6))
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser(description="Fetch real photographs for training")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dataset", default="div2k_valid", choices=sorted(DATASETS))
    ap.add_argument("--limit", type=int, default=0,
                    help="keep only the first N images (0 = all)")
    ap.add_argument("--max_side", type=int, default=1600,
                    help="downscale longest side; 2K originals are larger than "
                         "patch training needs and slow the loader down. 0 keeps them.")
    ap.add_argument("--keep_zip", action="store_true")
    args = ap.parse_args()

    spec = DATASETS[args.dataset]
    out_clean = args.out / "clean"
    out_clean.mkdir(parents=True, exist_ok=True)

    existing = len(list(out_clean.glob("*.png")))
    if existing:
        print("%d images already in %s; nothing to do." % (existing, out_clean))
        return

    archive = args.out / ("%s.zip" % args.dataset)
    print("dataset : %s (%d images, ~%d MB)" % (args.dataset, spec["images"], spec["size_mb"]))
    print("note    : %s" % spec["note"])
    print("url     : %s" % spec["url"])

    if not archive.exists():
        print("\ndownloading...")
        urllib.request.urlretrieve(spec["url"], archive, reporthook=report_progress)
        print()
    else:
        print("\nusing the archive already on disk")

    print("extracting...")
    staging = args.out / "_staging"
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(staging)

    import cv2

    sources = sorted(p for p in staging.rglob("*")
                     if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if args.limit:
        sources = sources[: args.limit]

    for i, path in enumerate(sources):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        if args.max_side:
            h, w = img.shape[:2]
            if max(h, w) > args.max_side:
                scale = args.max_side / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)),
                                 interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_clean / ("real_%05d.png" % i)), img)
        if (i + 1) % 25 == 0:
            print("  %d / %d" % (i + 1, len(sources)))

    shutil.rmtree(staging, ignore_errors=True)
    if not args.keep_zip:
        archive.unlink(missing_ok=True)

    kept = len(list(out_clean.glob("*.png")))
    print("-" * 58)
    print("wrote %d real photographs to %s" % (kept, out_clean))
    print("\nmix them with the synthetic scenes:")
    print("  python -m training.train_denoiser \\")
    print("      --train_dir data/synth/clean %s --arch nano" % out_clean)


if __name__ == "__main__":
    main()
