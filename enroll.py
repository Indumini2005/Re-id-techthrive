"""Manage the local threat vector DB.

Enrollment runs the same stages as live inference (denoise -> restore ->
embed), so the stored vectors sit in the same distribution as the query
vectors. Enrolling a clean studio photo against noisy CCTV queries is the
single most common reason a re-ID system silently underperforms.

With two cameras, enroll from both angles - fusion can only exploit angular
variety that the gallery also contains.

    python enroll.py add --image suspect.jpg --identity Subject_A
    python enroll.py add --dir enroll/subject_a --identity Subject_A
    python enroll.py capture --identity Subject_A --seconds 10   # live, both cameras
    python enroll.py compare --image_a cam_a.jpg --image_b cam_b.jpg
    python enroll.py list
    python enroll.py test --image probe.jpg
    python enroll.py remove --identity Subject_A
    python enroll.py clear
"""

import argparse
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

import config

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def _prepare_face(
    img_bgr: np.ndarray, denoise: bool, restore: bool, quiet: bool = False
) -> Optional[np.ndarray]:
    """Raw image -> aligned face, through the same chain as live frames.

    Enrolment must mirror the query path exactly. If a gallery vector comes
    from an aligned face and a query vector from an unaligned crop, the
    cosine between them measures the difference in framing as much as the
    difference in identity, and no threshold fixes that.
    """
    working = img_bgr

    if denoise:
        try:
            from pipeline.stage1_denoise import get_denoiser
            working = get_denoiser().denoise(working)
        except Exception as exc:
            if not quiet:
                print(f"  (denoise skipped: {exc})")

    if restore:
        try:
            from pipeline.stage3_restore import get_restorer
            face = get_restorer().restore(working)
            if face is not None:
                return face
            if not quiet:
                print("  (GFPGAN found no face; falling through to alignment)")
        except Exception as exc:
            if not quiet:
                print(f"  (restore skipped: {exc})")

    # Same detector and warp the live loop uses.
    from pipeline.face_align import prepare_for_embedding

    aligned = prepare_for_embedding(working, size=config.ARCFACE_INPUT)
    if aligned is None and not quiet:
        print("  (no face detected)")
    return aligned


def cmd_add(args) -> None:
    from pipeline.stage4_embed import get_embedder
    from pipeline.stage5_match import get_index

    paths: List[Path] = []
    if args.image:
        paths.append(args.image)
    if args.dir:
        paths += sorted(p for p in Path(args.dir).rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not paths:
        raise SystemExit("pass --image or --dir")

    embedder, index = get_embedder(), get_index()
    added = 0

    for path in paths:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  skip (unreadable): {path.name}")
            continue

        print(f"  {path.name}")
        face = _prepare_face(img, denoise=not args.no_denoise, restore=not args.no_restore)
        if face is None:
            continue

        embedding = embedder.embed_flipped(face)
        if embedding is None:
            print("    no embedding produced")
            continue

        # Warn if this vector is nearly identical to one already stored:
        # duplicates inflate the index without adding angular coverage.
        existing = index.search(embedding, top_k=1)
        if existing and existing[0].score > 0.98 and existing[0].identity == args.identity:
            print(f"    near-duplicate of an existing vector ({existing[0].score:.3f}); skipped")
            continue

        index.add(embedding, args.identity, note=path.name, persist=False)
        added += 1
        print("    enrolled")

        if args.save_faces:
            out = config.DB_DIR / "faces" / args.identity
            out.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / f"{path.stem}.png"), face)

    index.save()
    print(f"\nadded {added} vector(s) for '{args.identity}'")
    print(index.stats())


def cmd_capture(args) -> None:
    """Enroll a subject live from both cameras at once.

    Two-camera fusion only pays off if the gallery holds the same angular
    variety the cameras will see. Standing someone in front of both cameras
    for ten seconds gives better coverage than any single photo.
    """
    import time

    from pipeline.sources import SyncedCapture, cameras_from_args
    from pipeline.stage4_embed import get_embedder
    from pipeline.stage5_match import get_index

    cameras = cameras_from_args(args.source)
    capture = SyncedCapture(
        cameras, use_gstreamer=not args.no_gstreamer, require_all=False
    ).start()

    print(f"cameras: {', '.join(c['name'] for c in cameras)}")
    if not capture.wait_until_ready(timeout=15.0):
        capture.release()
        raise SystemExit("not every camera delivered a frame")

    try:
        from pipeline.stage2_detect import get_detector
        detector = get_detector()
    except Exception as exc:
        capture.release()
        raise SystemExit(f"detector required for live capture: {exc}")

    embedder = get_embedder()
    # candidates[camera] = list of (quality, embedding, face)
    candidates = {c["name"]: [] for c in cameras}

    print(f"\ncapturing '{args.identity}' for {args.seconds}s - "
          "have the subject turn slowly to cover both angles\n")
    deadline = time.monotonic() + args.seconds

    try:
        while time.monotonic() < deadline:
            pair = capture.read(timeout=2.0)
            if pair is None:
                continue

            for name, frame in pair.frames.items():
                subjects = detector.detect(frame.image, max_subjects=1)
                if not subjects:
                    continue
                crop = _prepare_face(
                    subjects[0].crop,
                    denoise=not args.no_denoise,
                    restore=not args.no_restore,
                    quiet=True,
                )
                if crop is None:
                    continue
                embedding = embedder.embed(crop)
                if embedding is None:
                    continue

                # Sharper faces make better gallery entries; a blurred
                # enrollment vector drags every future match toward it.
                sharpness = float(
                    cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                )
                quality = subjects[0].conf * min(sharpness / 100.0, 2.0)
                candidates[name].append((quality, embedding, crop))

            captured = sum(len(v) for v in candidates.values())
            remaining = deadline - time.monotonic()
            print(f"\r  {remaining:4.1f}s left   candidates: "
                  + "  ".join(f"{k} {len(v)}" for k, v in candidates.items())
                  + f"   total {captured}   ", end="", flush=True)
    except KeyboardInterrupt:
        print("\n  stopped early")
    finally:
        capture.release()

    print()
    index = get_index()
    added = 0

    for name, items in candidates.items():
        if not items:
            print(f"  {name}: no usable faces captured")
            continue

        items.sort(key=lambda t: t[0], reverse=True)
        selected = []
        # Greedily keep the best vectors that are not near-duplicates of each
        # other, so the gallery gets angular spread rather than N copies of
        # the same pose.
        for quality, embedding, face in items:
            if any(float(np.dot(embedding, chosen[1])) > args.diversity for chosen in selected):
                continue
            selected.append((quality, embedding, face))
            if len(selected) >= args.per_camera:
                break

        for i, (quality, embedding, face) in enumerate(selected):
            index.add(embedding, args.identity,
                      note=f"live:{name}:{i} q={quality:.2f}", persist=False)
            added += 1
            if args.save_faces:
                out = config.DB_DIR / "faces" / args.identity
                out.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out / f"{name}_{i}.png"), face)

        print(f"  {name}: {len(selected)} of {len(items)} candidates enrolled")

    index.save()
    print(f"\nadded {added} vector(s) for '{args.identity}'")
    print(index.stats())


def cmd_compare(args) -> None:
    """Compare two frames of the same moment, the way the live loop does.

    Reports the cross-view similarity between the two embeddings, then the
    gallery hit for each view alone and for the fused vector - which is the
    number that shows whether using both cameras actually bought anything.
    """
    from pipeline.fusion import CrossViewFusion, ViewObservation
    from pipeline.stage4_embed import get_embedder
    from pipeline.stage5_match import get_index

    embedder = get_embedder()
    observations = []

    for label, path in (("CAM_A", args.image_a), ("CAM_B", args.image_b)):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"could not read {path}")

        face = _prepare_face(img, denoise=not args.no_denoise, restore=not args.no_restore)
        embedding = embedder.embed(face)
        if embedding is None:
            raise SystemExit(f"no embedding produced for {path}")

        observations.append(
            ViewObservation(camera=label, box=(0, 0, img.shape[1], img.shape[0]),
                            conf=1.0, crop=img, face=face, embedding=embedding)
        )

    cross = float(np.dot(observations[0].embedding, observations[1].embedding))
    same = cross >= args.cross_view_threshold

    print(f"\n{args.image_a.name}  vs  {args.image_b.name}")
    print("-" * 58)
    print(f"cross-view cosine : {cross:.4f}")
    print(f"same person       : {'YES' if same else 'NO'} "
          f"(threshold {args.cross_view_threshold})")

    index = get_index()
    if index.index.ntotal == 0:
        print("\ngallery is empty - cross-view comparison only")
        return

    fusion = CrossViewFusion(
        index=index,
        cross_view_threshold=args.cross_view_threshold,
        gallery_threshold=config.MATCH_THRESHOLD,
        fusion_mode=args.fusion,
    )
    tracks = fusion.process({o.camera: [o] for o in observations})

    print(f"\ngallery identification (threshold {config.MATCH_THRESHOLD}, "
          f"fusion '{args.fusion}')")
    print("-" * 58)
    for track in tracks:
        for obs in track.observations:
            if obs.solo_match:
                print(f"  {obs.camera:<8} {obs.solo_match.identity:<22} "
                      f"{obs.solo_match.score:.4f}"
                      f"{'  MATCH' if obs.solo_match.is_match else ''}")
        if track.match:
            tag = "  MATCH" if track.match.is_match else ""
            scope = "fused" if track.is_multiview else f"single ({track.cameras[0]})"
            print(f"  {scope:<8} {track.match.identity:<22} {track.match.score:.4f}{tag}")
        for note in track.notes:
            print(f"           note: {note}")

    if not same:
        print("\nThe two views were NOT associated, so they were identified "
              "separately rather than fused.")


def cmd_list(args) -> None:
    from pipeline.stage5_match import get_index

    index = get_index()
    print(index.stats())
    if not index.labels:
        print("(empty)")
        return

    print(f"\n{'identity':<28} {'vectors':>8}")
    print("-" * 38)
    for identity in index.identities:
        print(f"{identity:<28} {index.labels.count(identity):>8}")

    if args.verbose:
        print("\nrows:")
        for i, (label, meta) in enumerate(zip(index.labels, index.meta)):
            print(f"  {i:4d}  {label:<24} {meta.get('note', ''):<28} {meta.get('added', '')}")


def cmd_test(args) -> None:
    from pipeline.stage4_embed import get_embedder
    from pipeline.stage5_match import get_index

    img = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"could not read {args.image}")

    face = _prepare_face(img, denoise=not args.no_denoise, restore=not args.no_restore)
    embedding = get_embedder().embed(face)
    if embedding is None:
        raise SystemExit("no embedding produced")

    hits = get_index().search(embedding, top_k=args.top_k)
    if not hits:
        print("index is empty")
        return

    print(f"\nprobe: {args.image.name}   (threshold {config.MATCH_THRESHOLD})")
    print(f"{'rank':<6}{'identity':<28}{'cosine':>8}  verdict")
    print("-" * 56)
    for hit in hits:
        verdict = "MATCH" if hit.score >= config.MATCH_THRESHOLD else "-"
        print(f"{hit.rank:<6}{hit.identity:<28}{hit.score:>8.4f}  {verdict}")


def cmd_remove(args) -> None:
    from pipeline.stage5_match import get_index

    removed = get_index().remove_identity(args.identity)
    print(f"removed {removed} vector(s) for '{args.identity}'")


def cmd_clear(args) -> None:
    from pipeline.stage5_match import get_index

    if not args.yes:
        confirm = input("This erases the whole vector DB. Type 'yes' to confirm: ")
        if confirm.strip().lower() != "yes":
            print("aborted")
            return
    get_index().clear()
    print("vector DB cleared")


def main():
    ap = argparse.ArgumentParser(description="Threat vector DB management")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def stage_flags(p):
        p.add_argument("--no_denoise", action="store_true")
        p.add_argument("--no_restore", action="store_true")

    p_add = sub.add_parser("add", help="enroll a subject")
    p_add.add_argument("--image", type=Path)
    p_add.add_argument("--dir", type=Path, help="enroll every image in a folder")
    p_add.add_argument("--identity", required=True)
    p_add.add_argument("--save_faces", action="store_true", help="keep the aligned crops")
    stage_flags(p_add)
    p_add.set_defaults(func=cmd_add)

    p_cap = sub.add_parser("capture", help="enroll live from both cameras")
    p_cap.add_argument("--identity", required=True)
    p_cap.add_argument("--source", action="append", default=None,
                       help="repeat per camera; defaults to config.CAMERAS")
    p_cap.add_argument("--seconds", type=float, default=10.0)
    p_cap.add_argument("--per_camera", type=int, default=5,
                       help="vectors kept per camera")
    p_cap.add_argument("--diversity", type=float, default=0.92,
                       help="reject a candidate more similar than this to one already kept")
    p_cap.add_argument("--no_gstreamer", action="store_true")
    p_cap.add_argument("--save_faces", action="store_true")
    stage_flags(p_cap)
    p_cap.set_defaults(func=cmd_capture)

    p_cmp = sub.add_parser("compare", help="compare two frames the way the live loop does")
    p_cmp.add_argument("--image_a", type=Path, required=True)
    p_cmp.add_argument("--image_b", type=Path, required=True)
    p_cmp.add_argument("--cross_view_threshold", type=float,
                       default=config.CROSS_VIEW_THRESHOLD)
    p_cmp.add_argument("--fusion", default=config.FUSION_MODE,
                       choices=["mean", "max", "first"])
    stage_flags(p_cmp)
    p_cmp.set_defaults(func=cmd_compare)

    p_list = sub.add_parser("list", help="show enrolled identities")
    p_list.add_argument("-v", "--verbose", action="store_true")
    p_list.set_defaults(func=cmd_list)

    p_test = sub.add_parser("test", help="query the DB with one image")
    p_test.add_argument("--image", type=Path, required=True)
    p_test.add_argument("--top_k", type=int, default=config.MATCH_TOP_K)
    stage_flags(p_test)
    p_test.set_defaults(func=cmd_test)

    p_rm = sub.add_parser("remove", help="delete an identity")
    p_rm.add_argument("--identity", required=True)
    p_rm.set_defaults(func=cmd_remove)

    p_clear = sub.add_parser("clear", help="wipe the whole DB")
    p_clear.add_argument("--yes", action="store_true")
    p_clear.set_defaults(func=cmd_clear)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
