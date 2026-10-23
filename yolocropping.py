"""Compatibility shim - the detector now lives in pipeline/stage2_detect.py.

Kept so existing imports (``from yolocropping import get_person_crop``) keep
working. New code should import from ``pipeline`` directly.
"""

from pipeline.stage2_detect import (  # noqa: F401
    PersonDetector,
    Subject,
    get_detector,
    get_person_crop,
    get_person_crops,
)

__all__ = ["PersonDetector", "Subject", "get_detector", "get_person_crop", "get_person_crops"]


if __name__ == "__main__":
    import sys
    from pathlib import Path

    import cv2

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("test_denoised_input.png")
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"could not read {path}")

    subjects = get_person_crops(img)
    print(f"{len(subjects)} person(s) detected")
    for i, s in enumerate(subjects):
        print(f"  {i}: box={s.box} conf={s.conf:.2f} crop={s.crop.shape}")
        cv2.imshow(f"person {i}", s.crop)
    if subjects:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
