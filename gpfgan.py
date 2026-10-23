"""Compatibility shim - face restoration now lives in pipeline/stage3_restore.py.

Kept so existing imports (``from gpfgan import restore_yolo_crop``) keep
working. New code should import from ``pipeline`` directly.
"""

from pipeline.stage3_restore import (  # noqa: F401
    FaceRestorer,
    get_restorer,
    restore_yolo_crop,
)

__all__ = ["FaceRestorer", "get_restorer", "restore_yolo_crop"]


if __name__ == "__main__":
    import sys
    from pathlib import Path

    import cv2

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("test_yolo_output.png")
    crop = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if crop is None:
        raise SystemExit(f"could not read {path}")

    face = restore_yolo_crop(crop)
    if face is None:
        print("GFPGAN could not find a distinct face inside the crop.")
    else:
        print(f"restored face: {face.shape}")
        cv2.imshow("GFPGAN Restored Face", face)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
