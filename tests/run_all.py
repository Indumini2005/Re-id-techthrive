"""Run every suite in this directory and report which ones failed.

    python tests/run_all.py

Each suite is a standalone script that exits non-zero on failure, so this
runs them as subprocesses rather than importing them - a suite that stubs
cv2 or onnxruntime must not leak those stubs into the next one.

The suites need no model files and no camera. They cover geometry, decoding,
synchronisation and policy: the parts where a silent error produces
plausible-looking output. For the model files themselves, run
check_models.py.
"""

from __future__ import print_function

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    suites = sorted(p for p in HERE.glob("test_*.py"))
    if not suites:
        print("no suites found in %s" % HERE)
        return 1

    width = max(len(p.name) for p in suites)
    results = []

    for path in suites:
        print("=" * (width + 30))
        print("RUN  %s" % path.name)
        print("=" * (width + 30))
        proc = subprocess.run([sys.executable, str(path)])
        results.append((path.name, proc.returncode))
        print()

    print("=" * (width + 30))
    print("SUMMARY")
    print("=" * (width + 30))
    for name, code in results:
        print("  %-*s  %s" % (width, name, "pass" if code == 0 else "FAIL (exit %d)" % code))

    failed = [n for n, c in results if c != 0]
    print()
    if failed:
        print("%d of %d suites failed: %s" % (len(failed), len(results), ", ".join(failed)))
        return 1
    print("all %d suites pass" % len(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
