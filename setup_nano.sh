#!/usr/bin/env bash
# One-time setup for the original Jetson Nano (JetPack 4.6 / L4T 32.7).
#
#     bash setup_nano.sh
#
# Keeps the system Python 3.6.9. Do NOT install a newer Python on this
# board: the apt OpenCV (built with CUDA + GStreamer, which RTSP capture
# needs) and NVIDIA's onnxruntime-gpu wheel are both built for 3.6, and a
# newer interpreter can use neither.
#
# Safe to re-run; every step checks before acting.

set -e
cd "$(dirname "$0")"

say()  { printf '\n=== %s\n' "$1"; }
ok()   { printf '  ok    %s\n' "$1"; }
warn() { printf '  WARN  %s\n' "$1"; }
die()  { printf '  FAIL  %s\n' "$1"; exit 1; }

# --- 1. Is this the board and OS we expect? ------------------------------
say "Platform"
if [ -r /etc/nv_tegra_release ]; then
    ok "$(head -n1 /etc/nv_tegra_release)"
    grep -q "R32" /etc/nv_tegra_release \
        || warn "not L4T R32 - this script assumes JetPack 4.6"
else
    die "/etc/nv_tegra_release not found - this is not a Jetson"
fi

PYV=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
[ "$PYV" = "3.6" ] || warn "python3 is $PYV, expected 3.6 (JetPack 4.6)"
ok "python3 $(python3 -c 'import platform; print(platform.python_version())')"

# --- 2. System packages ----------------------------------------------------
say "System packages (apt)"
sudo apt-get update -qq
# python3-opencv: JetPack's build, with CUDA and GStreamer. NEVER replace it
# with `pip install opencv-python` - that wheel has neither.
sudo apt-get install -y -qq python3-pip python3-opencv libopenblas-dev \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-libav
ok "apt packages installed"

# --- 3. pip ----------------------------------------------------------------
say "pip"
# JetPack ships pip 9, which cannot read manylinux2014 wheels or the
# environment markers in requirements-jetson.txt. 21.3.1 is the last pip
# that supports Python 3.6.
python3 -m pip install --user -q --upgrade "pip<22"
ok "pip $(python3 -m pip --version | cut -d' ' -f2)"

# --- 4. numpy needs this or it dies with "Illegal instruction" -------------
say "OpenBLAS core type"
# numpy 1.19.5's bundled OpenBLAS misdetects the Nano's Cortex-A57 and
# crashes on import with "Illegal instruction (core dumped)".
if ! grep -q OPENBLAS_CORETYPE ~/.bashrc; then
    echo 'export OPENBLAS_CORETYPE=ARMV8' >> ~/.bashrc
    ok "added OPENBLAS_CORETYPE=ARMV8 to ~/.bashrc"
else
    ok "already set in ~/.bashrc"
fi
export OPENBLAS_CORETYPE=ARMV8

# --- 5. Python packages ----------------------------------------------------
say "Python packages (pip)"
python3 -m pip install --user -q -r requirements-jetson.txt
ok "requirements-jetson.txt installed"

# --- 6. ONNX Runtime GPU ---------------------------------------------------
say "ONNX Runtime (GPU)"
if python3 -c "import onnxruntime" 2>/dev/null; then
    ok "onnxruntime $(python3 -c 'import onnxruntime as o; print(o.__version__, o.get_available_providers())')"
else
    # NVIDIA's wheel is not on PyPI. Download it from
    #   https://elinux.org/Jetson_Zoo#ONNX_Runtime
    # (the 1.11.0 entry for JetPack 4.6 / Python 3.6) and put it next to
    # this script.
    WHEEL=$(ls onnxruntime_gpu-*-cp36-cp36m-linux_aarch64.whl 2>/dev/null | head -n1 || true)
    if [ -n "$WHEEL" ]; then
        python3 -m pip install --user -q "$WHEEL"
        ok "installed $WHEEL"
    else
        warn "onnxruntime is not installed and no wheel was found here."
        warn "Download onnxruntime_gpu 1.11.0 (cp36, aarch64) from"
        warn "  https://elinux.org/Jetson_Zoo#ONNX_Runtime"
        warn "into $(pwd) and run this script again."
    fi
fi

# --- 7. Swap ---------------------------------------------------------------
say "Swap"
# 4 GB shared between CPU and GPU fills fast with two decoded streams.
if swapon --show | grep -q /swapfile; then
    ok "/swapfile already active"
else
    sudo fallocate -l 4G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile >/dev/null
    sudo swapon /swapfile
    grep -q '^/swapfile' /etc/fstab \
        || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
    ok "4 GB /swapfile created and enabled at boot"
fi

# --- 8. Verify -------------------------------------------------------------
say "Verify"
python3 - <<'EOF'
import cv2, numpy, sys
print("  python  %s" % sys.version.split()[0])
print("  numpy   %s" % numpy.__version__)
print("  opencv  %s" % cv2.__version__)
info = cv2.getBuildInformation()
gst = [l for l in info.splitlines() if "GStreamer" in l]
print("  opencv GStreamer: %s" % (gst[0].split(":")[-1].strip() if gst else "not found"))
import dataclasses
print("  dataclasses ok")
EOF

python3 check_py36.py
python3 tests/run_all.py

cat <<'EOF'

Done. Next:
  1. Copy the three models into models/ (see README, "Models to copy"):
       yolov8n.onnx  det_10g.onnx  arcface_w600k_r50.onnx
  2. python3 check_models.py      # should report CUDAExecutionProvider
  3. Optional, for full speed (needs the barrel jack + J48 jumper):
       sudo nvpmodel -m 0 && sudo jetson_clocks
  4. Optional, to free ~500 MB:
       sudo systemctl set-default multi-user.target && sudo reboot
EOF
