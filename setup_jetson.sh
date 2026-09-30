#!/usr/bin/env bash
# Set up a Jetson (JetPack 6, Python 3.10) for the feeding stack, everything under deployment.root:
# the two venvs, the Depth Anything 3 checkout, the models. Idempotent; rerun after a change.
#
#   ./setup_jetson.sh                # then: make images driver depth models
#
# The torch wheels are NVIDIA's for JetPack 6 / CUDA 12.6 (jetson-ai-lab); PyPI's aarch64 torch has no CUDA.
# decord2: LocateAnything's remote code imports decord, which publishes no aarch64 wheel; decord2 does.
# docker/patches/da3_optional_exports.patch: DA3 imports its export helpers' heavy deps; inference needs none.
set -euo pipefail
cd "$(dirname "$0")"
S() { python3 -m settings get "$1"; }
ROOT="$(S deployment.root)"
LA="$(S deployment.venvs.detector)"; MP="$(S deployment.venvs.face)"
DA3="$(S deployment.paths.da3_src)"; export HF_HOME="$(S deployment.paths.hf_home)"
FACE="$(S deployment.models.face_landmarker)"
export PIP_CACHE_DIR="$ROOT/cache/pip"
mkdir -p "$ROOT" "$(S deployment.runs_dir)" "$(dirname "$FACE")" "$PIP_CACHE_DIR"
JP6=https://pypi.jetson-ai-lab.io/jp6/cu126/+f
DA3_REF=3d835ec1a5802d64a8b8b15f817a1ab54809bfe4

echo "== Depth Anything 3 -> $DA3"
[ -d "$DA3" ] || git clone -q https://github.com/ByteDance-Seed/Depth-Anything-3 "$DA3"
git -C "$DA3" checkout -q "$DA3_REF"
# inference only: DA3's export helpers (video, gaussian splats, colmap) import moviepy / open3d / ... at import
# time; this makes them optional instead of installing them (no aarch64 wheels for several)
git -C "$DA3" apply --reverse --check "$PWD/docker/patches/da3_optional_exports.patch" 2>/dev/null \
  || git -C "$DA3" apply "$PWD/docker/patches/da3_optional_exports.patch"

echo "== detector venv -> $LA"
[ -x "$LA/bin/python" ] || /usr/bin/python3.10 -m venv "$LA"
"$LA/bin/pip" install -q --upgrade pip
"$LA/bin/pip" install -q "torch @ $JP6/37d/7e156cfb4a646/torch-2.10.0-cp310-cp310-linux_aarch64.whl" \
                         "torchvision @ $JP6/1b6/357c5532db61e/torchvision-0.25.0-cp310-cp310-linux_aarch64.whl"
"$LA/bin/pip" install -q "transformers==4.57.1" "huggingface_hub==0.36.2" accelerate peft safetensors tokenizers \
    einops omegaconf addict evo imageio lmdb psutil pyyaml regex requests scipy pillow "numpy<2.3" \
    "opencv-python-headless<5" decord2
"$LA/bin/pip" install -q --no-deps -e "$DA3"

echo "== face venv -> $MP"
[ -x "$MP/bin/python" ] || /usr/bin/python3.10 -m venv "$MP"
"$MP/bin/pip" install -q --upgrade pip
"$MP/bin/pip" install -q mediapipe pyyaml pillow "opencv-python-headless<5" "numpy<2.3"

echo "== models -> $HF_HOME"
"$LA/bin/python" - <<EOF
from huggingface_hub import snapshot_download
for repo in ("nvidia/LocateAnything-3B", "$(S deployment.models.depth_anything)"):
    print("  ", repo, snapshot_download(repo))
EOF
[ -f "$FACE" ] || curl -sSL -o "$FACE" \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

echo "== check"
"$LA/bin/python" -c "import torch, transformers, decord, depth_anything_3; print('  detector: torch', torch.__version__, 'cuda', torch.cuda.is_available())"
"$MP/bin/python" -c "import mediapipe; print('  face: mediapipe', mediapipe.__version__)"
echo "done. next: make images driver depth models"
