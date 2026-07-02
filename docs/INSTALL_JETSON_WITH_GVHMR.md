# Install GenMO on Jetson with GVHMR

Target stack:

```text
JetPack : 6.x
CUDA    : 12.6
Arch    : aarch64
Python  : 3.10 conda env `gvhmr`
PyTorch : NVIDIA Jetson PyTorch 2.5.0 wheel
```

Do not install PyPI `torch` or `torchvision` on Jetson.

## 0. Paths

Replace every `/absolute/path/...` value with a real path:

```bash
export GVHMR_ROOT=/absolute/path/to/GVHMR
export GENMO_ROOT=/absolute/path/to/GENMO
export MODEL_ROOT="$GVHMR_ROOT/inputs/checkpoints"
export DEPS_ROOT=/absolute/path/to/ort-deps
export ORT_SRC=/absolute/path/to/onnxruntime
```

`GENMO_ROOT`, `GVHMR_ROOT`, `MODEL_ROOT`, `DEPS_ROOT`, and `ORT_SRC` do not need
to be related directories.

## 1. Start From GVHMR

Follow `$GVHMR_ROOT/docs/INSTALL_JETSON_AARCH64.md`, then verify the shared
environment:

```bash
conda activate gvhmr

python - <<'PY'
import platform
import numpy
import torch

print("arch:", platform.machine())
print("numpy:", numpy.__version__)
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
```

Expected:

```text
arch: aarch64
numpy: 1.23.5
torch cuda: 12.6
cuda available: True
```

## 2. Install GenMO

```bash
cd "$GENMO_ROOT"
conda activate gvhmr
python -m pip install -e . --no-deps

python -m pip uninstall -y nvidia-cusparselt-cu13

python -m pip install --no-deps \
  "numpy==1.23.5" \
  "opencv-python<4.12" \
  pillow scipy tqdm einops \
  scikit-image \
  transformers sentencepiece \
  wandb \
  open3d
```

If GenMO requires its metadata-pinned detector versions:

```bash
python -m pip install --no-deps "timm==0.6.7" "ultralytics==8.3.50"
```

Ignore `pip check` warnings for Jetson wheel version strings such as:

```text
torch 2.5.0a0+...
torchvision 0.20.0a0+...
```

## 3. Link Model Files

```bash
mkdir -p "$GENMO_ROOT/inputs/checkpoints/body_models/smplx"
mkdir -p "$GENMO_ROOT/inputs/checkpoints/hmr2"
mkdir -p "$GENMO_ROOT/inputs/checkpoints/vitpose"

ln -sf "$MODEL_ROOT/body_models/smplx/SMPLX_NEUTRAL.npz" \
  "$GENMO_ROOT/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz"

ln -sf "$MODEL_ROOT/hmr2/epoch=10-step=25000.ckpt" \
  "$GENMO_ROOT/inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt"

ln -sf "$MODEL_ROOT/vitpose/vitpose-h-multi-coco.pth" \
  "$GENMO_ROOT/inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth"
```

Copy GVHMR body-model helper files if they are missing in GenMO:

```bash
mkdir -p "$GENMO_ROOT/gem/utils/body_model"

for f in \
  smplx2smpl_sparse.pt \
  smpl_coco17_J_regressor.pt \
  smplx_verts437.pt \
  smpl_neutral_J_regressor.pt \
  smpl_3dpw14_J_regressor_sparse.pt
do
  cp "$GVHMR_ROOT/hmr4d/utils/body_model/$f" \
    "$GENMO_ROOT/gem/utils/body_model/$f"
done
```

## 4. Build ONNX Runtime GPU

Remove existing ONNX Runtime wheels:

```bash
conda activate gvhmr
python -m pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime_gpu
```

Install build dependencies:

```bash
sudo apt-get update
sudo apt-get install -y \
  git build-essential ninja-build python3-dev \
  libopenblas-dev zlib1g-dev

CUDNN_VER="$(dpkg-query -W -f='${Version}' libcudnn9-cuda-12)"
sudo apt-get install -y \
  "libcudnn9-dev-cuda-12=$CUDNN_VER" \
  "libcudnn9-headers-cuda-12=$CUDNN_VER"

python -m pip uninstall -y cmake
python -m pip install --force-reinstall \
  "cmake>=3.28,<4" ninja packaging wheel setuptools pybind11 psutil

cmake --version
```

Prepare ONNX Runtime:

```bash
git clone --recursive https://github.com/microsoft/onnxruntime.git "$ORT_SRC"
cd "$ORT_SRC"
git checkout v1.20.0
git submodule sync --recursive
git submodule update --init --recursive
```

Prepare local Eigen and cuDNN Frontend sources:

```bash
mkdir -p "$DEPS_ROOT"

git clone https://gitlab.com/libeigen/eigen.git "$DEPS_ROOT/eigen"
cd "$DEPS_ROOT/eigen"
git checkout e7248b26a1ed53fa030c5c459f7ea095dfd276ac

git clone --branch v1.7.0 --depth 1 \
  https://github.com/NVIDIA/cudnn-frontend.git \
  "$DEPS_ROOT/cudnn-frontend-v1.7.0"

export EIGEN_SRC="$DEPS_ROOT/eigen"
export CUDNN_FRONTEND_SRC="$DEPS_ROOT/cudnn-frontend-v1.7.0"
```

Set build variables:

```bash
export CUDA_HOME=/usr/local/cuda
export CUDNN_HOME=/usr
export ORT_CUDA_ARCH=87
export CUDNN_INCLUDE_DIR="$(dirname "$(find /usr/include /usr/include/aarch64-linux-gnu -name cudnn_version.h 2>/dev/null | head -n 1)")"
export CUDNN_LIBRARY="$(find /usr/lib/aarch64-linux-gnu /usr/local/cuda/lib64 -name 'libcudnn.so' 2>/dev/null | head -n 1)"

echo "CUDNN_INCLUDE_DIR=$CUDNN_INCLUDE_DIR"
echo "CUDNN_LIBRARY=$CUDNN_LIBRARY"
test -d "$EIGEN_SRC"
test -d "$CUDNN_FRONTEND_SRC"
```

Build the wheel:

```bash
cd "$ORT_SRC"
rm -rf build/Linux/Release/CMakeCache.txt build/Linux/Release/CMakeFiles

./build.sh \
  --config Release \
  --update \
  --build \
  --parallel 2 \
  --build_wheel \
  --skip_tests \
  --use_cuda \
  --cuda_home "$CUDA_HOME" \
  --cudnn_home "$CUDNN_HOME" \
  --use_preinstalled_eigen \
  --eigen_path "$EIGEN_SRC" \
  --cmake_path "$(which cmake)" \
  --ctest_path "$(which ctest)" \
  --cmake_extra_defines \
    CMAKE_CUDA_ARCHITECTURES="$ORT_CUDA_ARCH" \
    CUDNN_INCLUDE_DIR="$CUDNN_INCLUDE_DIR" \
    CUDNN_LIBRARY="$CUDNN_LIBRARY" \
    FETCHCONTENT_SOURCE_DIR_CUDNN_FRONTEND="$CUDNN_FRONTEND_SRC"
```

Install the wheel:

```bash
python -m pip install --force-reinstall --no-deps \
  "$ORT_SRC"/build/Linux/Release/dist/onnxruntime_gpu-*.whl
```

Verify:

```bash
python - <<'PY'
import onnxruntime as ort
print("onnxruntime:", ort.__version__)
print("providers:", ort.get_available_providers())
assert "CUDAExecutionProvider" in ort.get_available_providers()
PY
```

## 5. Verify GenMO

```bash
cd "$GENMO_ROOT"
conda activate gvhmr

python - <<'PY'
import gem
import torch
import onnxruntime as ort

print("gem: ok")
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("onnxruntime providers:", ort.get_available_providers())
PY

python tools/benchmark/benchmark_modules.py --no_imgfeat
```
