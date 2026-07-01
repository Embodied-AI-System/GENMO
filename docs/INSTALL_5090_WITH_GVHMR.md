# Installing GenMO on Top of an Existing GVHMR Environment

This note mirrors `docs/INSTALL.md`, but assumes GenMO is installed into an existing GVHMR conda environment instead of creating a separate GenMO environment.

The current GVHMR stack is CUDA 13.1 with PyTorch 2.12. The checked environment reports `torch 2.12.1+cu130`; keep this PyTorch/CUDA stack for GenMO.

## Step 1 - Start from a working GVHMR installation

First follow the GVHMR installation instructions and make sure the GVHMR conda environment is already created and working.

Assume the environment is named `gvhmr`.

## Step 2 - Activate the GVHMR conda environment

```bash
conda activate gvhmr
```

Check that the environment is using the expected PyTorch/CUDA stack:

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
PY
```

Expected for the current setup: PyTorch 2.12.x with CUDA 13.x.

## Step 3 - Install GenMO in editable mode

From the GenMO repository root, run:

```bash
pip install -e .
```

Keep the existing GVHMR `torch`, `torchvision`, and CUDA runtime packages. If `pip` tries to replace them, use the no-dependency editable install instead:

```bash
pip install -e . --no-deps
```

Then install only the missing non-PyTorch dependencies manually, keeping the existing CUDA/PyTorch stack unchanged.

## Step 4 - Reuse GVHMR body models and checkpoints

The body model and demo checkpoints may already exist in the GVHMR setup, so they do not need to be downloaded again. Copy the corresponding files into the paths required by `docs/INSTALL.md`.

For the current local layout, the shared model directory is a sibling of both `GVHMR` and `GENMO`:

```bash
export MODEL_ROOT=../models
```

If your GVHMR checkout stores checkpoints under `GVHMR/inputs/checkpoints`, use this instead:

```bash
export MODEL_ROOT=/path/to/GVHMR/inputs/checkpoints
```

From the GenMO repository root, copy the files:

```bash
mkdir -p inputs/checkpoints/body_models/smplx
mkdir -p inputs/checkpoints/hmr2
mkdir -p inputs/checkpoints/vitpose

cp "$MODEL_ROOT/body_models/smplx/SMPLX_NEUTRAL.npz" \
  inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz

cp "$MODEL_ROOT/hmr2/epoch=10-step=25000.ckpt" \
  inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt

cp "$MODEL_ROOT/vitpose/vitpose-h-multi-coco.pth" \
  inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth
```

### Copy GVHMR body-model resource `.pt` files

GenMO also needs GVHMR's lightweight SMPL/SMPL-X regressor resources for `EnDecoder` and rendering. Copy them from the GVHMR checkout:

```bash
export GVHMR_ROOT=../GVHMR

cp "$GVHMR_ROOT/hmr4d/utils/body_model/smplx2smpl_sparse.pt" \
  gem/utils/body_model/smplx2smpl_sparse.pt

cp "$GVHMR_ROOT/hmr4d/utils/body_model/smpl_coco17_J_regressor.pt" \
  gem/utils/body_model/smpl_coco17_J_regressor.pt

cp "$GVHMR_ROOT/hmr4d/utils/body_model/smplx_verts437.pt" \
  gem/utils/body_model/smplx_verts437.pt

cp "$GVHMR_ROOT/hmr4d/utils/body_model/smpl_neutral_J_regressor.pt" \
  gem/utils/body_model/smpl_neutral_J_regressor.pt

cp "$GVHMR_ROOT/hmr4d/utils/body_model/smpl_3dpw14_J_regressor_sparse.pt" \
  gem/utils/body_model/smpl_3dpw14_J_regressor_sparse.pt
```

After copying, the GenMO paths should match the expected layout:

```text
inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz
inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt
inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth
gem/utils/body_model/smplx2smpl_sparse.pt
gem/utils/body_model/smpl_coco17_J_regressor.pt
gem/utils/body_model/smplx_verts437.pt
gem/utils/body_model/smpl_neutral_J_regressor.pt
gem/utils/body_model/smpl_3dpw14_J_regressor_sparse.pt
```

## Step 5 - Optional: install ONNX Runtime for the webcam demo

This step is only required for `scripts/demo/demo_webcam.py`. The offline video demos (`demo_smpl.py`, `demo_smpl_hpe.py`) do not need it.

For GPU ONNX Runtime on the current CUDA 13.1 environment, use the official CUDA 13 nightly wheel:

```bash
pip uninstall -y onnxruntime-gpu onnxruntime
pip install coloredlogs flatbuffers numpy packaging protobuf sympy
pip install --pre --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/ onnxruntime-gpu
```

For CPU-only ONNX Runtime:

```bash
pip install onnxruntime
```

If ONNX Runtime reports missing CUDA 12 libraries such as `libcublasLt.so.12`, `libcublas.so.12`, `libnvrtc.so.12`, or `libcudart.so.12`, the installed `onnxruntime-gpu` wheel is a CUDA 12 build. Replace it with the CUDA 13 nightly wheel above.

Verify the ONNX Runtime package and providers:

```bash
python - <<'PY'
import torch
import onnxruntime as ort

if hasattr(ort, "preload_dlls"):
    ort.preload_dlls()

print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("onnxruntime:", ort.__version__)
print("providers:", ort.get_available_providers())
PY
```

For GPU inference, the output should include `CUDAExecutionProvider` and should not print CUDA 12 library load errors.

## Step 6 - Optional: install Viser for global 3D rendering

Required only for `--render_mode viser` in the webcam demo. The OpenCV mesh-overlay mode (`--render_mode opencv`) does not need this.

```bash
pip install viser
```

## Step 7 - ONNX models for the webcam demo

The webcam demo loads four ONNX models. On first run, missing models are auto-downloaded from `nvidia/GEM-X` through the default HuggingFace-compatible mirror endpoint `https://hf-mirror.com` into `inputs/onnx/`. YOLOX is auto-downloaded from the OpenMMLab CDN.

To use a different endpoint, set `GENMO_HF_ENDPOINT` before running the demo:

```bash
export GENMO_HF_ENDPOINT=https://hf-mirror.com
```

| File | Size | Used when |
|---|---:|---|
| `gem_smpl_denoiser.onnx` | ~1.7 GB | default mode with HMR2 features |
| `gem_smpl_denoiser_no_imgfeat.onnx` | ~1.7 GB | `--no_imgfeat` mode |
| `vitpose_coco17.onnx` + `.onnx.data` | ~2.5 GB | always |
| `hmr2.onnx` + `.onnx.data` | ~2.7 GB | default mode; skipped under `--no_imgfeat` |

Total size is about 8.7 GB. Subsequent runs use the cached files.

### Re-exporting locally, optional

If you fine-tune a checkpoint or need a different `seq_len`, re-export with the scripts under `tools/export/`:

```bash
python tools/export/export_denoiser_onnx.py --ckpt <path/to/gem_smpl.ckpt> --exp gem_smpl
python tools/export/export_denoiser_onnx.py --ckpt <path/to/gem_smpl.ckpt> --exp gem_smpl --no-imgfeat
python tools/export/export_vitpose_onnx.py
python tools/export/export_hmr2_onnx.py
```

Each denoiser ONNX bakes in `seq_len=120` from the export trace. `demo_webcam.py --context_frames` must match this value unless the ONNX model is re-exported with a different sequence length.

## Step 8 - Verify the installation

```bash
python -c "import gem; print('Installation successful')"
```

Optional benchmark:

```bash
python tools/benchmark/benchmark_modules.py --no_imgfeat
```
