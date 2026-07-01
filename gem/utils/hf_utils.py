# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial

import os
from pathlib import Path

HF_REPO_ID = "nvidia/GEM-X"
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
DEFAULT_CKPT_FILENAME = "gem_smpl.ckpt"
DEFAULT_CKPT_DIR = "inputs/pretrained"
DEFAULT_ONNX_DIR = "inputs/onnx"
ONNX_HF_PREFIX = "gem_smpl/onnx"


def _hf_endpoint():
    return os.environ.get("GENMO_HF_ENDPOINT") or os.environ.get("HF_ENDPOINT") or DEFAULT_HF_ENDPOINT


def download_checkpoint(
    repo_id=HF_REPO_ID, filename=DEFAULT_CKPT_FILENAME, local_dir=DEFAULT_CKPT_DIR
):
    """Download checkpoint from a HuggingFace-compatible mirror if not cached."""
    local_path = Path(local_dir) / filename
    if local_path.exists():
        return str(local_path)
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
        endpoint=_hf_endpoint(),
    )
    return path


def download_onnx_model(
    name, repo_id=HF_REPO_ID, local_dir=DEFAULT_ONNX_DIR, hf_prefix=ONNX_HF_PREFIX,
):
    """Download an ONNX model and sidecar from a HuggingFace-compatible mirror.

    Args:
        name: file stem, e.g. "vitpose_coco17", "gem_smpl_denoiser_no_imgfeat".
              Both ``<name>.onnx`` and (if present) ``<name>.onnx.data`` are fetched.
        repo_id: HuggingFace repo. Defaults to ``nvidia/GEM-X``.
        local_dir: where files end up. ONNX runtime expects the .data sidecar
            next to the .onnx, so they share this directory.
        hf_prefix: subdirectory inside the repo. Defaults to ``gem_smpl/onnx``.

    Returns:
        Local path to ``<name>.onnx`` (may be skipped if already present).
    """
    final_onnx = Path(local_dir) / f"{name}.onnx"
    if final_onnx.exists():
        return str(final_onnx)

    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    Path(local_dir).mkdir(parents=True, exist_ok=True)

    onnx_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"{hf_prefix}/{name}.onnx",
        local_dir=local_dir,
        endpoint=_hf_endpoint(),
    )
    if str(Path(onnx_path).resolve()) != str(final_onnx.resolve()):
        Path(onnx_path).rename(final_onnx)

    try:
        data_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{hf_prefix}/{name}.onnx.data",
            local_dir=local_dir,
            endpoint=_hf_endpoint(),
        )
        final_data = Path(local_dir) / f"{name}.onnx.data"
        if str(Path(data_path).resolve()) != str(final_data.resolve()):
            Path(data_path).rename(final_data)
    except EntryNotFoundError:
        pass  # Model fits within ONNX's 2GB protobuf limit; no sidecar needed.

    # NOTE: previously we tidied up the empty <hf_prefix> directory tree
    # here, but that races with concurrent downloads in the same process —
    # rmdir would fire while another thread's download was still mid-flight,
    # leaving its final move-from-.incomplete with no parent directory.
    # The empty tree is cosmetic; users can wipe it manually if it bothers them.
    return str(final_onnx)
