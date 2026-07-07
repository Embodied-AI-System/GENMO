# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""Real-time GEM-SMPL webcam demo using ONNX inference.

Runs the GEM-SMPL pipeline frame-by-frame with a sliding window:
  YOLOX detection + ByteTrack  -->  ViTPose-H 2D keypoints (COCO-17, ONNX)
  -->  HMR2 ViT image features (ONNX, optional)
  -->  GEM-SMPL denoiser (ONNX)  -->  streaming rollout

Usage::

    # From video file
    python scripts/demo/demo_webcam.py --video inputs/demo/walk.mp4

    # From webcam, no image features (faster)
    python scripts/demo/demo_webcam.py --camera_id 0 --no_imgfeat

    # With Viser web rendering of global view
    python scripts/demo/demo_webcam.py --video inputs/demo/walk.mp4 \
        --render --render_mode viser --render_port 8012

    # OpenCV in-camera mesh overlay
    python scripts/demo/demo_webcam.py --video inputs/demo/walk.mp4 \
        --render --render_mode opencv
"""
# ruff: noqa: E402, I001
import argparse
import csv
import html
import multiprocessing as mp
import queue as _queue_mod
import sys
import threading
import time
from collections import deque
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Preload cuDNN 9 from pip nvidia-cudnn so ONNX Runtime CUDA EP works.
try:
    import nvidia.cudnn as _cudnn
    import ctypes as _ctypes
    if _cudnn.__file__ is not None:
        _cudnn_lib = str(Path(_cudnn.__file__).parent / "lib" / "libcudnn.so.9")
        _ctypes.cdll.LoadLibrary(_cudnn_lib)
except (ImportError, OSError):
    pass

import cv2
import numpy as np
import torch

# Wrap torch.load for numpy 1.x / 2.x compat.
_original_torch_load = torch.load
_need_numpy_shim = not hasattr(np, "_core")
def _compat_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    if _need_numpy_shim:
        import numpy.core as _np_core
        _added = {}
        for mod in ("numpy._core", "numpy._core.multiarray"):
            if mod not in sys.modules:
                target = _np_core if mod == "numpy._core" else getattr(_np_core, "multiarray", _np_core)
                sys.modules[mod] = target
                _added[mod] = True
        try:
            return _original_torch_load(*args, **kwargs)
        finally:
            for mod in _added:
                sys.modules.pop(mod, None)
    return _original_torch_load(*args, **kwargs)
torch.load = _compat_torch_load

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Make sibling demo modules importable when running this file directly.
_DEMO_DIR = Path(__file__).resolve().parent
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

from gem.utils.cam_utils import compute_transl_full_cam, estimate_K
from gem.utils.geo_transform import compute_cam_angvel, get_bbx_xys_from_xyxy
from gem.utils.motion_utils import init_rollout_w_Rt_state, rollout_step_w_Rt
from gem.utils.pylogger import Log

from onnx_runners import (
    hmr2_preprocess_256x256,
    load_denoiser,
    load_hmr2,
    load_vitpose,
    load_yolox,
    run_hmr2_single_frame,
    run_vitpose_single_frame,
)


if torch.cuda.is_available():
    _DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    _DEVICE = torch.device("mps")
else:
    _DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
#  Runtime pipeline profiler
# ---------------------------------------------------------------------------


class PipelineProfiler:
    """Collect per-stage wall-time events and export CSV + SVG timeline."""

    ROWS = [
        ("capture", "Stage 1: cap.read"),
        ("yolox", "Stage 2: YOLOX"),
        ("bytetrack", "Stage 2: ByteTrack"),
        ("bbox_convert", "Stage 3: bbox convert"),
        ("vitpose", "Stage 4: ViTPose"),
        ("hmr2", "Stage 4: HMR2"),
        ("window", "Stage 5: sliding window"),
        ("denoiser", "Stage 6: denoiser"),
        ("decode", "Stage 7: decode"),
        ("rollout", "Stage 8: rollout"),
        ("bbox_feedback", "Stage 9: bbox feedback"),
        ("render_enqueue", "Stage 10: render enqueue"),
    ]
    COLORS = {
        "capture": "#9e9e9e",
        "yolox": "#1976d2",
        "bytetrack": "#64b5f6",
        "bbox_convert": "#90caf9",
        "vitpose": "#7b1fa2",
        "hmr2": "#ba68c8",
        "window": "#388e3c",
        "denoiser": "#f57c00",
        "decode": "#ffb74d",
        "rollout": "#d84315",
        "bbox_feedback": "#fbc02d",
        "render_enqueue": "#0288d1",
    }

    def __init__(self, enabled=False, output_prefix=None, source_name="demo",
                 plot_frames=300, sync_cuda=False):
        self.enabled = bool(enabled)
        self.output_prefix = output_prefix
        self.source_name = source_name
        self.plot_frames = int(plot_frames)
        self.sync_cuda = bool(sync_cuda)
        self.events = []
        self.frames = []
        self.lock = threading.Lock()
        self.t0 = time.perf_counter()
        self.wall_t0 = time.time()

    def _sync(self):
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _now(self):
        return time.perf_counter() - self.t0

    @contextmanager
    def record(self, stage, frame_id=None, window_start=None, window_end=None, note=""):
        if not self.enabled:
            yield
            return

        self._sync()
        start = self._now()
        try:
            yield
        finally:
            self._sync()
            end = self._now()
            with self.lock:
                self.events.append({
                    "stage": stage,
                    "frame_id": "" if frame_id is None else int(frame_id),
                    "window_start": "" if window_start is None else int(window_start),
                    "window_end": "" if window_end is None else int(window_end),
                    "start_s": start,
                    "end_s": end,
                    "wall_start_s": self.wall_t0 + start,
                    "wall_end_s": self.wall_t0 + end,
                    "duration_ms": (end - start) * 1000.0,
                    "thread": threading.current_thread().name,
                    "note": note,
                })

    def mark_frame(self, frame_id, start_s, end_s):
        if not self.enabled:
            return
        with self.lock:
            self.frames.append({
                "frame_id": int(frame_id),
                "start_s": float(start_s),
                "end_s": float(end_s),
            })

    def export(self):
        if not self.enabled:
            return None

        output_prefix = self.output_prefix
        if output_prefix is None:
            output_prefix = Path("outputs/profile") / f"{self.source_name}_pipeline_profile"
        else:
            output_prefix = Path(output_prefix)
            if output_prefix.suffix:
                output_prefix = output_prefix.with_suffix("")

        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        csv_path = output_prefix.with_suffix(".csv")
        svg_path = output_prefix.with_suffix(".svg")
        summary_path = output_prefix.with_suffix(".summary.txt")

        with self.lock:
            events = list(self.events)
            frames = list(self.frames)

        self._write_csv(csv_path, events)
        summary = self._write_summary(summary_path, events)
        self._write_svg(svg_path, events, frames, summary)
        return csv_path, svg_path, summary_path, summary

    def _write_csv(self, path, events):
        fields = [
            "stage", "frame_id", "window_start", "window_end",
            "start_s", "end_s", "wall_start_s", "wall_end_s",
            "duration_ms", "thread", "note",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for event in events:
                writer.writerow(event)

    def _summarize(self, events):
        by_stage = {}
        for event in events:
            by_stage.setdefault(event["stage"], []).append(event["duration_ms"])

        rows = []
        for stage, durations in by_stage.items():
            durations = sorted(durations)
            n = len(durations)
            p50 = durations[int((n - 1) * 0.50)]
            p95 = durations[int((n - 1) * 0.95)]
            total = sum(durations)
            rows.append({
                "stage": stage,
                "count": n,
                "total_ms": total,
                "mean_ms": total / n,
                "p50_ms": p50,
                "p95_ms": p95,
                "max_ms": durations[-1],
            })
        return sorted(rows, key=lambda row: row["total_ms"], reverse=True)

    def _write_summary(self, path, events):
        summary = self._summarize(events)
        with open(path, "w", encoding="utf-8") as f:
            f.write("stage,count,total_ms,mean_ms,p50_ms,p95_ms,max_ms\n")
            for row in summary:
                f.write(
                    f"{row['stage']},{row['count']},{row['total_ms']:.3f},"
                    f"{row['mean_ms']:.3f},{row['p50_ms']:.3f},"
                    f"{row['p95_ms']:.3f},{row['max_ms']:.3f}\n"
                )
        return summary

    def _write_svg(self, path, events, frames, summary):
        row_order = [stage for stage, _label in self.ROWS]
        row_labels = dict(self.ROWS)
        frames = sorted(frames, key=lambda item: item["frame_id"])
        if frames and self.plot_frames > 0 and len(frames) > self.plot_frames:
            frames = frames[-self.plot_frames:]

        if frames:
            first_frame = frames[0]["frame_id"]
            last_frame = frames[-1]["frame_id"]
            t_min = frames[0]["start_s"]
            t_max = max(frames[-1]["end_s"], frames[-1]["start_s"])
        elif events:
            first_frame = 0
            last_frame = 0
            t_min = min(event["start_s"] for event in events)
            t_max = max(event["end_s"] for event in events)
        else:
            first_frame = 0
            last_frame = 0
            t_min = 0.0
            t_max = 1.0

        def event_in_frame_range(event):
            if event["frame_id"] == "":
                return False
            return first_frame <= int(event["frame_id"]) <= last_frame

        visible_events = [
            event for event in events
            if event_in_frame_range(event)
            or (event["end_s"] >= t_min and event["start_s"] <= t_max)
        ]
        if visible_events:
            t_min = min(t_min, min(event["start_s"] for event in visible_events))
            t_max = max(t_max, max(event["end_s"] for event in visible_events))
        t_span = max(t_max - t_min, 1e-6)

        px_per_second = 120
        if t_span * px_per_second < 900:
            px_per_second = 900 / t_span
        if t_span * px_per_second > 5000:
            px_per_second = 5000 / t_span

        def time_to_x(t):
            return left + (t - t_min) * px_per_second

        left = 210
        top = 95
        row_h = 38
        plot_width = max(900, t_span * px_per_second)
        width = int(left + plot_width + 260)
        height = top + len(row_order) * row_h + 170

        tick_step_s = 0.1
        while t_span / tick_step_s > 80:
            tick_step_s *= 2 if tick_step_s < 1.0 else 5

        svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">',
            '<style>',
            'text{font-family:Arial,Helvetica,sans-serif;font-size:12px;fill:#222}',
            '.small{font-size:11px;fill:#555}',
            '.title{font-size:22px;font-weight:700}',
            '.axis{stroke:#333;stroke-width:1}',
            '.grid{stroke:#e5e5e5;stroke-width:1}',
            '.rowline{stroke:#eeeeee;stroke-width:1}',
            '</style>',
            '<rect width="100%" height="100%" fill="#ffffff"/>',
            '<text x="30" y="34" class="title">demo_webcam.py pipeline profile</text>',
            f'<text x="30" y="58" class="small">x axis = wall time '
            f'({t_min:.3f}s..{t_max:.3f}s); frame-start ticks mark processed frames '
            f'{first_frame}..{last_frame}; labels show latency in ms</text>',
        ]

        if summary:
            top_stage = summary[0]
            svg.append(
                f'<text x="30" y="78" class="small">Top cumulative bottleneck: '
                f'{html.escape(top_stage["stage"])} '
                f'({top_stage["total_ms"]:.1f} ms total, '
                f'{top_stage["mean_ms"]:.1f} ms mean)</text>'
            )

        for row_idx, stage in enumerate(row_order):
            y = top + row_idx * row_h
            svg.append(f'<line x1="{left}" y1="{y + row_h}" x2="{width - 40}" y2="{y + row_h}" class="rowline"/>')
            svg.append(
                f'<text x="20" y="{y + 24}">{html.escape(row_labels.get(stage, stage))}</text>'
            )

        tick = (int(t_min / tick_step_s) * tick_step_s)
        while tick <= t_max + 1e-9:
            if tick >= t_min - 1e-9:
                x = time_to_x(tick)
                svg.append(f'<line x1="{x:.2f}" y1="{top - 8}" x2="{x:.2f}" y2="{top + len(row_order) * row_h}" class="grid"/>')
                svg.append(f'<line x1="{x:.2f}" y1="{top - 16}" x2="{x:.2f}" y2="{top - 2}" class="axis"/>')
                svg.append(f'<text x="{x - 14:.2f}" y="{top - 22}" class="small">{tick - t_min:.1f}s</text>')
            tick += tick_step_s

        frame_tick_y1 = top + len(row_order) * row_h + 8
        frame_tick_y2 = frame_tick_y1 + 12
        svg.append(f'<text x="20" y="{frame_tick_y2}" class="small">frame starts</text>')
        frame_label_step = 1
        if len(frames) > 80:
            frame_label_step = max(1, len(frames) // 40)
        for idx, frame in enumerate(frames):
            x = time_to_x(frame["start_s"])
            svg.append(f'<line x1="{x:.2f}" y1="{frame_tick_y1}" x2="{x:.2f}" y2="{frame_tick_y2}" stroke="#555" stroke-width="1"><title>frame {frame["frame_id"]} start {frame["start_s"]:.6f}s</title></line>')
            if idx % frame_label_step == 0:
                svg.append(f'<text x="{x - 8:.2f}" y="{frame_tick_y2 + 14}" class="small">{frame["frame_id"]}</text>')

        svg.append(f'<line x1="{left}" y1="{top - 2}" x2="{width - 40}" y2="{top - 2}" class="axis"/>')

        for event in events:
            stage = event["stage"]
            if stage not in row_order:
                continue
            if event["end_s"] < t_min or event["start_s"] > t_max:
                continue
            x = time_to_x(max(event["start_s"], t_min))
            x_end = time_to_x(min(event["end_s"], t_max))
            w = max(2.0, x_end - x)
            y = top + row_order.index(stage) * row_h + 8
            color = self.COLORS.get(stage, "#777777")
            duration = event["duration_ms"]
            frame_label = event["frame_id"]
            title = (
                f'{stage} frame={frame_label} duration={duration:.3f}ms '
                f'start={event["start_s"]:.6f}s end={event["end_s"]:.6f}s'
            )
            svg.append(
                f'<rect x="{x:.2f}" y="{y}" width="{w:.2f}" height="20" rx="3" '
                f'fill="{color}" opacity="0.86"><title>{html.escape(title)}</title></rect>'
            )
            if w > 38:
                svg.append(
                    f'<text x="{x + 4:.2f}" y="{y + 14}" fill="#ffffff" '
                    f'font-size="10">{duration:.1f}ms</text>'
                )

        legend_x = left
        legend_y = top + len(row_order) * row_h + 45
        for idx, (stage, label) in enumerate(self.ROWS):
            lx = legend_x + (idx % 4) * 290
            ly = legend_y + (idx // 4) * 24
            svg.append(
                f'<rect x="{lx}" y="{ly - 12}" width="14" height="14" '
                f'fill="{self.COLORS.get(stage, "#777777")}"/>'
            )
            svg.append(f'<text x="{lx + 20}" y="{ly}" class="small">{html.escape(label)}</text>')

        svg.append("</svg>")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(svg))


# ---------------------------------------------------------------------------
#  Denoiser inference helper
# ---------------------------------------------------------------------------


def _run_preproc_paired(
    vitpose_runner, vitpose_backend,
    hmr2_runner, hmr2_backend,
    frame_bgr, bbx_xys, no_imgfeat,
    profiler=None, frame_id=None,
):
    """Run ViTPose + (optionally) HMR2 and return a four-tuple paired with inputs.

    Returns (kp2d, f_img, bbx_xys, frame_bgr) — all describing the same logical
    frame so the sliding windows stay self-consistent under async pipelining.
    Combining the two models into a single future trades a small amount of
    parallelism (~10 ms on GPU when both are ONNX) for window-level alignment
    between kp2d, image features, bbox, and the rendered image.
    """
    profile = profiler if profiler is not None else PipelineProfiler(enabled=False)
    with profile.record("vitpose", frame_id=frame_id, note=vitpose_backend):
        kp2d = run_vitpose_single_frame(vitpose_runner, vitpose_backend, frame_bgr, bbx_xys)
    if no_imgfeat or hmr2_runner is None:
        f_img = torch.zeros(1024)
    else:
        frame_rgb = frame_bgr[..., ::-1].copy()
        with profile.record("hmr2", frame_id=frame_id, note=hmr2_backend):
            f_img = run_hmr2_single_frame(hmr2_runner, hmr2_backend, frame_rgb, bbx_xys)
    return kp2d, f_img, bbx_xys, frame_bgr


def run_denoiser(runner, _backend, batch):
    """Run GEM-SMPL denoiser. Returns (pred_x, pred_cam) on the active device."""
    out = runner(
        obs=batch["obs"],
        bbx_xys=batch["bbx_xys"],
        K_fullimg=batch["K_fullimg"],
        f_imgseq=batch["f_imgseq"],
        f_cam_angvel=batch["f_cam_angvel"],
    )
    pred_x = out["pred_x"]
    pred_cam = out["pred_cam"]
    if isinstance(pred_x, torch.Tensor):
        pred_x = pred_x.to(_DEVICE)
        pred_cam = pred_cam.to(_DEVICE)
    return pred_x, pred_cam


# ---------------------------------------------------------------------------
#  Background render workers (mp.Process)
# ---------------------------------------------------------------------------


def _render_worker_viser(render_queue, width, height, port):
    """Viser-based web renderer (runs in a child process). Shows global world view."""
    import viser

    sys.path.insert(0, str(PROJECT_ROOT))
    from gem.utils.smplx_utils import make_smplx

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    body_model = make_smplx("supermotion")
    body_model = body_model.to(device).eval()
    faces_np = np.asarray(body_model.faces).astype(np.uint32)

    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.scene.set_up_direction("+y")
    server.scene.add_grid("/ground", width=10.0, height=10.0, plane="xz")
    mesh_handle = None

    print(f"[Render] Viser server running at http://localhost:{port}")

    while True:
        try:
            data = render_queue.get(timeout=2.0)
        except Exception:
            continue

        if data is None:
            break

        body_params = {
            k: v.to(device) for k, v in data["body_params_global"].items()
        }
        # Merge body_pose + betas from in-camera decode (rollout only emits global_orient/transl)
        for key in ("body_pose", "betas"):
            if key not in body_params and key in data["body_params_incam"]:
                body_params[key] = data["body_params_incam"][key].to(device)

        with torch.no_grad():
            verts = body_model(**body_params).vertices  # (1, V, 3)
        verts_np = verts[0].detach().cpu().numpy().astype(np.float32)

        if mesh_handle is None:
            mesh_handle = server.scene.add_mesh_simple(
                "/body", vertices=verts_np, faces=faces_np,
                color=(102, 204, 102), flat_shading=False, side="double",
            )
        else:
            mesh_handle.vertices = verts_np

    server.stop()


def _render_mesh_opencv(verts_cam, faces, K, frame_bgr, color=(102, 204, 102), alpha=0.6):
    """Render a mesh overlay using pure OpenCV (no OpenGL/EGL needed)."""
    proj = verts_cam @ K.T  # (V, 3)
    z = proj[:, 2:3]
    z_safe = np.where(z > 0.01, z, 0.01)
    uv = (proj[:, :2] / z_safe).astype(np.int32)

    face_z = verts_cam[faces, 2]  # (F, 3)
    visible = (face_z > 0.1).all(axis=1)

    face_pts = uv[faces]
    face_pts = face_pts[visible]

    if len(face_pts) == 0:
        return frame_bgr

    v0, v1, v2 = face_pts[:, 0], face_pts[:, 1], face_pts[:, 2]
    cross = (
        (v1[:, 0] - v0[:, 0]).astype(np.int64) * (v2[:, 1] - v0[:, 1]).astype(np.int64)
        - (v1[:, 1] - v0[:, 1]).astype(np.int64) * (v2[:, 0] - v0[:, 0]).astype(np.int64)
    )
    front = cross > 0
    face_pts = face_pts[front]

    if len(face_pts) == 0:
        return frame_bgr

    overlay = frame_bgr.copy()
    cv2.fillPoly(overlay, face_pts, color)
    return cv2.addWeighted(overlay, alpha, frame_bgr, 1.0 - alpha, 0)


def _render_worker_cv2(render_queue, width, height, _port, display_queue=None):
    """OpenCV mesh overlay renderer (runs in a child process)."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from gem.utils.smplx_utils import make_smplx

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    body_model = make_smplx("supermotion")
    body_model = body_model.to(device).eval()
    faces_np = np.asarray(body_model.faces).astype(np.int64)

    K_np = None

    print("[Render] OpenCV overlay renderer starting...")

    while True:
        try:
            data = render_queue.get(timeout=2.0)
        except Exception:
            continue

        if data is None:
            break

        body_params = {
            k: v.to(device) for k, v in data["body_params_incam"].items()
        }

        with torch.no_grad():
            verts = body_model(**body_params).vertices
        verts_np = verts[0].detach().cpu().numpy().astype(np.float32)

        if K_np is None:
            K = data["K_fullimg"]
            K_np = K.numpy() if hasattr(K, "numpy") else np.array(K)

        composite = _render_mesh_opencv(verts_np, faces_np, K_np, data["frame_bgr"])

        if display_queue is not None:
            try:
                display_queue.put_nowait(composite)
            except _queue_mod.Full:
                pass
        else:
            cv2.imshow("GEM-SMPL Webcam", composite)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if display_queue is None:
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
#  WebcamGEMSMPLDemo
# ---------------------------------------------------------------------------


class WebcamGEMSMPLDemo:
    """Real-time GEM-SMPL pose estimation with sliding-window rollout."""

    def __init__(self, args):
        torch.backends.cudnn.benchmark = True

        self.context_frames = args.context_frames
        self.yolo_period = args.yolo_period
        self.vitpose_period = args.vitpose_period
        self.no_imgfeat = args.no_imgfeat
        self.render_enabled = args.render

        self._async = args.async_pipeline and not args.no_async_pipeline
        Log.info(
            f"[Config] device={_DEVICE}, context_frames={self.context_frames}, "
            f"yolo_period={self.yolo_period}, vitpose_period={self.vitpose_period}, "
            f"no_imgfeat={self.no_imgfeat}, async={self._async}"
        )

        # --- Video capture ---
        if args.video is not None:
            self.cap = cv2.VideoCapture(args.video)
            self.source_name = Path(args.video).stem
        else:
            self.cap = cv2.VideoCapture(args.camera_id)
            self.source_name = f"cam{args.camera_id}"
        assert self.cap.isOpened(), "Failed to open video source"

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        Log.info(f"[Source] {self.source_name}: {self.width}x{self.height} @ {self.fps:.1f} fps")

        self.profiler = PipelineProfiler(
            enabled=args.profile_pipeline,
            output_prefix=args.profile_output,
            source_name=self.source_name,
            plot_frames=args.profile_plot_frames,
            sync_cuda=args.profile_sync_cuda,
        )
        if self.profiler.enabled:
            Log.info(
                f"[Profile] Enabled | output={args.profile_output or 'outputs/profile'} | "
                f"plot_frames={args.profile_plot_frames} | sync_cuda={args.profile_sync_cuda}"
            )

        # --- Camera intrinsics ---
        self.K_fullimg = estimate_K(self.width, self.height)  # (3, 3)
        self._K_fullimg_cpu = self.K_fullimg.cpu()

        # --- Static camera angular velocity (identity rotation) ---
        R_eye = torch.eye(3).unsqueeze(0).repeat(self.context_frames, 1, 1)
        self._cam_angvel_static = compute_cam_angvel(R_eye)  # (context_frames, 6)

        # --- Pre-allocated zero tensor for f_imgseq when no_imgfeat ---
        self._f_imgseq_zeros = torch.zeros(1, self.context_frames, 1024)

        # --- Sliding windows ---
        self.bbx_xys_window = deque(maxlen=self.context_frames)
        self.kp2d_window = deque(maxlen=self.context_frames)
        self.f_imgseq_window = deque(maxlen=self.context_frames)

        # --- State ---
        self.frame_index = 0
        self.bbox_xyxy = None
        self.rollout_state = None
        self.primary_track_id = None
        self._last_kp2d = None

        # --- Async executors ---
        self._denoiser_executor = ThreadPoolExecutor(max_workers=1) if self._async else None
        self._denoiser_future = None
        self._pending_result = None
        # Single preproc future runs ViTPose + (optionally) HMR2 together so
        # the resulting bundle (kp2d, f_img, bbx_xys, frame_bgr) is paired —
        # all four describe the same logical frame.
        self._preproc_executor = ThreadPoolExecutor(max_workers=1) if self._async else None
        self._preproc_future = None
        self._last_f_img = torch.zeros(1024)
        self._last_bbx_xys = None
        self._last_frame_bgr = None

        # --- Models ---
        self._load_models()

        # --- Background render process ---
        self._render_queue = None
        self._render_proc = None
        self._display_queue = None
        if self.render_enabled:
            render_mode = args.render_mode
            render_port = args.render_port
            # Fail loud in the main process if a render-backend dependency is
            # missing — otherwise the subprocess traceback gets eaten.
            if render_mode == "viser":
                try:
                    import viser  # noqa: F401
                except ImportError as e:
                    raise RuntimeError(
                        "Viser renderer requested but `viser` is not installed. "
                        "Install with: pip install viser"
                    ) from e
            worker_fn = _render_worker_viser if render_mode == "viser" else _render_worker_cv2
            ctx = mp.get_context("spawn")
            self._render_queue = ctx.Queue(maxsize=2)
            worker_args = (self._render_queue, self.width, self.height, render_port)
            if render_mode == "opencv" and sys.platform == "darwin":
                self._display_queue = ctx.Queue(maxsize=2)
                worker_args = (*worker_args, self._display_queue)
            self._render_proc = ctx.Process(
                target=worker_fn, args=worker_args, daemon=True,
            )
            self._render_proc.start()
            Log.info(
                f"[Render] Background {render_mode} render process started "
                f"(pid={self._render_proc.pid})"
            )

    def _load_models(self):
        """Load YOLOX, ViTPose, HMR2, denoiser, and the EnDecoder in parallel."""
        Log.info("[Init] Loading models...")
        t0 = time.monotonic()

        def _load_yolox_local():
            return load_yolox()

        def _load_endecoder_local():
            from gem.network.endecoder import EnDecoder
            enc = EnDecoder(
                stats_name="MM_V1_AMASS_LOCAL_BEDLAM_CAM",
                encode_type="gvhmr",
                feat_dim=151,
                clip_std=True,
            )
            enc.build_obs_indices_dict()
            return enc.eval().to(_DEVICE)

        with ThreadPoolExecutor(max_workers=4) as pool:
            fut_yolox = pool.submit(_load_yolox_local)
            fut_vp = pool.submit(load_vitpose)
            fut_dn = pool.submit(load_denoiser, no_imgfeat=self.no_imgfeat)
            fut_ed = pool.submit(_load_endecoder_local)
            fut_hm = (
                pool.submit(load_hmr2)
                if not self.no_imgfeat else None
            )

            self.yolox, self.tracker, _ = fut_yolox.result()
            Log.info("[Init] YOLOX + ByteTrack ready")

            self.vitpose_runner, self.vitpose_backend = fut_vp.result()
            Log.info(f"[Init] ViTPose: {self.vitpose_backend}")

            self.denoiser_runner, self.denoiser_backend = fut_dn.result()
            if self.denoiser_runner is None:
                raise RuntimeError(
                    "No ONNX/TRT denoiser found. Export first:\n"
                    "  python tools/export/export_denoiser_onnx.py "
                    f"--ckpt <path>{' --no-imgfeat' if self.no_imgfeat else ''}"
                )
            Log.info(f"[Init] Denoiser: {self.denoiser_backend}")

            self.endecoder = fut_ed.result()
            Log.info("[Init] EnDecoder ready")

            if fut_hm is not None:
                self.hmr2_runner, self.hmr2_backend = fut_hm.result()
                Log.info(f"[Init] HMR2: {self.hmr2_backend}")
            else:
                self.hmr2_runner, self.hmr2_backend = None, "none"

        Log.info(f"[Init] All models loaded in {time.monotonic() - t0:.2f}s")

        self._warmup()

    def _warmup(self):
        """Run dummy inference to trigger graph optimization.

        Note: the denoiser ONNX bakes seq_len from export into a constant in
        a text cross-attention reshape, so warmup must use the same L as
        inference (== context_frames). Re-export with --seq_len <N> if you
        want to use a different context_frames at runtime.
        """
        Log.info("[Init] Warming up...")
        t0 = time.monotonic()
        F = self.context_frames

        def _warmup_vp():
            dummy = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            dummy_bbx = torch.tensor([self.width / 2, self.height / 2, 200.0])
            run_vitpose_single_frame(self.vitpose_runner, self.vitpose_backend, dummy, dummy_bbx)

        def _warmup_dn():
            dummy_batch = {
                "obs": torch.zeros(1, F, 17, 3),
                "bbx_xys": torch.zeros(1, F, 3),
                "K_fullimg": self.K_fullimg.unsqueeze(0).unsqueeze(0).repeat(1, F, 1, 1),
                "f_imgseq": torch.zeros(1, F, 1024),
                "f_cam_angvel": self._cam_angvel_static[:F].unsqueeze(0),
            }
            run_denoiser(self.denoiser_runner, self.denoiser_backend, dummy_batch)

        def _warmup_hm():
            if self.hmr2_runner is None:
                return
            dummy = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            dummy_bbx = torch.tensor([self.width / 2, self.height / 2, 200.0])
            run_hmr2_single_frame(self.hmr2_runner, self.hmr2_backend, dummy, dummy_bbx)

        with ThreadPoolExecutor(max_workers=3) as pool:
            f1 = pool.submit(_warmup_vp)
            f2 = pool.submit(_warmup_dn)
            f3 = pool.submit(_warmup_hm)
            for f in (f1, f2, f3):
                f.result()
        Log.info(f"[Init] Warmup done in {time.monotonic() - t0:.2f}s")

    # ------------------------------------------------------------------
    #  Backend: denoiser → decode → rollout (runs in thread when async)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _run_backend(
        self,
        bbx_xys_seq,
        kp2d_seq,
        f_imgseq_seq,
        kp2d_last,
        frame_bgr=None,
        frame_id=None,
        window_start=None,
        window_end=None,
    ):
        """Denoiser + decode + streaming rollout.

        ``frame_bgr`` is paired into the result so the renderer can show the
        image that corresponds to the predicted mesh — without this, the
        async pipeline would render the latest camera frame while the mesh
        is several frames behind.
        """
        timings = {}
        F_len = bbx_xys_seq.shape[0]
        K_fullimg_seq = self.K_fullimg.unsqueeze(0).repeat(F_len, 1, 1)
        cam_angvel = self._cam_angvel_static[:F_len]

        batch = {
            "obs": kp2d_seq.unsqueeze(0),
            "bbx_xys": bbx_xys_seq.unsqueeze(0),
            "K_fullimg": K_fullimg_seq.unsqueeze(0),
            "f_imgseq": (
                self._f_imgseq_zeros[:, :F_len]
                if self.no_imgfeat else f_imgseq_seq.unsqueeze(0)
            ),
            "f_cam_angvel": cam_angvel.unsqueeze(0),
        }

        t0 = time.perf_counter()
        with self.profiler.record(
            "denoiser",
            frame_id=frame_id,
            window_start=window_start,
            window_end=window_end,
            note=self.denoiser_backend,
        ):
            pred_x, pred_cam = run_denoiser(self.denoiser_runner, self.denoiser_backend, batch)
        timings["denoiser"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        with self.profiler.record(
            "decode",
            frame_id=frame_id,
            window_start=window_start,
            window_end=window_end,
        ):
            decode_dict = self.endecoder.decode(pred_x)
        timings["decode"] = time.perf_counter() - t0

        # --- Streaming rollout ---
        t0 = time.perf_counter()
        with self.profiler.record(
            "rollout",
            frame_id=frame_id,
            window_start=window_start,
            window_end=window_end,
        ):
            gv_curr = decode_dict["global_orient_gv"][0, -1].detach()
            gc_curr = decode_dict["global_orient"][0, -1].detach()
            lv_prev = decode_dict["local_transl_vel"][0, -2].detach() if F_len > 1 else None
            lv_curr = decode_dict["local_transl_vel"][0, -1].detach()
            cam_prev = cam_angvel[-2] if F_len > 1 else None

            if self.rollout_state is None:
                self.rollout_state = init_rollout_w_Rt_state(
                    gv_curr, gc_curr, device=gv_curr.device,
                )

            body_params_global, self.rollout_state = rollout_step_w_Rt(
                self.rollout_state,
                global_orient_gv_curr=gv_curr,
                global_orient_c_curr=gc_curr,
                cam_angvel_prev=cam_prev,
                local_transl_vel_prev=lv_prev,
                local_transl_vel_curr=(None if lv_prev is not None else lv_curr),
            )
        timings["rollout"] = time.perf_counter() - t0

        # --- In-camera body params ---
        pred_body_params_incam = {
            "body_pose": decode_dict["body_pose"][0, -1:],
            "global_orient": decode_dict["global_orient"][0, -1:],
            "transl": compute_transl_full_cam(
                pred_cam[:, -1:],
                bbx_xys_seq[-1:].unsqueeze(0).to(_DEVICE),
                K_fullimg_seq[-1:].unsqueeze(0).to(_DEVICE),
            )[0],
        }
        if "betas" in decode_dict:
            pred_body_params_incam["betas"] = decode_dict["betas"][0, -1:]

        # --- Bbox update from last-frame keypoints ---
        with self.profiler.record(
            "bbox_feedback",
            frame_id=frame_id,
            window_start=window_start,
            window_end=window_end,
        ):
            updated_bbox = None
            visible = kp2d_last[:, 2] > 0.5
            if visible.any():
                vis_kp = kp2d_last[visible, :2]
                xmin, ymin = vis_kp[:, 0].min().item(), vis_kp[:, 1].min().item()
                xmax, ymax = vis_kp[:, 0].max().item(), vis_kp[:, 1].max().item()
                cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
                w, h = (xmax - xmin) * 1.1, (ymax - ymin) * 1.1
                updated_bbox = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

        return {
            "ready": True,
            "body_params_incam": {k: v.detach().cpu() for k, v in pred_body_params_incam.items()},
            "body_params_global": {k: v.detach().cpu() for k, v in body_params_global.items()},
            "timing": timings,
            "_updated_bbox": updated_bbox,
            "_frame_bgr": frame_bgr,
        }

    # ------------------------------------------------------------------
    #  Frontend: detect → ViTPose → HMR2 → window append → dispatch
    # ------------------------------------------------------------------

    @torch.no_grad()
    def process_frame(self, frame_bgr):
        """Process a single BGR frame. Returns dict with results and timing."""
        frame_id = self.frame_index
        timings = {}
        t_total = time.perf_counter()

        # --- 1. Collect any completed async backend result ---
        if self._async and self._denoiser_future is not None:
            if self._denoiser_future.done():
                self._pending_result = self._denoiser_future.result()
                self._denoiser_future = None
                if self._pending_result.get("_updated_bbox") is not None:
                    self.bbox_xyxy = self._pending_result["_updated_bbox"]

        # --- 2. Detection + tracking ---
        t0 = time.perf_counter()
        if self.frame_index % self.yolo_period == 0 or self.bbox_xyxy is None:
            with self.profiler.record("yolox", frame_id=frame_id):
                boxes, scores = self.yolox.detect(frame_bgr)
        else:
            boxes, scores = np.empty((0, 4)), np.empty((0,))

        with self.profiler.record("bytetrack", frame_id=frame_id):
            tracked = self.tracker.update(boxes, scores)

        if tracked:
            best = max(
                tracked,
                key=lambda r: r[2] * (r[0][2] - r[0][0]) * (r[0][3] - r[0][1]),
            )
            self.bbox_xyxy = best[0]
            self.primary_track_id = best[1]

        if self.bbox_xyxy is None:
            self.frame_index += 1
            return None
        timings["detect"] = time.perf_counter() - t0

        # --- 3. xyxy → xys ---
        with self.profiler.record("bbox_convert", frame_id=frame_id):
            bbx_xyxy_t = torch.from_numpy(np.asarray(self.bbox_xyxy)).float().unsqueeze(0)
            bbx_xyxy_t[:, [0, 2]] = bbx_xyxy_t[:, [0, 2]].clamp(0, self.width - 1)
            bbx_xyxy_t[:, [1, 3]] = bbx_xyxy_t[:, [1, 3]].clamp(0, self.height - 1)
            bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy_t, base_enlarge=1.2)[0]

        # --- 4. Preproc (ViTPose + HMR2 paired) ---
        t0 = time.perf_counter()
        run_vp_this_frame = (
            self.vitpose_period <= 1
            or self._last_kp2d is None
            or self.frame_index % self.vitpose_period == 0
        )
        if self._preproc_future is not None and self._preproc_future.done():
            (self._last_kp2d, self._last_f_img,
             self._last_bbx_xys, self._last_frame_bgr) = self._preproc_future.result()
            self._preproc_future = None

        if run_vp_this_frame:
            if self._preproc_executor is not None and self._last_kp2d is not None:
                # Async: submit (frame, bbx) for the next logical frame and
                # reuse the cached bundle for this iteration. All four cached
                # values come from the same logical frame.
                if self._preproc_future is None:
                    self._preproc_future = self._preproc_executor.submit(
                        _run_preproc_paired,
                        self.vitpose_runner, self.vitpose_backend,
                        self.hmr2_runner, self.hmr2_backend,
                        frame_bgr.copy(), bbx_xys.clone(), self.no_imgfeat,
                        self.profiler, frame_id,
                    )
                kp2d = self._last_kp2d
                f_img = self._last_f_img
            else:
                # Sync (or first-frame seed): pair current inputs with outputs.
                kp2d, f_img, _, _ = _run_preproc_paired(
                    self.vitpose_runner, self.vitpose_backend,
                    self.hmr2_runner, self.hmr2_backend,
                    frame_bgr, bbx_xys, self.no_imgfeat,
                    self.profiler, frame_id,
                )
                self._last_kp2d = kp2d
                self._last_f_img = f_img
                self._last_bbx_xys = bbx_xys.clone()
                self._last_frame_bgr = frame_bgr.copy()
        else:
            kp2d = self._last_kp2d
            f_img = self._last_f_img
        timings["preproc"] = time.perf_counter() - t0

        # In async mode, replace the "current frame" bbx_xys/frame_bgr with the
        # bundle's so window entries (kp2d, f_img, bbx) are all from the same frame.
        if self._async and self._last_bbx_xys is not None:
            bbx_xys = self._last_bbx_xys
            frame_bgr_for_render = self._last_frame_bgr
        else:
            frame_bgr_for_render = frame_bgr

        # --- 6. Append windows ---
        with self.profiler.record("window", frame_id=frame_id, note="append"):
            self.bbx_xys_window.append(bbx_xys.cpu())
            self.kp2d_window.append(kp2d.cpu())
            self.f_imgseq_window.append(f_img.cpu())
        self.frame_index += 1

        # --- 7. Warmup until window full ---
        if len(self.kp2d_window) < self.context_frames:
            timings["total"] = time.perf_counter() - t_total
            return {
                "ready": False,
                "warmup": f"{len(self.kp2d_window)}/{self.context_frames}",
                "timing": timings,
            }

        # --- 8. Snapshot window ---
        window_end = frame_id
        window_start = frame_id - len(self.kp2d_window) + 1
        with self.profiler.record(
            "window",
            frame_id=frame_id,
            window_start=window_start,
            window_end=window_end,
            note="snapshot",
        ):
            bbx_xys_seq = torch.stack(list(self.bbx_xys_window), dim=0)
            kp2d_seq = torch.stack(list(self.kp2d_window), dim=0)
            f_imgseq_seq = torch.stack(list(self.f_imgseq_window), dim=0)

        # --- 9. Dispatch backend ---
        if self._async:
            if self._denoiser_future is None:
                # Bind the bundled frame_bgr (paired with window's last kp2d
                # and bbx_xys) so all three streams stay aligned in render.
                self._denoiser_future = self._denoiser_executor.submit(
                    self._run_backend, bbx_xys_seq, kp2d_seq, f_imgseq_seq, kp2d,
                    frame_bgr_for_render.copy() if frame_bgr_for_render is not frame_bgr
                    else frame_bgr.copy(),
                    frame_id, window_start, window_end,
                )
            if self._pending_result is not None:
                result = dict(self._pending_result)
                result["timing"] = {**self._pending_result["timing"], **timings}
                result["timing"]["total"] = time.perf_counter() - t_total
                return result
            timings["total"] = time.perf_counter() - t_total
            return {
                "ready": False,
                "warmup": "async-init",
                "timing": timings,
            }

        # --- Synchronous path ---
        result = self._run_backend(
            bbx_xys_seq,
            kp2d_seq,
            f_imgseq_seq,
            kp2d,
            frame_bgr,
            frame_id,
            window_start,
            window_end,
        )
        if result.get("_updated_bbox") is not None:
            self.bbox_xyxy = result["_updated_bbox"]
        result["timing"].update(timings)
        result["timing"]["total"] = time.perf_counter() - t_total
        return result

    def run(self):
        """Main processing loop."""
        Log.info(
            f"[Run] {self.source_name} | window={self.context_frames} | "
            f"yolo_period={self.yolo_period} | vitpose_period={self.vitpose_period} | "
            f"denoiser={self.denoiser_backend} | no_imgfeat={self.no_imgfeat} | "
            f"async={self._async}"
        )

        fps_history = deque(maxlen=60)
        n_frames = 0

        try:
            while True:
                frame_id = self.frame_index
                frame_t0 = self.profiler._now()
                with self.profiler.record("capture", frame_id=frame_id):
                    ok, frame_bgr = self.cap.read()
                if not ok:
                    break

                result = self.process_frame(frame_bgr)
                self.profiler.mark_frame(frame_id, frame_t0, self.profiler._now())
                n_frames += 1

                if result is None:
                    if self._display_queue is not None:
                        try:
                            disp_frame = self._display_queue.get_nowait()
                            cv2.imshow("GEM-SMPL Webcam", disp_frame)
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                break
                        except _queue_mod.Empty:
                            cv2.waitKey(1)
                    print(f"\rFrame {self.frame_index}: no person detected", end="")
                    continue

                if self._render_queue is not None and result["ready"]:
                    # Pair mesh with the frame it was predicted from (defaults
                    # to current frame for the sync path); in async mode this
                    # avoids a 5-10 frame visual lag between mesh and image.
                    paired = result.get("_frame_bgr")
                    if paired is None:
                        paired = frame_bgr
                    try:
                        with self.profiler.record("render_enqueue", frame_id=self.frame_index - 1):
                            self._render_queue.put_nowait({
                                "frame_bgr": paired,
                                "body_params_incam": result["body_params_incam"],
                                "body_params_global": result["body_params_global"],
                                "K_fullimg": self._K_fullimg_cpu,
                            })
                    except _queue_mod.Full:
                        pass

                if self._display_queue is not None:
                    try:
                        disp_frame = self._display_queue.get_nowait()
                        cv2.imshow("GEM-SMPL Webcam", disp_frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    except _queue_mod.Empty:
                        pass

                t = result["timing"]

                if not result["ready"]:
                    print(
                        f"\rWarmup {result['warmup']} | "
                        f"det={t.get('detect', 0)*1000:.0f}ms "
                        f"pp={t.get('preproc', 0)*1000:.0f}ms "
                        f"tot={t['total']*1000:.0f}ms",
                        end="",
                    )
                    continue

                fps = 1.0 / max(t["total"], 1e-6)
                fps_history.append(fps)
                avg_fps = sum(fps_history) / len(fps_history)

                transl = result["body_params_global"]["transl"]
                tx, ty, tz = (
                    transl[0, 0].item(),
                    transl[0, 1].item(),
                    transl[0, 2].item(),
                )

                print(
                    f"\rFrame {self.frame_index:5d} | "
                    f"FPS {fps:5.1f} (avg {avg_fps:5.1f}) | "
                    f"det={t.get('detect', 0)*1000:4.0f} "
                    f"pp={t.get('preproc', 0)*1000:4.0f} "
                    f"den={t.get('denoiser', 0)*1000:4.0f} "
                    f"dec={t.get('decode', 0)*1000:3.0f} "
                    f"rol={t.get('rollout', 0)*1000:3.0f} "
                    f"tot={t['total']*1000:4.0f}ms | "
                    f"transl=({tx:+.2f}, {ty:+.2f}, {tz:+.2f})",
                    end="",
                )

        except KeyboardInterrupt:
            print("\n[Interrupted]")
        finally:
            self.cap.release()
            if self._denoiser_executor is not None:
                self._denoiser_executor.shutdown(wait=True, cancel_futures=True)
            if self._preproc_executor is not None:
                self._preproc_executor.shutdown(wait=True, cancel_futures=True)
            if self._render_queue is not None:
                self._render_queue.put(None)
            if self._render_proc is not None:
                self._render_proc.join(timeout=3)
                if self._render_proc.is_alive():
                    self._render_proc.terminate()
            if self._display_queue is not None:
                cv2.destroyAllWindows()
            print()

            if fps_history:
                avg = sum(fps_history) / len(fps_history)
                Log.info(f"[Done] Processed {n_frames} frames | Average FPS: {avg:.1f}")
            else:
                Log.info(f"[Done] Processed {n_frames} frames (no inference completed)")

            profile_export = self.profiler.export()
            if profile_export is not None:
                csv_path, svg_path, summary_path, summary = profile_export
                Log.info(f"[Profile] CSV: {csv_path}")
                Log.info(f"[Profile] SVG: {svg_path}")
                Log.info(f"[Profile] Summary: {summary_path}")
                if summary:
                    top = summary[0]
                    Log.info(
                        "[Profile] Top cumulative bottleneck: "
                        f"{top['stage']} total={top['total_ms']:.1f}ms "
                        f"mean={top['mean_ms']:.1f}ms p95={top['p95_ms']:.1f}ms"
                    )


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="GEM-SMPL Webcam Demo (real-time, ONNX)")
    parser.add_argument("--camera_id", type=int, default=0, help="Webcam device ID")
    parser.add_argument("--video", type=str, default=None, help="Video file (overrides camera)")
    parser.add_argument(
        "--context_frames", type=int, default=120,
        help="Sliding window length (model max=120, shorter=faster)",
    )
    parser.add_argument(
        "--yolo_period", type=int, default=5,
        help="Run YOLOX every N frames (ByteTrack interpolates between)",
    )
    parser.add_argument(
        "--vitpose_period", type=int, default=1,
        help="Run ViTPose every N frames (reuse keypoints between)",
    )
    parser.add_argument(
        "--no_imgfeat", action="store_true",
        help="Skip HMR2 features; use the no-imgfeat ONNX denoiser variant",
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Enable background rendering in a separate process",
    )
    parser.add_argument(
        "--render_mode", type=str, default="viser", choices=["viser", "opencv"],
        help="Render backend: 'viser' (web 3D world) or 'opencv' (in-camera overlay)",
    )
    parser.add_argument(
        "--render_port", type=int, default=8012,
        help="Port for Viser web server (only used with --render_mode viser)",
    )
    parser.add_argument(
        "--async_pipeline", action="store_true", default=True,
        help="Overlap ViTPose, HMR2, and denoiser across frames (default: True)",
    )
    parser.add_argument(
        "--no_async_pipeline", action="store_true",
        help="Disable async pipeline (force synchronous mode)",
    )
    parser.add_argument(
        "--profile_pipeline", action="store_true",
        help="Record stage/model runtime events and export CSV + SVG timeline on exit",
    )
    parser.add_argument(
        "--profile_output", type=str, default=None,
        help="Profile output file prefix. Default: outputs/profile/<source>_pipeline_profile",
    )
    parser.add_argument(
        "--profile_plot_frames", type=int, default=300,
        help="Number of recent processed frames to draw in the SVG timeline; <=0 draws all",
    )
    parser.add_argument(
        "--profile_sync_cuda", action="store_true",
        help="Synchronize CUDA before/after each profiled event for more accurate GPU latency",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    demo = WebcamGEMSMPLDemo(args)
    demo.run()
