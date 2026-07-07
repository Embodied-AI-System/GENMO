# demo_webcam.py 流水线说明

本文档说明 `scripts/demo/demo_webcam.py` 在运行时构建的实时推理流水线。

## 总览

```text
视频文件 / 摄像头
  -> OpenCV 取帧
  -> YOLOX 检测 + ByteTrack 跟踪
  -> bbox 格式转换: xyxy -> [cx, cy, size]
  -> ViTPose 2D 关键点
  -> 可选 HMR2 图像特征
  -> 滑动上下文窗口
  -> GEM-SMPL denoiser
  -> EnDecoder
  -> streaming global rollout
  -> 可选渲染
```

代码整体可以理解为三部分：

- 前端：`process_frame()` 中的逐帧图像处理。
- 后端：`_run_backend()` 中的滑窗动作推理。
- 渲染：可选的子进程，消费已经完成推理的人体参数。

## 启动阶段

`WebcamGEMSMPLDemo.__init__()` 做一次性初始化：

1. 用 `cv2.VideoCapture` 打开 `--video` 或 `--camera_id`。
2. 读取图像宽度、高度和输入源 FPS。
3. 用 `estimate_K(width, height)` 估计整图相机内参。
4. 为上下文窗口创建静态相机角速度序列。
5. 创建 bbox、2D 关键点、图像特征的滑动窗口。
6. 通过 `_load_models()` 并行加载模型。
7. 通过 `_warmup()` 做一次 warmup 推理。
8. 如果打开 `--render`，启动渲染子进程。

加载的模型如下：

| 组件 | 加载函数 | 作用 |
| --- | --- | --- |
| YOLOX + ByteTrack | `load_yolox()` | 人体检测和 track 连续性维护 |
| ViTPose | `load_vitpose()` | COCO-17 2D 关键点 |
| HMR2 | `load_hmr2()` | 可选的 1024 维图像特征 |
| GEM-SMPL denoiser | `load_denoiser()` | 预测 SMPL 动作表示和 weak-perspective camera |
| EnDecoder | `gem.network.endecoder.EnDecoder` | 将 denoiser 输出解码成人体参数和运动字段 |

ONNX/TRT/PyTorch 的 fallback 逻辑在 `scripts/demo/onnx_runners.py`。

## 主循环

`run()` 是最外层循环：

```text
cap.read()
  -> process_frame(frame_bgr)
  -> 可选：把结果送入 render queue
  -> 打印 timing / FPS
```

每个成功读取到的输入帧都会传给 `process_frame()`。

## Stage 触发条件

并不是每一轮主循环都会完整执行所有模型。当前脚本默认打开 async pipeline，
因此很多主循环迭代只是复用缓存结果或检查后台 future 是否完成。

| Stage | 触发条件 | 输入 | 输出 | 含义 |
| --- | --- | --- | --- | --- |
| Stage 1: 取帧 | `run()` 每轮循环都会调用 `cap.read()` | `--video` 视频文件或 `--camera_id` 摄像头 | `frame_bgr` | 得到一帧 OpenCV BGR 图像；视频文件读取可能远快于原始 FPS |
| Stage 2: 检测和跟踪 | 每个有输入帧的 `process_frame()` 都会调用 ByteTrack；YOLOX 只在 `frame_index % yolo_period == 0` 或当前没有 bbox 时运行 | `frame_bgr`、上一轮 track 状态、可选 YOLOX `boxes/scores` | `self.bbox_xyxy`、`self.primary_track_id` | 找到当前主人体目标；默认 `--yolo_period 5`，未触发 YOLOX 时给 ByteTrack 传空 detections |
| Stage 3: bbox 格式转换 | 当前帧存在 `self.bbox_xyxy` 时 | `self.bbox_xyxy = [x1, y1, x2, y2]`、图像宽高 | `bbx_xys = [cx, cy, size]` | 将检测框转成后续模型共用的中心点和尺度表示 |
| Stage 4: ViTPose 和 HMR2 预处理 | 满足 `vitpose_period <= 1`、无缓存关键点，或 `frame_index % vitpose_period == 0`；async 下后台最多同时跑一个 preproc future | `frame_bgr`、`bbx_xys`、可选 HMR2 runner | `kp2d: (17, 3)`、`f_img: (1024,)`、配对的 `bbx_xys/frame_bgr` | 生成 2D 关键点和可选图像特征；`--no_imgfeat` 时 HMR2 跳过并输出零向量 |
| Stage 5: 滑动上下文窗口 | 有 bbox 和可用 `kp2d/f_img` 时 | 当前帧的 `bbx_xys`、`kp2d`、`f_img` | `bbx_xys_window`、`kp2d_window`、`f_imgseq_window` | 缓存最近 `context_frames` 个有效帧；没填满时只返回 warmup |
| Stage 6: 后端 denoiser | 窗口长度达到 `context_frames` 后触发；async 下只有 `_denoiser_future is None` 时才提交新任务 | `obs`、`bbx_xys`、`K_fullimg`、`f_imgseq`、`f_cam_angvel` | `pred_x`、`pred_cam` | 对最近窗口做时序动作推理，输出 GEM-SMPL 动作表示和 weak-perspective camera |
| Stage 7: 解码 | denoiser 完成后立即执行，属于 `_run_backend()` | `pred_x` | `decode_dict`，包含 `body_pose/global_orient/global_orient_gv/local_transl_vel/betas` 等 | 将 denoiser 的紧凑动作表示解码为 SMPL 和 rollout 需要的字段 |
| Stage 8: Streaming Global Rollout | Stage 7 完成后立即执行，属于 `_run_backend()` | `decode_dict`、`pred_cam`、最近一帧 `bbx_xys`、`K_fullimg`、`self.rollout_state` | `body_params_global`、`body_params_incam`、更新后的 `self.rollout_state` | 对最新一帧做增量全局轨迹更新，同时计算相机坐标系下的人体参数 |
| Stage 9: bbox 反馈 | `_run_backend()` 中最后一帧存在可见 2D 关键点时 | `kp2d_last` | `_updated_bbox`，并可能更新 `self.bbox_xyxy` | 用关键点反推下一轮 bbox，减少只依赖周期性 YOLOX 的漂移 |
| Stage 10: 可选渲染 | `--render` 打开且 `result["ready"] == True`；render worker 在子进程中循环消费 queue | `frame_bgr`、`body_params_incam`、`body_params_global`、`K_fullimg` | Viser 3D mesh 或 OpenCV mesh overlay | 可视化当前人体结果；queue 满时丢弃本次渲染数据，不阻塞主推理循环 |

async 模式下还有两个额外的调度动作：`_preproc_future.done()` 为 True 时回收
Stage 4 的后台结果；`_denoiser_future.done()` 为 True 时回收 Stage 6-9 的后台
结果。这两个动作不产生新的 pipeline stage，只是把后台计算完成的数据接回主循环。

## Stage 1: 取帧

输入帧是 OpenCV 读出的 BGR 图像：

```python
ok, frame_bgr = self.cap.read()
```

脚本支持两种输入：

- `--video <path>`：视频文件。
- `--camera_id <id>`：摄像头。

## Stage 2: 检测和跟踪

`process_frame()` 每隔 `--yolo_period` 帧运行一次 YOLOX；如果当前还没有
bbox，也会强制运行 YOLOX：

```text
YOLOX.detect(frame_bgr) -> boxes, scores
ByteTrack.update(boxes, scores) -> tracked persons
```

如果检测到多个人，代码会选择：

```text
score * bbox_area
```

最大的 track，作为当前主目标。它的 bbox 会保存到 `self.bbox_xyxy`。

ByteTrack 每轮 `process_frame()` 都会调用。YOLOX 没有触发时，会传入空
detections，让 tracker 维护已有 track 状态。

如果当前帧没有可用的人体 track，这一帧会被跳过，不再进入后续模型。

## Stage 3: bbox 格式转换

选中的 bbox 会先被 clamp 到图像范围内，然后从：

```text
[x1, y1, x2, y2]
```

转换为：

```text
[cx, cy, size]
```

对应代码是：

```text
get_bbx_xys_from_xyxy(..., base_enlarge=1.2)
```

这个 `bbx_xys` 会被 ViTPose、HMR2 和 denoiser 共用。

## Stage 4: ViTPose 和 HMR2 预处理

`_run_preproc_paired()` 运行：

```text
ViTPose(frame_bgr, bbx_xys) -> kp2d: (17, 3)
HMR2(frame_rgb, bbx_xys)    -> f_img: (1024,)
```

它返回：

```text
kp2d, f_img, bbx_xys, frame_bgr
```

这里把 keypoints、image feature、bbox 和 render frame 打成同一个 bundle，
目的是保证它们都来自同一个逻辑帧。异步 pipeline 下尤其重要，否则容易出现
mesh 和图像帧错位。

如果使用 `--no_imgfeat`，HMR2 会被跳过，`f_img` 会变成全零向量。

`--vitpose_period N` 可以让 ViTPose 每 N 帧运行一次，中间帧复用上一组
keypoints 和 image features。默认 `--vitpose_period 1` 表示策略上每帧需要
一组 keypoints；但在 async 模式下，后台 preproc future 没完成时主循环会复用
上一组结果，而不是阻塞等待 ViTPose/HMR2。

## Stage 5: 滑动上下文窗口

模型输入是一个时间窗口。`process_frame()` 每处理一帧，就往下面三个窗口里
append 一个元素：

```text
bbx_xys_window    -> (F, 3)
kp2d_window       -> (F, 17, 3)
f_imgseq_window   -> (F, 1024)
```

`F` 由 `--context_frames` 控制，默认是 `120`。

窗口没填满之前，`process_frame()` 返回：

```text
ready: False
warmup: <current>/<context_frames>
```

也就是说，前 `context_frames` 帧主要是在积累上下文。

## Stage 6: 后端 denoiser

窗口填满后，`_run_backend()` 会构造 denoiser 的 batch。同步模式下当前帧会
等待 `_run_backend()` 完成；async 模式下只有没有正在运行的 `_denoiser_future`
时才会提交新的后台 backend 任务。

| Batch key | Shape | 含义 |
| --- | --- | --- |
| `obs` | `(1, F, 17, 3)` | ViTPose COCO-17 关键点 |
| `bbx_xys` | `(1, F, 3)` | bbox 中心和尺度 |
| `K_fullimg` | `(1, F, 3, 3)` | 整图相机内参 |
| `f_imgseq` | `(1, F, 1024)` | HMR2 图像特征，或全零 |
| `f_cam_angvel` | `(1, F, 6)` | 相机角速度特征 |

denoiser 输出：

```text
pred_x
pred_cam
```

`pred_x` 是 GEM-SMPL 的动作表示。`pred_cam` 是窗口内每帧的
weak-perspective camera 参数。

## Stage 7: 解码

`EnDecoder.decode(pred_x)` 将 denoiser 的动作表示解码为 SMPL 和 rollout
需要的字段，例如：

```text
body_pose
global_orient
global_orient_gv
local_transl_vel
betas, if present
```

实时路径只取解码后窗口的最后一帧作为当前输出。

## Stage 8: Streaming Global Rollout

denoiser 预测的是局部运动。`_run_backend()` 会把最新一帧的解码运动转换成
持续更新的全局轨迹：

```text
init_rollout_w_Rt_state()
rollout_step_w_Rt()
```

`self.rollout_state` 会跨帧保存。这样每一帧只需要做增量 rollout，不需要每次
重新处理完整序列。

输出的全局人体参数是：

```text
body_params_global = {
  global_orient,
  transl,
}
```

同时也会计算相机坐标系下的人体参数：

```text
body_params_incam = {
  body_pose,
  global_orient,
  transl,
  betas, optional
}
```

其中相机坐标系下的 `transl` 来自：

```text
compute_transl_full_cam(pred_cam, bbx_xys, K_fullimg)
```

## Stage 9: bbox 反馈

后端会根据最后一帧可见的 2D 关键点估计一个新的 bbox：

```text
kp2d_last -> visible joints -> updated_bbox
```

这个 bbox 可以更新 `self.bbox_xyxy`，用于后续帧。YOLOX 仍然会按照
`--yolo_period` 周期性运行，用来纠正跟踪漂移。

## Stage 10: 可选渲染

如果打开 `--render`，`run()` 会把 ready 的结果放入 multiprocessing queue：

```text
frame_bgr
body_params_incam
body_params_global
K_fullimg
```

渲染模式有两种：

| 模式 | Worker | 输出 |
| --- | --- | --- |
| `viser` | `_render_worker_viser()` | 浏览器里的全局 3D mesh 视图 |
| `opencv` | `_render_worker_cv2()` | 叠加在相机画面上的 mesh overlay |

渲染进程和主推理循环是分开的。这样渲染慢时不会直接阻塞模型推理。

## 异步 Pipeline

默认打开异步模式：

```text
--async_pipeline default True
--no_async_pipeline disables it
```

异步模式下：

- ViTPose + HMR2 运行在 `_preproc_executor`。
- Denoiser + decode + rollout 运行在 `_denoiser_executor`。
- 当前循环可以复用已经完成的上一份结果，同时下一次模型调用还在后台运行。
- bbox、keypoints、image features 和 render frame 会作为配对 bundle 传递，
  避免把不同帧的数据混在一起。

异步模式能提高吞吐，但会引入少量帧延迟。同步模式更适合调试，因为每一帧都会
等待所有 stage 执行完。

## Timing 输出

终端里的 timing 字段对应下面这些 stage：

| 字段 | Stage |
| --- | --- |
| `det` | YOLOX + ByteTrack |
| `pp` | ViTPose + 可选 HMR2 预处理 |
| `den` | GEM-SMPL denoiser |
| `dec` | EnDecoder decode |
| `rol` | streaming global rollout |
| `tot` | `process_frame()` 总耗时 |

## 后续接入点

如果后面要接机器人动作模仿、控制器，或者其他实时消费者，最直接的输出是
`process_frame()` 返回的 ready result：

```text
result["body_params_incam"]
result["body_params_global"]
```

比较合适的插入点：

- `_run_backend()` 之后：直接使用解码后的 SMPL / global pose。
- `run()` 中送入 render queue 之前：只消费 `result["ready"] == True` 的结果。
- 复用 render queue 的进程边界：如果下游消费者不能阻塞主推理循环，可以用类似
  render worker 的独立进程方式接入。

不建议直接读取中间异步 cache，除非同时保持 `_run_preproc_paired()` 使用的
同帧配对关系。

## 运行时 Profiling

`demo_webcam.py` 支持在运行过程中记录各 stage/model 的实际执行时间，并在退出时
导出 CSV、summary 和 SVG 时空图：

```bash
python scripts/demo/demo_webcam.py \
  --video inputs/demo/walk.mp4 \
  --profile_pipeline \
  --profile_sync_cuda \
  --profile_plot_frames 300
```

默认输出：

```text
outputs/profile/<source>_pipeline_profile.csv
outputs/profile/<source>_pipeline_profile.summary.txt
outputs/profile/<source>_pipeline_profile.svg
```

也可以指定输出前缀：

```bash
python scripts/demo/demo_webcam.py \
  --camera_id 0 \
  --profile_pipeline \
  --profile_output outputs/profile/cam0_run1
```

SVG 图的含义：

- 纵轴是 stage/model，例如 `YOLOX`、`ViTPose`、`HMR2`、`denoiser`、
  `decode`、`rollout`。
- 横轴是均匀 wall time，单位是秒。
- 时间轴下方会用短 tick 标出每个 processed frame 的处理开始时间点，并标注部分
  frame id。
- 每个色块表示该 stage/model 的一次实际执行。
- 色块宽度表示真实 wall-time latency。
- 色块上的文字是该次执行的耗时，单位 ms。

CSV 中每行是一个事件：

```text
stage,frame_id,window_start,window_end,start_s,end_s,duration_ms,thread,note
```

其中：

- `frame_id` 是触发该事件的主循环帧。
- `window_start/window_end` 只对 denoiser/decode/rollout 等窗口后端阶段有意义。
- `start_s/end_s` 是相对 demo 开始后的 wall time。
- `thread` 可用来区分主线程和 async executor 线程。

`summary.txt` 按 stage 汇总：

```text
stage,count,total_ms,mean_ms,p50_ms,p95_ms,max_ms
```

用于快速判断瓶颈。`total_ms` 最大的 stage 是累计耗时瓶颈；`p95_ms` 高的 stage
说明存在明显尾延迟。

建议用 `--profile_sync_cuda` 测 GPU 推理延迟。否则 PyTorch/CUDA 后端可能只测到
CPU launch 时间，不能准确反映 GPU kernel 实际完成时间。打开同步会改变运行时序，
但更适合定位模型级瓶颈。

在 async 模式下，图里会看到：

- 主线程很多帧没有 `denoiser` 色块，因为后台已有 `_denoiser_future` 在跑。
- `ViTPose/HMR2` 色块可能和主循环后续帧重叠，因为它们在 `_preproc_executor`
  里执行。
- `denoiser/decode/rollout` 色块可能跨越多个 processed frames，因为它们在
  `_denoiser_executor` 里执行。

如果要看同步逐帧延迟，可以加：

```bash
--no_async_pipeline
```

这时窗口填满后，每个有效帧都会等待 denoiser/decode/rollout 完成，SVG 更容易
对应单帧端到端耗时。
