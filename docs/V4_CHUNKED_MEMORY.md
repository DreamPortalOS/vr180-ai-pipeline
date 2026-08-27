# V-4 — 长片分块内存管理（非流式路径 OOM 治理）

Issue #37. Companion doc to `pipeline/chunked_processor.py`.

## 背景

流式路径 `pipeline/streaming_pipeline.py` 已是 O(1) 内存（逐帧读→处理→写 ffmpeg 管道，
中间张量用完即释放），是长片的**默认推荐路径**（`--quality standard|high` 自动启用）。
但非流式（批处理）路径 `scripts/run_pipeline.py` 仍有多处把**整段帧序列**驻留内存：

| 路径 | 代码位置 | 驻留内存 |
|------|----------|----------|
| 读全部帧 | `main()` 的 `frames = list(read_frames(...))` | 全部 RGB 帧 |
| depth 段 | `run_depth_stage` 的 `depths = []` | 全部深度图 |
| stereo 段 | `run_stereo_stage` 的 `left_frames/right_frames` | 全部 L/R 帧 |
| equirect（批处理） | `map_sequence` → 读回全部 SBS 帧 | 全部 SBS 帧 |
| outpaint | `Outpainter.process` 的 `result` | 全部帧 |
| metadata 编码 | `embed_single_frame_batch` 的 `raw = b"".join(...)` | 全部帧的原始字节 |

8K/眼的长片，单帧 ≈7680×3840×3 ≈ 84 MB，几百帧即数十 GB → OOM。

## 解决方案：分块原语 + 可选 `--chunk-size`

`pipeline/chunked_processor.py` 提供内存受界的分块原语：

- `chunk_ranges(n, chunk_size, overlap)` — 把 `[0, n)` 切成重叠窗口，峰值内存 ∝ `chunk_size`
  （+`overlap` warmup），**与片长无关**。
- `process_in_chunks(frames, process_fn, chunk_size, overlap)` — 按 chunk 驱动逐帧处理，
  warmup 帧仅重建时序状态、其输出被丢弃。

**正确性契约（V-4 验收核心）**：分块结果与整段处理**逐帧一致**。两种达成方式：

1. **无状态环节**（per-frame depth、equirect per-frame、outpaint、无 EMA 的 depth）：`overlap=0`
   即精确——每帧独立。
2. **有限窗口时序滤波**（输出依赖最近 W 帧）：`overlap ≥ W-1` 即精确——warmup 重建完整窗口。
3. **无限 IIR 滤波**（`StereoRenderer._prev_disparity`、depth EMA）：复用**单一持久实例**
   跨 chunk 处理 → 状态连续 → `overlap=0` 即精确。

### 已分块的环节（`--chunk-size` 开启时）

- **depth 段**（`run_depth_stage`，Depth-Anything 路径）：`estimate_sequence_chunked` +
  EMA `prev_depth` 跨 chunk 持久 → 逐帧一致。逐 chunk 写 `depth_{i}.npy/.png` 检查点。
- **stereo 段**（`run_stereo_stage`，默认渲染器路径）：`render_sequence_chunked`，单一
  `StereoRenderer` 跨 chunk 复用 → `_prev_disparity` 连续 → 逐帧一致。逐 chunk 写 L/R PNG。
- **`DepthEstimator.estimate_sequence_chunked`** / **`StereoRenderer.render_sequence_chunked`**：
  可复用的内存受界序列方法。

用法：`python scripts/run_pipeline.py -i in.mp4 -o out.mp4 --chunk-size 16 [--overlap 0]`

## 无法分块的环节（显式文档化 + 上限估算）

以下环节具有**全局语义**，分块会改变输出，故不分块；给出峰值内存上限估算（按 8K/眼 SBS
≈84 MB/帧计）：

| 环节 | 原因 | 峰值内存上限 |
|------|------|-------------|
| **DepthCrafter**（`depth_model=depthcrafter`） | 整段视频一次性时序一致深度推理（外部进程） | 由外部进程的 `max_resolution` 控制，**与帧长线性**；上限 ≈ `num_frames × max_res² × 4 B`。建议用 `--depthcrafter-max-res` 压短边到 ≤540，长片另分多段输入 |
| **StereoCrafter**（`stereo_model=stereocrafter`） | 整段视频外部推理 + 回读全部 L/R 帧 | 同上外部进程；回读段 `_load_video_frames` 驻留全部 L/R 帧 ≈ `2 × num_frames × W×H×3` |
| **批处理 equirect**（`map_sequence`/ffmpeg v360） | 单次 ffmpeg v360 per-eye 跑完整序列才有 ~10× 加速 | `_write_image_sequence` 写 PNG 到盘（不驻留），但 `_read_image_sequence` 回读全部 ≈ `num_frames × eye_W×eye_H×3`。**缓解**：`--no-equirect-batched` 走 per-frame OpenCV 路径（每帧独立、可分块、`overlap=0` 精确） |
| **metadata 编码**（`embed_single_frame_batch`） | 单个 ffmpeg 进程吃完整 raw 流 | `raw = b"".join(frame.tobytes())` ≈ `num_frames × W×H×3` 字节。**这是最后一步**，与流式路径共用同款 raw-pipe；如需受界请改用流式路径（已 O(1)） |

**与 V-1 流式路径的关系**：流式路径已覆盖 depth→stereo→project→encode 的 O(1) 融合跑批，
是长片默认。`--chunk-size` 面向**需要分阶段/检查点**的批处理场景，提供阶段内内存受界；
两套机制不冲突、各司其职。

## 测试

`tests/test_chunked_processor.py` + `tests/test_pipeline.py::TestRunPipelineChunked`：

- 无状态 & 有限窗口时序：分块 == 整段，逐帧一致（含 overlap 边界）；
- 极端：`chunk_size=1` 与 `chunk_size>总帧数` 各有测试；
- `overlap < 窗口-1` / `overlap=0`（时序态）会发散——证伪测试，证明等价测试非空真；
- `run_depth_stage`/`run_stereo_stage` 端到端 chunked == whole（mock 深度估算，CPU-only，
  无模型/无真实推理）。

CPU-only、无新依赖、不跑真实模型，满足 CI（ubuntu + CPU + 无模型）。
