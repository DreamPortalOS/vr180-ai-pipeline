# VR180 AI Pipeline — 完整会话总结与开发文档

> 本文档总结了从零到可用的全部开发过程，包括需求分析、试错经验、已实现功能和未来开发指南。
> 
> 生成时间: 2026-06-23（最后更新: 2026-06-23 Overnight Session）

---

## 第一部分：需求总览

### 用户原始需求

将一个 2D AI 生成视频（Sora/Runway/Kling/Vidu 等）转换为可在 VR 头显（Meta Quest 3）上播放的 VR180 3D 沉浸式视频。

### 核心技术路径

```
2D视频 → 深度估计 → 立体视差 → 等距柱状投影 → VR元数据注入 → VR180视频
```

### 分阶段需求

| 阶段 | 需求 | 状态 |
|------|------|------|
| **Phase 0: Bug修复** | 修复 pipeline 中的关键 bug | ✅ 完成 |
| **Phase 1: 后端优化** | 设备检测、流式处理、分块超分 | ✅ 完成 |
| **Phase 2: VR180 高级修正 & 时序 AI 外绘** | SBS 检测、方向矩阵、光流时序外绘 | ✅ 完成 |
| **Phase 3: Web 基础设施 & API** | FastAPI REST API、任务管理、CRUD 端点 | ✅ 完成 |
| **Phase 4: 公测** | 限额、保存结果、空间视频 | 📋 规划中 |

---

## 第二部分：试错经验详解

### Bug 1: 等距柱状投影画面上下颠倒

**现象**: 输出 VR 视频在 Quest 3 中查看时，天空在下面、地面在上面。

**排查过程**:
1. 检查 `equirectangular_mapper.py` 的 `_map_via_opencv` 方法
2. 发现 OpenCV 的 `cv2.remap` 和 ffmpeg `v360` 滤镜的坐标系不一致
3. OpenCV: y=0 是顶部；VR 头显: phi=0 是正上方（north pole）

**修复**: 在 OpenCV fallback 路径中添加 `cv2.flip(equirect, 0)`；在 ffmpeg v360 路径中添加 `,vflip` 滤镜链。

**关键代码**:
```python
# equirectangular_mapper.py
def __init__(self, ..., flip_vertical: bool = True):
    self.flip_vertical = flip_vertical

# OpenCV 路径
if self.flip_vertical:
    equirect = cv2.flip(equirect, 0)

# ffmpeg 路径
if self.flip_vertical:
    vfilter += ",vflip"
```

**经验**: VR 坐标系和 2D 图像坐标系不同，需要明确约定并做好翻转。`flip_vertical` 参数让用户可以灵活控制。

---

### Bug 2: 非 180° 源视频投影拉伸变形

**现象**: 70° FOV 的 Sora 生成视频被拉伸到整个 180° 球面，人物变得扁平。

**排查过程**:
1. 发现 `src_hfov` 参数存在但 `iv_fov`（垂直 FOV）没有正确计算
2. ffmpeg v360 的 `iv_fov` 被硬编码为与 `ih_fov` 相同
3. 对于 16:9 的视频（1920×1080），如果 hfov=70°，vfov 应该是 ~42°，不是 70°

**修复**: 添加 `_calc_vertical_fov` 方法，基于针孔相机模型自动计算：
```python
def _calc_vertical_fov(self, src_width, src_height):
    hfov_rad = math.radians(self.src_hfov)
    vfov_rad = 2.0 * math.atan(math.tan(hfov_rad / 2.0) * src_height / src_width)
    return math.degrees(vfov_rad)
```

**经验**: FOV 不是各向同性的！必须根据源视频的宽高比分别计算水平和垂直 FOV。使用 `fill_black=1`（或在 OpenCV 中用 `-1` 标记越界像素）避免拉伸。

---

### Bug 3: 视频时长严重截断（0.17秒 vs 28秒）

**现象**: 渲染输出的 VR 视频只有 0.17 秒（5帧），而源视频是 28 秒。

**排查过程**:
1. 最初用 glob pattern `equirect_*.png` 直接编码，但中间文件只保留了少量帧
2. 发现 `run_pipeline.py` 的批量处理模式会把所有帧加载到内存，对于长视频会出现截断
3. 24 帧测试时正常，但全量渲染时因为内存管理问题导致中间文件不完整

**修复**: 
- 确保 glob pattern 正确排序（`sorted(glob.glob(...))`）
- 改用 `StreamingPipeline` 避免全量内存加载
- 手动用 ffmpeg 直接编码 equirect 帧目录

**经验**: 
- 对于帧序列编码，**必须**确保 glob 排序正确
- 大视频**必须**用流式处理，不能全量加载到内存
- 先用少量帧（如 `--max-frames 24`）验证流程，再跑全量

---

### Bug 4: VR 元数据注入损坏视频文件

**现象**: 注入 sv3d 元数据后，视频文件无法播放（ISOBMFF stco offset 表损坏）。

**排查过程**:
1. 原始代码用 Python 手动解析和修改 MP4 ISOBMFF 二进制结构
2. 手动插入 sv3d box 后没有正确更新 stco（sample table chunk offset）表
3. 导致播放器在读取 sample 数据时 seek 到错误的 offset

**修复**: 完全放弃手动 ISOBMFF 操作，改用 Google spatial-media CLI 工具：
```python
cmd = [
    "python3", "-m", "spatialmedia",
    "-i", "-2",  # V2 spec (sv3d + st3d)
    "-s", "left-right",
    "-p", "equirectangular",
    input_path, output_path,
]
subprocess.run(cmd, check=True)
```

**经验**: 
- **永远不要手动修改 MP4 二进制结构**，除非你有完整的 ISOBMFF parser
- Google spatial-media 是唯一可靠的 VR 元数据注入方案
- ffmpeg 的 `-metadata:s:v` 方式只支持 V1 XML，不支持 sv3d

---

### Bug 5: upscaler.py 文件损坏

**现象**: `pipeline/upscaler.py` 包含内嵌的 XML 标记和 git diff 冲突标记，Python 无法解析。

**排查过程**: 
1. 文件中混入了 `<thinking_mode>interleaved</thinking_mode>` 等 XML 标记
2. 看起来是 AI 编辑过程中产生的 artifact

**修复**: 完全重写 `upscaler.py`，保持干净的 Python 代码。

**经验**: AI 生成代码时需要严格验证输出的完整性，特别是文件开头和结尾。

---

### Bug 6: 测试视频 7680×1920 被错误处理

**现象**: 某些测试输出是 7680×1920（已经包含 SBS），但 pipeline 把它当作单眼处理。

**排查**: SBS 视频的宽度是单眼的 2 倍。如果输出已经包含左右眼，就不需要再做立体渲染。需要在 pipeline 入口检查输入是否已经是 SBS 格式。

**状态**: ✅ 已修复。在 `scripts/run_pipeline.py` 中实现自动检测：当输入宽高比 ≈ 4:1（如 7680×1920）时自动识别为 SBS 格式，跳过 Stage 1（深度估计）和 Stage 2（立体渲染），直接进入 Stage 3（等距柱状投影）。

**修复代码**: `_detect_sbs_input()` 函数基于 aspect ratio 检测，另有 `--force-sbs` 标志手动覆盖。

**测试**: 7 个自动化测试覆盖（标准 16:9 不误判、4:1 正确检测、force flag、超宽检测、不存在文件处理、SBS 帧分离、pipeline 集成）。

---

## 第三部分：已实现的全部功能

### 3.1 Phase 0 — Bug 修复

| 文件 | 修复内容 |
|------|----------|
| `pipeline/equirectangular_mapper.py` | 翻转修复 + VFOV 自动计算 + black fill |
| `pipeline/spherical_injector.py` | 改用 spatial-media CLI 注入 sv3d/st3d |
| `pipeline/upscaler.py` | 完全重写，干净模块 |

### 3.2 Phase 1 — 后端优化

#### Device Detection (`pipeline/device_utils.py`)
- `detect_best_device()`: CUDA → MPS → CPU 自动检测
- `get_device_info()`: 返回详细设备信息字典
- `resolve_device()`: 用户指定或自动检测

#### Streaming Pipeline (`pipeline/streaming_pipeline.py`)
- `StreamingPipeline` 类: 逐帧读取 + 处理 + ffmpeg pipe 写入
- 内存使用 O(1) 而非 O(总帧数)
- 支持 `--streaming` CLI 标志
- 自动 BGR↔RGB 转换

#### Tiled Upscaling (`pipeline/upscaler.py`)
- `upscale_tiled()`: 将大帧分割为 512×512 瓦片独立超分
- padding 处理避免拼接缝
- OOM 自动 fallback 到 Lanczos
- 支持 `--tiled-upscale` 和 `--tile-size` CLI 标志

#### CLI 集成 (`scripts/run_pipeline.py`)
- `--streaming`: 启用流式处理模式
- `--tiled-upscale`: 启用分块超分
- `--tile-size`: 设置瓦片大小
- 自动设备检测集成

### 3.3 Phase 2 — VR180 高级修正 & 时序 AI 外绘

#### Smart SBS Input Detection (`scripts/run_pipeline.py`)
- `_detect_sbs_input()`: 自动检测输入是否已为 SBS 格式（宽高比 ≈ 4:1）
- 检测到 SBS 时跳过 Stage 1（深度估计）和 Stage 2（立体渲染）
- 直接进入 Stage 3（等距柱状投影）
- `--force-sbs` CLI 标志：手动覆盖自动检测

#### Orientation Matrix Harness (`pipeline/research/orientation_matrix.py`)
- `apply_flip()`: 应用 cv2.flip (0=垂直, 1=水平, -1=双向)
- `apply_transpose()`: 应用 cv2.transpose + 旋转（t1=90°CW, t2=90°CCW）
- `generate_orientation_matrix()`: 生成 flip×transpose 组合矩阵（3×3=9 种变体）
- `generate_ffmpeg_filter_map()`: 输出 ffmpeg transpose 滤镜映射报告
- `run_diagnostic()`: 一键生成诊断视频网格 + JSON 报告

#### AI Temporal Outpainter (`pipeline/research/ai_outpainter.py`)
- `generate_boundary_mask()`: 等距柱状投影边界遮罩生成（极点+边缘检测）
- `compute_optical_flow()`: Farneback 稠密光流计算
- `warp_frame_by_flow()`: 基于光流向量的帧扭曲
- `poisson_blend()`: 泊松无缝混合填充
- `outpaint_frame()`: 单帧外绘（光流+混合+质量评估）
- `outpaint_sequence()`: 多帧序列外绘（时序一致性+收敛检测）
- 质量指标：PSNR、SSIM 计算

#### ISOBMFF Box Building (`pipeline/spherical_injector.py`)
- 新增完整 ISOBMFF box 构建函数：
  - `build_st3d_box()`: 立体 3D box（SBS/Top-Bottom/Mono）
  - `build_sv3d_box()`: 球形视频 3D 容器
  - `build_svhd_box()`: 球形视频头部
  - `build_svv3d_box()`: 球形视频视口
  - `build_svproj_box()`: 球形投影
  - `build_svmi_box()`: 球形视频媒体信息
- 所有 box 遵循 ISOBMFF FullBox 格式（version + flags）

#### Upscaler Graceful Fallback (`pipeline/upscaler.py`)
- 当 realesrgan 未安装时自动降级为 OpenCV bicubic 插值
- `_use_opencv_fallback` 标志自动设置
- 所有 upscale 方法（单帧、分块、批量）均支持降级

### 3.4 Phase 3 — Web 基础设施 & API

#### FastAPI Application (`web/app.py`)
- `GET /health` — 健康检查，返回版本和服务状态
- `POST /api/v1/tasks` — 创建转换任务（input_path, output_path, metadata）
- `GET /api/v1/tasks/{task_id}` — 获取任务详情
- `GET /api/v1/tasks` — 任务列表（支持 status 过滤、分页）
- `PATCH /api/v1/tasks/{task_id}` — 更新任务状态
- `DELETE /api/v1/tasks/{task_id}` — 删除任务
- `POST /api/v1/tasks/{task_id}/cancel` — 取消任务（queued/processing）
- Pydantic 数据验证 + 详细错误响应

#### Task Store (`web/task_store.py`)
- `TaskStore` 类：线程安全内存任务存储
- 完整任务生命周期：queued → processing → completed / failed / cancelled
- `create_task()`: 创建任务，自动生成 UUID
- `get_task()`: 按 ID 获取
- `list_tasks()`: 分页列表 + status 过滤
- `update_status()`: 状态更新 + 时间戳追踪
- `cancel_task()`: 取消（仅 queued/processing 状态）
- `delete_task()`: 删除任务
- `count_tasks()`: 计数

#### Schemas (`web/schemas.py`)
- `CreateTaskRequest`: 创建任务请求模型
- `TaskResponse`: 任务响应模型（含完整状态字段）
- `TaskListResponse`: 任务列表响应（含分页信息）
- `UpdateTaskRequest`: 更新请求模型
- `HealthResponse`: 健康检查响应

### 3.5 测试覆盖

| 测试文件 | 测试数 | 覆盖范围 |
|----------|--------|----------|
| `test_pipeline.py` | 16 | 核心 pipeline + 端到端 |
| `test_spherical_injector.py` | 25 | ISOBMFF box 构建 + 查找 |
| `test_vr_metadata.py` | 16 | VR 元数据 XML + 注入 |
| `test_phase1_optimizations.py` | 24 | 设备检测 + 流式 + 分块超分 |
| `test_sbs_detection.py` | 7 | SBS 输入自动检测 |
| `test_orientation_matrix.py` | 17 | 方向矩阵诊断 |
| `test_temporal_outpainter.py` | 19 | 光流外绘 + 质量指标 |
| `test_web_api.py` | 30 | Web API 全端点 |
| **总计** | **152** | **全部通过** |

---

## 第四部分：项目架构

```
vr180-ai-pipeline/
├── pipeline/                        # 核心处理模块
│   ├── __init__.py
│   ├── depth_estimator.py           # Stage 1: Depth Anything V2 深度估计
│   ├── stereo_renderer.py           # Stage 2: 立体视差渲染
│   ├── equirectangular_mapper.py    # Stage 3: 等距柱状投影 (ffmpeg v360 + OpenCV)
│   ├── spherical_injector.py        # Stage 4: VR 元数据注入 (spatial-media CLI + ISOBMFF)
│   ├── vr_metadata.py               # Stage 4 备选: 原始 VR 元数据封装
│   ├── upscaler.py                  # Real-ESRGAN 超分 + 分块处理 + OpenCV fallback
│   ├── device_utils.py              # CUDA/MPS/CPU 自动检测
│   ├── streaming_pipeline.py        # 流式处理 pipeline
│   └── research/                    # R&D 研究模块
│       ├── __init__.py
│       ├── orientation_matrix.py    # VR180 方向诊断矩阵 (flip×transpose)
│       ├── ai_outpainter.py         # 光流时序 AI 外绘引擎
│       ├── temporal_outpainter.py   # 时序外绘器（简化版）
│       ├── benchmark_upscale.py     # 超分基准测试
│       └── test_inversion_matrix.py # 翻转矩阵测试工具
├── web/                             # Web API 基础设施
│   ├── __init__.py
│   ├── app.py                       # FastAPI REST API (health, task CRUD, cancel)
│   ├── schemas.py                   # Pydantic 请求/响应模型
│   └── task_store.py                # 线程安全内存任务存储
├── scripts/
│   ├── run_pipeline.py              # CLI 入口 (全部参数 + SBS 自动检测)
│   └── download_models.py           # 模型下载脚本
├── tests/
│   ├── test_pipeline.py             # 核心 pipeline 测试
│   ├── test_spherical_injector.py   # ISOBMFF 注入测试 (sv3d/st3d)
│   ├── test_vr_metadata.py          # VR 元数据测试
│   ├── test_phase1_optimizations.py # Phase 1 优化测试
│   ├── test_sbs_detection.py        # SBS 输入自动检测测试 (7 tests)
│   ├── test_orientation_matrix.py   # 方向矩阵诊断测试 (17 tests)
│   ├── test_temporal_outpainter.py  # 时序外绘器测试 (19 tests)
│   └── test_web_api.py              # Web API 端点测试 (30 tests)
├── docs/
│   ├── PRD-v2-vr180-studio.md       # 产品需求文档
│   ├── session-summary-and-dev-guide.md  # 开发文档 (本文件)
│   └── ...                          # 其他技术文档
├── video/                           # 测试视频和输出
├── pyproject.toml                   # Python 项目配置
├── requirements.txt                 # 依赖
├── Dockerfile                       # 容器化
├── CLINE_TASK_BOARD.md              # 开发任务看板
└── OVERNIGHT_RD_REPORT.md           # Overnight R&D 报告
```

---

## 第五部分：依赖与环境

### 核心依赖

```
torch>=2.0.0           # PyTorch (CUDA/MPS/CPU)
depth-anything-v2      # 深度估计模型
opencv-python>=4.8.0   # 图像处理
numpy>=1.24.0          # 数组操作
ffmpeg-python          # ffmpeg 绑定
tqdm                   # 进度条
spatial-media          # VR 元数据注入 (Google)
```

### Web API 依赖

```
fastapi>=0.100.0       # Web 框架
uvicorn>=0.23.0        # ASGI 服务器
httpx>=0.24.0          # HTTP 客户端（测试用）
pydantic>=2.0.0        # 数据验证
```

### 可选依赖

```
realesrgan             # Real-ESRGAN 超分
basicsr                # Real-ESRGAN 依赖
pytest                 # 测试框架
```

### 环境要求

- Python 3.10+
- ffmpeg（系统安装）
- macOS (MPS) 或 Linux (CUDA) 或 Windows (CUDA)

---

## 第六部分：CLI 使用指南

### 基础用法

```bash
# 最简单的转换
python scripts/run_pipeline.py -i video.mp4 -o vr180.mp4

# 流式处理（推荐大视频）
python scripts/run_pipeline.py -i video.mp4 -o vr180.mp4 --streaming

# 限制帧数测试
python scripts/run_pipeline.py -i video.mp4 -o vr180.mp4 --max-frames 30

# 自定义 FOV
python scripts/run_pipeline.py -i video.mp4 -o vr180.mp4 --src-hfov 90

# 带超分
python scripts/run_pipeline.py -i video.mp4 -o vr180.mp4 --upscale 2 --tiled-upscale

# 验证输入格式
python scripts/run_pipeline.py -i video.mp4 --validate-input
```

### 输出格式

- **分辨率**: 3840×1920（单眼）× 2（SBS）= 7680×1920
- **编码**: H.264（默认）或 H.265
- **VR 元数据**: sv3d + st3d（Spherical Video V2）
- **立体模式**: Side-by-Side (left-right)
- **投影**: Equirectangular（半球 180°）

---

## 第七部分：关键设计决策

### 1. 为什么用 ffmpeg v360 而不是纯 OpenCV？

- ffmpeg v360 是工业级的视频滤镜，处理速度快 10-50 倍
- OpenCV 的 remap 需要手动构建映射表，对于大分辨率（7680×1920）内存开销大
- v360 支持 fill_black，自动处理 FOV 外的区域
- **保留 OpenCV 作为 fallback**，在没有 ffmpeg 的环境也能工作

### 2. 为什么用 spatial-media CLI 而不是手动注入？

- MP4 ISOBMFF 格式极其复杂，手动修改二进制容易损坏文件
- stco/stsc/stsz/stts 等表需要精确同步更新
- Google spatial-media 是 VR 元数据的事实标准
- V2 spec（sv3d + st3d）是 Quest/YouTube/Facebook 的标准格式

### 3. 为什么需要 flip_vertical？

- VR 头显约定: 球面投影的 phi=0 是正上方（天顶）
- 2D 图像约定: y=0 是图像顶部
- ffmpeg v360 的 output=hequirect 输出时 phi=0 在图像底部
- 所以需要翻转才能让 VR 头显正确显示

### 4. 为什么分块超分用 512×512 瓦片？

- Real-ESRGAN 2× 超分时，512→1024 约需 4GB VRAM
- 4× 超分时，512→2048 约需 8GB VRAM
- 512 是大多数 GPU（8GB+）的安全范围
- padding=10 像素避免拼接缝

### 5. 流式处理的架构

```
cv2.VideoCapture → [逐帧读取]
    → DepthEstimator.estimate()    # GPU 上
    → StereoRenderer.render()      # CPU
    → EquirectangularMapper.map()  # ffmpeg subprocess
    → ffmpeg.stdin.write()         # pipe 写入
    → del frame, depth, left, right  # 立即释放
```

内存峰值 = 1帧 × (原始 + 深度 + 左眼 + 右眼 + SBS) ≈ 50-100MB，与视频长度无关。

---

## 第八部分：待开发功能 (Phase 4+)

### Phase 4: 部署 & 运维

| 功能 | 说明 | 优先级 |
|------|------|--------|
| Docker 优化 | 多阶段构建、CUDA 基础镜像 | P0 |
| CI/CD | GitHub Actions 自动测试 + 部署 | P0 |
| Storage | S3/R2 视频存储 | P1 |
| 任务队列 | Celery/Redis 后台异步任务 | P0 |
| 进度轮询 | WebSocket 实时渲染进度推送 | P0 |

### Phase 5: 公测

| 功能 | 说明 | 优先级 |
|------|------|--------|
| 限额系统 | 免费 3 个、付费无限制 | P0 |
| 结果保存 | 我的 VR 视频列表 | P0 |
| 交互预览 | A-Frame 180° 前后对比 | P1 |
| 空间视频 | MV-HEVC 自动转码 (Spatialify) | P1 |
| 多端下载 | Quest/Apple Vision Pro 格式 | P2 |
| 前端 UI | React/Next.js 用户界面 | P0 |

---

## 第九部分：CLI 新增参数

### SBS 检测相关

```bash
# 强制 SBS 模式（跳过自动检测）
python scripts/run_pipeline.py -i sbs_video.mp4 -o vr180.mp4 --force-sbs

# 自动检测（默认启用）
python scripts/run_pipeline.py -i 7680x1920_video.mp4 -o vr180.mp4
# → 自动识别为 SBS，跳过深度估计和立体渲染
```

### Web API 启动

```bash
# 启动 API 服务器
uvicorn web.app:app --host 0.0.0.0 --port 8000

# 创建任务
curl -X POST http://localhost:8000/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{"input_path": "video.mp4", "output_path": "vr180.mp4"}'

# 查询任务
curl http://localhost:8000/api/v1/tasks/{task_id}

# 列出所有任务
curl http://localhost:8000/api/v1/tasks?status=queued&limit=10&offset=0

# 健康检查
curl http://localhost:8000/health
```

### 方向矩阵诊断

```bash
# 在 Python 中使用
from pipeline.research.orientation_matrix import run_diagnostic
report = run_diagnostic("input_video.mp4", output_dir="orientation_output/")
```

### 时序外绘

```bash
# 在 Python 中使用
from pipeline.research.ai_outpainter import TemporalOutpainter
outpainter = TemporalOutpainter()
result_frames, metrics = outpainter.outpaint_sequence(
    frames, context_frames=None, num_iterations=3
)
```
