# 🎭 VR180 Studio — Active Task Board

> **主脑（Claude Code）负责任务设计与审查；Cline 负责编码执行。**
> Cline 每次启动后：读本文件 → 找到第一个 `[ ] In Progress` 或最高优先级未完成任务 → 立即开始执行。

---

## ✅ 已完成阶段（历史）

<details>
<summary>展开查看历史完成任务</summary>

- [x] Phase 0: SBS Input Detection, Orientation Matrix
- [x] Phase 1: Device detection, streaming pipeline, tiled upscaling
- [x] Phase 3: AI Temporal Outpainting (optical flow)
- [x] Phase 3.5: FastAPI REST API, Task Store, Schemas
- [x] Phase 4: Quota, Storage, Spatial Converter, Frontend SPA, v1 API — **195/195 tests passing**

</details>

---

## 🔴 Phase A — 基础设施补全（当前阶段）

> **目标**：解决 3 个 P0 问题，让系统具备真正的生产能力。
> **执行顺序**：A1 → A2 → A3（严格按序，每个完成后再开始下一个）

---

### [x] A1 — Celery 异步任务队列

**状态**: ✅ Completed  
**优先级**: P0 — 阻塞后续所有阶段  
**预计工作量**: 中等（约 200-300 行新代码 + 测试）

#### 背景

现有 `web/task_store.py` 是纯内存 dict，`TaskStatus` 只是枚举值，没有真正的异步执行能力。VR180 转换任务耗时 5-30 分钟，必须有真正的后台队列。

#### 需要安装的依赖

```bash
pip install celery[redis]==5.4.0 redis==5.0.7
# 追加到 requirements.txt
```

#### 需要创建的文件

**1. `workers/__init__.py`** — 空文件

**2. `workers/celery_app.py`** — Celery 配置

```python
"""Celery application configuration."""
import os
from celery import Celery

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

app = Celery(
    "vr180_studio",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["workers.convert_tasks"],
)

app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,  # GPU 任务不预取
    result_expires=86400,  # 结果保留 24 小时
)
```

**3. `workers/convert_tasks.py`** — VR180 转换 Celery 任务

```python
"""Celery tasks for VR180 conversion pipeline."""
import logging
from pathlib import Path
from celery import shared_task
from workers.celery_app import app

log = logging.getLogger(__name__)

@app.task(bind=True, name="convert.vr180", max_retries=2)
def convert_to_vr180(self, input_path: str, output_dir: str, params: dict) -> dict:
    """
    Full VR180 conversion pipeline task.
    
    params keys:
      - depth_model: "small" | "base" | "large"  (default: "small")
      - upscale_factor: 2 | 4  (default: 2)
      - outpainting: bool  (default: False)
    
    Returns: {"output_path": str, "metadata": dict}
    """
    from pipeline.streaming_pipeline import StreamingPipeline
    
    self.update_state(state="STARTED", meta={"stage": "initializing", "progress": 0})
    
    try:
        pipeline = StreamingPipeline(
            depth_model=params.get("depth_model", "small"),
        )
        
        self.update_state(state="PROGRESS", meta={"stage": "depth_estimation", "progress": 10})
        
        output_path = pipeline.process(
            input_path=input_path,
            output_dir=output_dir,
            progress_callback=lambda pct, stage: self.update_state(
                state="PROGRESS",
                meta={"stage": stage, "progress": pct}
            ),
        )
        
        return {
            "output_path": str(output_path),
            "metadata": {"input": input_path, "params": params},
        }
    except Exception as exc:
        log.exception("convert_to_vr180 failed: %s", exc)
        raise self.retry(exc=exc, countdown=30)


@app.task(bind=True, name="convert.depth_only", max_retries=1)
def estimate_depth_only(self, input_path: str, output_dir: str, model_size: str = "small") -> dict:
    """Standalone depth estimation task (for preview)."""
    from pipeline.depth_estimator import DepthEstimator
    import cv2, numpy as np
    
    self.update_state(state="STARTED", meta={"stage": "loading_model", "progress": 0})
    
    estimator = DepthEstimator(model_size=model_size)
    cap = cv2.VideoCapture(input_path)
    
    depth_frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for i in range(min(total, 30)):  # preview: 최대 30프레임
        ret, frame = cap.read()
        if not ret:
            break
        depth = estimator.estimate(frame)
        depth_frames.append(depth)
        self.update_state(state="PROGRESS", meta={"stage": "depth", "progress": int(i / min(total, 30) * 100)})
    
    cap.release()
    
    out_path = Path(output_dir) / "depth_preview.npy"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), np.array(depth_frames))
    
    return {"depth_preview_path": str(out_path), "frames_processed": len(depth_frames)}
```

**4. `workers/upscale_tasks.py`** — 超分任务

```python
"""Celery tasks for video upscaling."""
from workers.celery_app import app

@app.task(bind=True, name="upscale.video", max_retries=1)
def upscale_video(self, input_path: str, output_path: str, factor: int = 2) -> dict:
    from pipeline.upscaler import Upscaler
    
    self.update_state(state="STARTED", meta={"stage": "upscaling", "progress": 0})
    upscaler = Upscaler(scale_factor=factor)
    upscaler.upscale_video(
        input_path=input_path,
        output_path=output_path,
        progress_callback=lambda pct: self.update_state(
            state="PROGRESS", meta={"stage": "upscaling", "progress": pct}
        ),
    )
    return {"output_path": output_path}
```

#### 需要修改的文件

**5. `web/app.py`** — 在任务提交端点中调用 Celery 任务

在现有 `POST /api/v1/tasks` 端点中，当 `task_type == "convert"` 时：

```python
from workers.convert_tasks import convert_to_vr180

# 提交 Celery 任务
celery_task = convert_to_vr180.apply_async(
    kwargs={
        "input_path": str(upload_path),
        "output_dir": str(_OUTPUT_DIR / task_id),
        "params": body.params or {},
    }
)
# 将 celery_task.id 存入 task_store，以便后续查询进度
```

新增端点 `GET /api/v1/tasks/{task_id}/progress`，返回 Celery 任务实时进度：

```python
from celery.result import AsyncResult
from workers.celery_app import app as celery_app

@router.get("/tasks/{task_id}/progress")
def get_task_progress(task_id: str):
    result = AsyncResult(task_id, app=celery_app)
    return {
        "state": result.state,
        "progress": result.info.get("progress", 0) if result.info else 0,
        "stage": result.info.get("stage", "") if result.info else "",
    }
```

#### 需要创建的测试文件

**6. `tests/test_celery_tasks.py`**

```python
"""Tests for Celery tasks (mocked, no real Celery broker needed)."""
import pytest
from unittest.mock import patch, MagicMock

def test_convert_task_exists():
    from workers.convert_tasks import convert_to_vr180
    assert convert_to_vr180 is not None

def test_convert_task_signature():
    from workers.convert_tasks import convert_to_vr180
    # 验证任务名称注册正确
    assert convert_to_vr180.name == "convert.vr180"

def test_upscale_task_exists():
    from workers.upscale_tasks import upscale_video
    assert upscale_video.name == "upscale.video"

def test_celery_app_configured():
    from workers.celery_app import app
    assert app.conf.task_serializer == "json"
    assert app.conf.task_track_started is True

@patch("workers.convert_tasks.StreamingPipeline")
def test_convert_task_calls_pipeline(mock_pipeline_cls):
    """Test that convert task instantiates and calls StreamingPipeline."""
    mock_instance = MagicMock()
    mock_instance.process.return_value = "/tmp/output.mp4"
    mock_pipeline_cls.return_value = mock_instance
    
    # 用 apply 同步执行（绕过 broker）
    from workers.convert_tasks import convert_to_vr180
    from workers.celery_app import app
    app.conf.task_always_eager = True  # 同步模式
    
    result = convert_to_vr180.apply(
        kwargs={
            "input_path": "/tmp/test.mp4",
            "output_dir": "/tmp/out",
            "params": {"depth_model": "small"},
        }
    ).get()
    
    assert result["output_path"] == "/tmp/output.mp4"
    mock_instance.process.assert_called_once()
```

#### 需要更新的文件

**7. `requirements.txt`** — 追加依赖：
```
celery[redis]==5.4.0
redis==5.0.7
```

**8. `docker-compose.yml`** — 新建（如果不存在），添加 Redis 服务：

```yaml
version: "3.9"
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis
    command: uvicorn web.app:app --host 0.0.0.0 --port 8000 --reload

  worker:
    build: .
    environment:
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis
    command: celery -A workers.celery_app worker --loglevel=info --concurrency=1

volumes:
  redis_data:
```

#### 验收标准

验收分两层：**自动化验收**（Cline 自行完成）+ **人工 Quest 实测**（由用户完成）。

##### 层 1 — 自动化验收（Cline 完成后提交）

```bash
# 1. 安装依赖
pip install celery[redis]==5.4.0 redis==5.0.7

# 2. 启动 Redis（Docker）
docker run -d -p 6379:6379 redis:7-alpine

# 3. 启动 Celery Worker，确认输出 "ready."
celery -A workers.celery_app worker --loglevel=info

# 4. 用本地 FPV 视频触发完整转换任务（异步）
# 将视频放入 data/uploads/test_fpv.mp4，然后：
python scripts/run_pipeline.py --input data/uploads/test_fpv.mp4 --output data/outputs/test_vr180.mp4
# 预期：输出 test_vr180.mp4，文件大小 > 0

# 5. 运行新测试
pytest tests/test_celery_tasks.py -v

# 6. 运行全量回归测试
pytest tests/ -v --ignore=tests/e2e
# 预期：195 + 新增测试，全部通过

# 7. 代码检查
ruff check workers/ --fix
```

Cline 完成后需在本文件写入以下信息：
```
输出文件路径: data/outputs/test_vr180.mp4
文件大小: XX MB
分辨率: XXXX × XXXX
```

##### 层 2 — Quest 实测（用户人工验收，决定是否进入 A2）

> ⚠️ 此步骤由**用户**完成，Cline 不参与。

**步骤**：
1. 将 `data/outputs/test_vr180.mp4` 传输到 Meta Quest（通过 ADB 或 Air Link）
2. 在 Quest 上用 **Skybox VR Player** 或 **DeoVR** 打开视频
3. 播放模式选择：**180° SBS（Side-by-Side）**

**检查项**（用户实测后反馈给 Claude Code）：

| 检查项 | 通过标准 |
|--------|---------|
| 立体感 | 有明显的景深层次，不是平面图 |
| 畸变 | 画面无明显拉伸变形，边缘可接受 |
| 分辨率 | 清晰度可接受（不模糊到影响观看） |
| 方向 | 头部转动与画面方向一致，无颠倒 |
| 黑边 | 视场角边缘黑边面积（小/中/大）|

用户实测后告知结果，Claude Code 根据反馈决定：
- 全部通过 → 开始 A2
- 有问题 → 分析原因，给 Cline 下发修复任务

#### 不做的事（边界）

- 不修改 `pipeline/` 目录下任何现有文件
- 不引入数据库（那是 A2 的任务）
- 不修改前端
- 不修改现有 195 个测试

---

### [ ] A2 — 数据库持久化

**状态**: 等待 A1 完成后开始  
**依赖**: A1 完成

- 引入 SQLAlchemy 2.0 + Alembic
- 将 `web/task_store.py` 的内存存储迁移到 SQLite（开发）/PostgreSQL（生产）
- 将 `web/quota.py` 的配额数据迁移到数据库
- `DATABASE_URL` 环境变量控制数据库类型

---

### [ ] A3 — API Key 认证

**状态**: 等待 A2 完成后开始  
**依赖**: A2 完成

- 实现 `X-API-Key` header 认证
- API Key 存储在数据库
- 无 Key 请求返回 401

---

## 🔵 Phase Q — 画质提升（核心竞争力）

> 战略依据：`docs/STRATEGY_AI_VR180.md`
> ⚠️ **依赖 NVIDIA CUDA GPU**。Mac 本机无法真实验证；Cline 可完成代码封装 + mock 测试，真实跑通需云端/本地 GPU。
> 优先级与排期待用户确认（质量优先 vs 平台优先）。

### [ ] Q1 — 抽象 depth/stereo 后端接口

- 定义 `pipeline/backends/base.py`：`DepthBackend` / `StereoBackend` 抽象基类
- 现有 Depth Anything V2 + 简单视差封装为 `baseline` 后端（Mac 可跑，快速预览档）
- 通过 `--quality {fast,high}` 或环境变量切换后端

### [ ] Q2 — 集成 Video-Depth-Anything（时序一致深度）

- 仓库：https://github.com/DepthAnything/Video-Depth-Anything
- 替换逐帧深度，消除闪烁/抖动
- 作为 `high` 档 DepthBackend

### [ ] Q3 — 集成 StereoCrafter（高保真立体对 + AI 补全）

- 仓库：https://github.com/TencentARC/StereoCrafter
- depth-based splatting + stereo inpainting，解决左右眼空洞/背景不一致
- 作为 `high` 档 StereoBackend

### [ ] Q4 — 上下黑边 AI outpainting

- 等距投影后上下约 30% 为黑边
- 复用现有 `temporal_outpainter.py` 或接入扩散模型补全天空/地面

---

## 🟡 Phase B — AI 视频生成接入（等待 Phase A 全部完成）

### [ ] B1 — VideoGen 抽象接口 + Kling API 接入

**状态**: 等待 Phase A 完成  

> ⚠️ 调研结论：Kling/Veo/Seedance **均不支持原生立体输出**，B1 仅生成 2D 源视频，
> 立体化由 Phase Q 完成。Prompt 需内置广角/低空/FPV 关键词（见 STRATEGY 文档第七节）。

- 创建 `integrations/base.py` 抽象基类
- 实现 `integrations/kling.py`（快手可灵 API）
- 实现 FPV 专用 Prompt 构建器

---

## 🟢 Phase C — 前端工作流 UI（等待 Phase B 完成）

**状态**: 等待 Phase B 完成  
详见 `docs/DEV_GUIDE.md` Phase C 章节

---

## 📋 任务完成后 Cline 必须执行的收尾步骤

1. 更新本文件（将对应任务标记为 `[x]`，填写完成时间）
2. 运行 `pytest tests/ -v --ignore=tests/e2e` 确认测试全绿
3. 运行 `ruff check . --fix` 确认 lint 干净
4. `git add -A && git commit -m "feat(workers): add celery task queue for vr180 conversion"`
