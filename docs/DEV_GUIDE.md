# VR180 Studio — 开发指南 (Dev Guide)

> **版本**: 1.0  
> **日期**: 2026-06-23  
> **适用对象**: Cline / Claude Code 等 AI 开发 Agent，以及参与本项目的人工开发者

---

## 目录

1. [项目全景](#1-项目全景)
2. [当前架构现状与问题](#2-当前架构现状与问题)
3. [目标架构](#3-目标架构)
4. [开发路线图（分阶段任务）](#4-开发路线图)
5. [CI/CD 流程规范](#5-cicd-流程规范)
6. [代码审查与测试流程](#6-代码审查与测试流程)
7. [Agent 协作规范](#7-agent-协作规范)
8. [关键技术决策记录](#8-关键技术决策记录)

---

## 1. 项目全景

### 1.1 核心目标

**VR180 Studio** 是一个 AI 驱动的 VR 视频生产工作台，核心业务流程分三步：

```
[Step 1] AI 视频生成
   用户描述场景 → AI 生成 FPV 第一视角飞行视频（2D, 16:9）
   使用引擎：Seedance / Kling / Wan2.2 / Google Veo / HappyHouse

[Step 2] VR180 格式转换
   2D 视频 → 深度估计 → 立体视差渲染 → 等距柱状投影 → 球面补全 → VR 元数据嵌入
   关键挑战：防止畸变、视场角适配、边缘像素 AI 补全

[Step 3] 质量提升与输出
   超分辨率（Real-ESRGAN / Video2X）→ 单眼 4K-8K 输出
   最终格式：SBS VR180 MP4，兼容 Meta Quest / YouTube VR / Apple Vision Pro
```

### 1.2 最终产品形态

参考 ComfyUI 的节点化工作流思路，最终产品是一个 **可对外发布的 Web 平台**：
- 非技术用户通过 Web UI 完成全部流程
- 每个阶段有预览+确认的 Checkpoint
- 技术用户可通过 REST API / CLI 直接调用

### 1.3 现有代码库状态（截至 2026-06-23）

| 模块 | 状态 | 备注 |
|------|------|------|
| `pipeline/depth_estimator.py` | ✅ 已实现 | Depth Anything V2, MPS/CUDA |
| `pipeline/stereo_renderer.py` | ✅ 已实现 | 视差渲染，含遮挡修复 |
| `pipeline/equirectangular_mapper.py` | ✅ 已实现 | 等距柱状投影 |
| `pipeline/spherical_injector.py` | ✅ 已实现 | VR 元数据嵌入 |
| `pipeline/upscaler.py` | ✅ 已实现 | Real-ESRGAN + OpenCV |
| `pipeline/streaming_pipeline.py` | ✅ 已实现 | 内存高效流式处理 |
| `pipeline/spatial_converter.py` | ✅ 已实现 | MV-HEVC / SBS 转换 |
| `pipeline/research/ai_outpainter.py` | ✅ 已实现 | 光流时序外扩 |
| `web/app.py` | ✅ 已实现 | FastAPI REST API |
| `web/task_store.py` | ✅ 已实现 | 内存任务存储（非持久化） |
| `web/quota.py` | ✅ 已实现 | 用户配额管理 |
| `web/storage.py` | ✅ 已实现 | 文件存储与元数据 |
| `web/static/` | ✅ 已实现 | 前端 SPA（基础版） |
| **AI 视频生成接入** | ❌ 缺失 | 无任何视频生成模型 API 集成 |
| **任务队列** | ❌ 缺失 | 当前 task_store 为内存存储，无真正异步队列 |
| **数据库持久化** | ❌ 缺失 | 重启后数据丢失 |
| **用户认证** | ❌ 缺失 | API 无鉴权 |
| **前端工作流 UI** | ⚠️ 基础版 | 缺少分步工作流、预览、节点编辑器 |

---

## 2. 当前架构现状与问题

### 2.1 关键问题清单

**P0（阻塞发布）**

1. **任务队列缺失**：`web/task_store.py` 是纯内存 dict，无法处理真正的长耗时异步任务（深度估计一个视频可能需要 5-30 分钟）。需引入 Celery + Redis 或 ARQ。

2. **AI 视频生成未集成**：工作流的第一步（视频生成）完全缺失。需要接入至少一个视频生成 API（推荐 Seedance 或 Kling，二者有国内可用的 API）。

3. **无持久化存储**：任务状态、用户配额重启即丢失。需接入数据库（SQLite 开发环境，PostgreSQL 生产）。

**P1（影响用户体验）**

4. **无用户认证**：API 完全开放，无法部署到公网。需接入 JWT 或 API Key 认证。

5. **前端工作流 UI 不完整**：当前 SPA 是基础展示页，缺少分步工作流 UI（分镜、生成、转换、预览 checkpoint）。

6. **球面补全（Outpainting）未集成进主流程**：`ai_outpainter.py` 在 research/ 目录下，未接入主流水线。

**P2（优化提升）**

7. **上传文件无大小限制**：生产环境需要文件大小校验和存储配额。

8. **CORS 完全开放**：`allow_origins=["*"]` 生产环境需收紧。

9. **无可观测性**：缺少结构化日志、指标监控（Prometheus）、错误追踪（Sentry）。

### 2.2 现有代码亮点（可复用）

- Pipeline 各模块设计清晰，接口干净，可直接复用
- 195 个测试，覆盖 pipeline 核心逻辑
- FastAPI schema 设计合理（Pydantic v2）
- Dockerfile 存在，容器化基础已有

---

## 3. 目标架构

### 3.1 系统架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                         用户层                                   │
│  Web 前端 (React SPA)     CLI (scripts/)     第三方 API 调用者   │
└────────────────┬───────────────┬────────────────────────────────┘
                 │               │
┌────────────────▼───────────────▼────────────────────────────────┐
│                    API 网关层 (FastAPI)                           │
│  /api/v1/auth    /api/v1/projects    /api/v1/tasks               │
│  /api/v1/generate  /api/v1/convert  /api/v1/export               │
│  认证中间件 (JWT)   速率限制   文件校验   CORS                     │
└─────────────────┬────────────────────────────────────────────────┘
                  │
┌─────────────────▼────────────────────────────────────────────────┐
│                    任务队列层 (Celery + Redis)                     │
│                                                                  │
│  generate_video_task    convert_to_vr180_task    upscale_task    │
│  depth_estimate_task    outpainting_task         export_task     │
└──────────────┬───────────────────────────────────────────────────┘
               │
┌──────────────▼───────────────────────────────────────────────────┐
│                    核心处理层                                      │
│                                                                  │
│  [AI 生成]              [VR180 转换]           [质量提升]          │
│  VideoGenClient         DepthEstimator         Upscaler          │
│  - Seedance API         StereoRenderer         - Real-ESRGAN     │
│  - Kling API            EquirectMapper         - Video2X         │
│  - Wan2.2 API           AIOutpainter                             │
│  - Google Veo API       SphericalInjector                        │
│                         SpatialConverter                         │
└──────────────┬───────────────────────────────────────────────────┘
               │
┌──────────────▼───────────────────────────────────────────────────┐
│                    数据层                                          │
│  PostgreSQL (任务/用户/项目)   Redis (队列/缓存)   S3/本地文件存储  │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 目录结构（目标状态）

```
vr180-ai-pipeline/
├── api/                        # FastAPI 应用（从 web/ 重构）
│   ├── main.py                 # 应用入口
│   ├── routers/
│   │   ├── auth.py             # 认证路由
│   │   ├── projects.py         # 项目管理
│   │   ├── tasks.py            # 任务 CRUD
│   │   ├── generate.py         # AI 视频生成
│   │   ├── convert.py          # VR180 转换
│   │   └── export.py           # 导出
│   ├── models/                 # SQLAlchemy ORM 模型
│   ├── schemas/                # Pydantic schemas
│   ├── middleware/             # 认证、限流、日志
│   └── dependencies.py         # FastAPI 依赖注入
│
├── workers/                    # Celery 任务
│   ├── celery_app.py           # Celery 配置
│   ├── generate_tasks.py       # 视频生成任务
│   ├── convert_tasks.py        # VR180 转换任务
│   └── upscale_tasks.py        # 超分任务
│
├── integrations/               # 外部 AI 服务接入
│   ├── base.py                 # 抽象基类
│   ├── seedance.py             # Seedance API
│   ├── kling.py                # Kling API
│   ├── wan22.py                # Wan2.2 API
│   └── google_veo.py           # Google Veo API
│
├── pipeline/                   # 核心处理模块（现有，继续扩展）
│   ├── depth_estimator.py
│   ├── stereo_renderer.py
│   ├── equirectangular_mapper.py
│   ├── outpainter.py           # 从 research/ 提升到主目录
│   ├── spherical_injector.py
│   ├── upscaler.py
│   └── streaming_pipeline.py
│
├── frontend/                   # React 前端（替换现有 web/static/）
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Storyboard.tsx  # 分镜板页面
│   │   │   ├── Generate.tsx    # 视频生成页面
│   │   │   ├── Convert.tsx     # VR180 转换页面
│   │   │   └── Export.tsx      # 导出页面
│   │   ├── components/
│   │   └── workflow/           # 工作流状态管理
│   └── package.json
│
├── db/
│   ├── migrations/             # Alembic 迁移文件
│   └── models.py
│
├── tests/                      # 现有测试（继续扩展）
├── scripts/                    # CLI 工具
├── docs/
├── .github/workflows/          # CI/CD
├── docker-compose.yml          # 本地开发环境
├── Dockerfile
└── pyproject.toml
```

---

## 4. 开发路线图

### Phase A：基础设施补全（优先级最高）

> 目标：让现有流程端对端跑通，消除 P0 问题

**A1 — 任务队列接入**
- 引入 Celery + Redis
- 将现有 `pipeline/streaming_pipeline.py` 封装为 Celery 任务
- 实现任务状态轮询 API（SSE 或 WebSocket）
- 文件：`workers/celery_app.py`, `workers/convert_tasks.py`
- 验收：提交一个 VR180 转换任务，关闭 API 进程重启后任务仍可继续

**A2 — 数据库持久化**
- 引入 SQLAlchemy + Alembic
- 迁移 `web/task_store.py` → 数据库
- 迁移 `web/quota.py` → 数据库
- 本地开发用 SQLite，生产用 PostgreSQL（通过 `DATABASE_URL` 环境变量切换）
- 文件：`db/models.py`, `db/migrations/`
- 验收：重启服务后任务列表不丢失

**A3 — 用户认证**
- 实现 API Key 认证（简单，适合初期）
- 可选：加 JWT（适合 Web 前端登录场景）
- 文件：`api/middleware/auth.py`
- 验收：无 API Key 的请求返回 401

---

### Phase B：AI 视频生成接入

> 目标：实现从"文字描述 → 2D 视频"的第一步

**B1 — 抽象接口设计**

```python
# integrations/base.py
class VideoGenClient(ABC):
    async def generate(
        self,
        prompt: str,
        duration_sec: float,
        aspect_ratio: str = "16:9",
        fps: int = 24,
        **kwargs
    ) -> AsyncIterator[GenerationEvent]:
        ...
```

**B2 — 优先接入 Kling API**
- Kling（快手可灵）有国内可访问的 API，文档完整
- 支持 FPV 风格提示词，生成效果稳定
- 文件：`integrations/kling.py`
- 验收：通过 API 生成一个 5 秒的 FPV 飞行视频

**B3 — 接入 Seedance（字节跳动）**
- 文件：`integrations/seedance.py`

**B4 — Prompt 工程模块**
- 基于现有 `docs/prompt-design-guide.md`，实现 FPV 专用 Prompt 生成器
- 输入：场景描述（中文）→ 输出：视频生成 API 的 prompt（英文，带 FPV 相关关键词）
- 文件：`api/routers/generate.py` 中的 `build_fpv_prompt()` 函数

---

### Phase C：前端工作流 UI

> 目标：实现分步工作流，每步有预览和确认

**C1 — 技术选型**
- 框架：React 18 + TypeScript + Vite
- UI 组件：Shadcn/UI（基于 Tailwind CSS，无需额外配置）
- 状态管理：Zustand
- 视频预览：Video.js 或原生 `<video>`
- VR 预览：three.js（WebXR 支持等距柱状投影预览）

**C2 — 核心页面**

1. **分镜板页（Storyboard）**
   - 输入场景描述
   - AI 生成分镜脚本（每个镜头的 prompt）
   - 可手动编辑每个镜头

2. **视频生成页（Generate）**
   - 选择 AI 引擎（Kling / Seedance / …）
   - 显示每个镜头的生成进度（实时）
   - 生成完成后可预览，不满意可重生成

3. **VR180 转换页（Convert）**
   - 选择转换参数（IPD、视差强度、投影方式）
   - 显示转换进度（含每个子阶段：深度估计、立体渲染、投影、补全）
   - 预览 VR180 效果（three.js 球面投影预览）

4. **导出页（Export）**
   - 选择分辨率（4K / 8K）
   - 下载最终 MP4

---

### Phase D：生产化

> 目标：使系统可以稳定地对外提供服务

**D1 — 可观测性**
- 结构化日志（`structlog` 库）
- Prometheus 指标（任务耗时、成功率、队列深度）
- Sentry 错误追踪

**D2 — 性能优化**
- 深度估计批处理优化（已有帧流式处理，需验证 GPU 利用率）
- 大文件分片上传（前端 + 后端）
- CDN 加速视频预览（生产环境）

**D3 — 部署**
- Docker Compose（开发环境：API + Worker + Redis + PostgreSQL）
- 生产部署文档（推荐：Kubernetes 或 Railway.app）

---

## 5. CI/CD 流程规范

### 5.1 分支策略

```
main          ← 保护分支，只接受 PR 合并，永远可部署
  └── develop ← 开发集成分支，日常工作基础
        ├── feat/A1-celery-task-queue    ← 功能分支
        ├── feat/B1-video-gen-interface
        ├── fix/xxx-bug
        └── chore/update-deps
```

**命名规范**：
- `feat/[phase]-[简短描述]`：新功能
- `fix/[简短描述]`：Bug 修复
- `chore/[简短描述]`：工具链、依赖、文档
- `refactor/[简短描述]`：重构（不改变功能）

**提交信息规范（Conventional Commits）**：
```
feat(workers): add celery task queue for conversion jobs
fix(depth): handle edge case when frame has all-black regions
chore(deps): upgrade torch to 2.4.1
test(api): add integration tests for task CRUD endpoints
docs(dev-guide): update phase B roadmap
```

### 5.2 GitHub Actions 工作流

**文件路径**：`.github/workflows/`

#### `ci.yml` — 主 CI 流程（每个 PR 触发）

```yaml
name: CI

on:
  pull_request:
    branches: [main, develop]
  push:
    branches: [develop]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install ruff mypy
      - run: ruff check .
      - run: mypy pipeline/ api/ workers/ --ignore-missing-imports

  test:
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v --tb=short --cov=pipeline --cov=api --cov-report=xml
      - uses: codecov/codecov-action@v4

  docker-build:
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4
      - run: docker build -t vr180-studio:ci .
```

#### `release.yml` — 发布流程（push to main 触发）

```yaml
name: Release

on:
  push:
    branches: [main]

jobs:
  deploy-staging:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build and push Docker image
        run: |
          docker build -t vr180-studio:${{ github.sha }} .
          # push to registry
      - name: Deploy to staging
        run: |
          # Railway / Render / Fly.io 部署命令

  integration-test:
    runs-on: ubuntu-latest
    needs: deploy-staging
    steps:
      - run: pytest tests/integration/ --base-url=$STAGING_URL
```

### 5.3 本地开发环境启动

```bash
# 1. 克隆仓库，进入项目
git clone <repo> && cd vr180-ai-pipeline

# 2. 创建 Python 虚拟环境
python3.11 -m venv .venv && source .venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt
pip install -r requirements-dev.txt   # 开发依赖（ruff, mypy, pytest 等）

# 4. 启动基础设施（Redis + PostgreSQL）
docker compose up -d redis db

# 5. 初始化数据库
alembic upgrade head

# 6. 启动 API 服务
uvicorn api.main:app --reload --port 8000

# 7. 启动 Celery Worker（新终端）
celery -A workers.celery_app worker --loglevel=info

# 8. 访问
open http://localhost:8000
```

---

## 6. 代码审查与测试流程

### 6.1 测试分层

```
tests/
├── unit/               # 单元测试（无 I/O，纯函数逻辑）
│   ├── test_depth.py
│   ├── test_stereo.py
│   └── test_prompt_builder.py
├── integration/        # 集成测试（需要数据库/Redis，但不需要 GPU）
│   ├── test_api_tasks.py
│   ├── test_task_queue.py
│   └── test_auth.py
├── e2e/                # 端对端测试（完整流程，需要 GPU）
│   └── test_full_pipeline.py
└── fixtures/           # 测试用的视频/图片素材（小文件）
    ├── test_video_720p.mp4
    └── test_frame.jpg
```

**测试要求**：
- 每个新功能模块必须有单元测试，**覆盖率不低于 80%**
- API 路由必须有集成测试
- 端对端测试在 CI 中默认跳过（需要 GPU），但本地开发时应定期运行
- Mock 规则：外部 AI API（Kling/Seedance）在单元/集成测试中必须 Mock；Pipeline 模型（Depth Anything）在单元测试中 Mock，集成测试中使用真实模型的最小版本

### 6.2 代码审查 Checklist

每个 PR 合并前，Cline/开发者需自查以下内容：

**功能正确性**
- [ ] 新功能按 PRD/任务描述实现
- [ ] 边界条件有处理（空输入、超大文件、GPU OOM）
- [ ] 没有引入 regression（全量测试通过）

**代码质量**
- [ ] 无多余注释（代码应自解释，注释只留 WHY）
- [ ] 函数单一职责，超过 50 行的函数需拆分
- [ ] 无 hardcode 的路径/密钥/配置（用环境变量或 config 文件）
- [ ] 无 `print()` 调试语句（用 `logging`）

**安全性**
- [ ] 用户输入有校验（文件类型、大小、内容）
- [ ] 无命令注入风险（subprocess 调用需用 list 而非 shell=True）
- [ ] 敏感信息（API Key）不写入日志

**性能**
- [ ] 大文件/大批量处理有流式处理，不全量加载到内存
- [ ] 数据库查询有适当索引（N+1 问题检查）

**测试**
- [ ] 新代码有对应测试
- [ ] `pytest tests/ -v` 全部通过
- [ ] `ruff check .` 无错误
- [ ] `mypy pipeline/ api/` 无类型错误

### 6.3 AI Agent 提交前必做步骤

```bash
# 每次提交前运行以下命令，确保全部通过才能提交
ruff check . --fix          # 自动修复 lint 问题
mypy pipeline/ api/ workers/ --ignore-missing-imports
pytest tests/unit/ tests/integration/ -v --tb=short
```

---

## 7. Agent 协作规范

### 7.1 任务分解原则

每个 Cline 开发 session 应专注于 **一个明确的功能单元**，包含：

1. **目标**：做什么（对应路线图中的哪个子任务）
2. **输入文件**：需要读取/修改哪些文件
3. **输出文件**：需要创建/修改哪些文件
4. **验收标准**：如何验证完成（具体命令 + 预期输出）
5. **不做什么**：明确边界，避免 scope creep

### 7.2 任务卡片模板（供人工向 Cline 下发任务用）

```markdown
## 任务：[Phase X]-[功能名]

**目标**：
[一句话描述]

**背景**：
[为什么要做这个，依赖哪些已有代码]

**需要创建/修改的文件**：
- `workers/celery_app.py`（新建）
- `workers/convert_tasks.py`（新建）
- `api/routers/tasks.py`（修改：添加任务提交端点）

**验收标准**：
```bash
# 命令 1：启动 Celery Worker
celery -A workers.celery_app worker --loglevel=info

# 命令 2：提交转换任务（预期：返回 task_id）
curl -X POST http://localhost:8000/api/v1/tasks \
  -F "file=@test.mp4" \
  -F "type=convert"

# 命令 3：查询任务状态（预期：status 变为 "completed"）
curl http://localhost:8000/api/v1/tasks/{task_id}

# 命令 4：测试通过
pytest tests/integration/test_task_queue.py -v
```

**不做**：
- 不修改前端
- 不修改 pipeline/ 核心模块
- 不引入数据库（Task A2 负责）
```

### 7.3 进度检查规范

每个阶段完成后，由 **Claude Code（本 AI）** 执行以下检查：

1. **代码审查**：阅读新增/修改文件，对照 Checklist 检查
2. **测试验证**：运行全量测试，确认 195+N 个测试通过
3. **集成验证**：端对端运行新功能，截图或命令输出为证
4. **文档更新**：更新 `CLINE_TASK_BOARD.md` 中的任务状态
5. **反馈给下一个 Agent**：描述当前状态，给出下一步任务卡片

---

## 8. 关键技术决策记录

### ADR-001：任务队列选型

- **决策**：使用 Celery + Redis
- **理由**：Celery 是 Python 生态最成熟的任务队列，文档完整；Redis 兼做缓存和消息中间件；团队熟悉度高；有 GPU 任务监控支持（Flower）
- **备选方案**：ARQ（更轻量，但功能少）；Ray（适合 ML 分布式，过重）

### ADR-002：数据库选型

- **决策**：SQLAlchemy + Alembic；开发用 SQLite，生产用 PostgreSQL
- **理由**：SQLAlchemy 是 Python 标准；Alembic 保证 schema 演进可控；SQLite 无需额外服务，降低本地开发门槛
- **环境变量**：`DATABASE_URL=sqlite:///./dev.db`（开发），`DATABASE_URL=postgresql://...`（生产）

### ADR-003：AI 视频生成接入顺序

- **决策**：先 Kling，再 Seedance，再 Google Veo
- **理由**：Kling API 国内可访问、文档完整、FPV 效果好；Seedance 为备选；Google Veo 需要特殊权限，排后

### ADR-004：前端框架选型

- **决策**：React 18 + TypeScript + Vite + Shadcn/UI
- **理由**：TypeScript 保证类型安全；Vite 构建快；Shadcn/UI 组件可直接 copy 进项目，无锁定风险；替换现有纯 HTML/JS SPA

### ADR-005：球面补全方案

- **决策**：默认用高斯模糊渐变（方案 B），高质量模式用 AI Outpainting（方案 A）
- **理由**：高斯模糊速度快，效果可接受；AI Outpainting 质量更好但慢 10x；让用户选择

---

## 附录：常用命令速查

```bash
# 运行所有测试
pytest tests/ -v

# 运行单个测试文件
pytest tests/test_web_api.py -v

# 代码检查
ruff check . && mypy pipeline/ api/ workers/

# 启动完整开发环境
docker compose up -d
uvicorn api.main:app --reload &
celery -A workers.celery_app worker &

# 下载模型权重
python scripts/download_models.py

# 运行完整 pipeline（CLI）
python scripts/run_pipeline.py --input video.mp4 --output vr180.mp4
```
