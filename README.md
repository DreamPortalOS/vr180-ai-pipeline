# VR180 AI Pipeline / VR180 Studio

> 把 AI 生成的「第一视角 FPV 飞行视频」自动转换成可在 Meta Quest / Apple Vision Pro 等头显沉浸观看的 **VR180 立体视频**，并逐步发展为一个「从创意到成片」的在线工作流平台（类 ComfyUI）。

本文件是项目的**总入口**。任何人（包括其他设备上的 Claude Code）读完本文件即可理解项目目标、架构、开发协作方式，并直接上手继续工作。

---

## 一、项目目标与背景

### 核心目标
依托真实的地理环境，**生成第一视角（FPV）的飞行视频**，并将其转换为高清 VR180 沉浸内容。完整工作流分三步：

1. **视频生成**：用 Google Veo / 字节 Seedance / 快手 Kling 等视频生成模型，生成 FPV 第一视角飞行视频。
2. **格式转化与适配**：把生成的 2D 视频转换为 VR180 格式，包含：
   - (a) 防止畸变处理
   - (b) VR 适配（等距投影 / 立体视差）
   - (c) 周边像素补充（AI 外绘 outpainting，或在生成阶段就按 VR 构图生成）
3. **质量提升**：超分（upscale）到 4K–8K，确保头显观看清晰度。

### 终极愿景
做成一个**在线工作流平台**（可发布的网站或内部工具，类似 ComfyUI），覆盖「策划创意 → 视频生成 → VR180 成片」的全链路。

### 关键技术判断（来自实测）
- **没有任何模型支持「原生 VR180 / 双眼立体」直接输出**——必须走「2D 生成 → 深度估计 → 立体重建 → 投影」的转换路线。详见 [docs/STRATEGY_AI_VR180.md](docs/STRATEGY_AI_VR180.md)。
- **VR 致晕的主因是「立体视差精度」，不是分辨率**。根治靠高质量立体重建（StereoCrafter / DepthCrafter），缓解靠在生成阶段控制运动（慢速、稳定地平线）。详见 [docs/PROMPT_GUIDE_VR180.md](docs/PROMPT_GUIDE_VR180.md)。
- 竞品 buildvr.ai 刻意只做「360° 单目」规避立体难题；我们做**真立体 + AI 原生生成**作为差异化。详见 [docs/COMPETITOR_AND_BUSINESS.md](docs/COMPETITOR_AND_BUSINESS.md)。

---

## 二、技术管线（Pipeline）

```
2D FPV 视频
   │
   ├─ Stage 1  深度估计 (Depth Anything V2 / MiDaS)
   ├─ Stage 2  立体视差渲染 (左右眼视图 + 空洞修补)
   ├─ Stage 3  等距投影 (平面 → 180° 半球, 每眼 3840×1920, SBS 7680×1920)
   ├─ Stage 4  VR 元数据嵌入 (Spherical V2 / st3d / sv3d)
   └─（可选）超分到 4K–8K + 边缘 AI 外绘
        │
        ▼
   VR180 SBS 视频 (H.264/H.265)
```

源相机视场角默认 `src_hfov=120°`（匹配 AI 生成的广角 FPV 素材）。投影**不做垂直翻转**（ffmpeg v360 hequirect 输出已符合 Quest/YouTube 规范）。

---

## 三、代码架构

```
vr180-ai-pipeline/
├── README.md                    # 本文件（总入口）
├── CLINE_TASK_BOARD.md          # ★ 任务看板（主脑派活、Cline 执行的依据）
├── .clinerules                  # ★ Cline 自主开发协议（YOLO 模式 + 工程纪律）
├── .pre-commit-config.yaml      # 提交前 ruff 自动检查/格式化
├── pyproject.toml               # ruff/pytest 配置
├── requirements.txt
├── docker-compose.yml           # Redis + API + Worker
│
├── pipeline/                    # 核心转换管线
│   ├── depth_estimator.py       #   深度估计
│   ├── stereo_renderer.py       #   立体视差渲染
│   ├── equirectangular_mapper.py#   等距投影（已修正方向/FOV）
│   ├── vr_metadata.py           #   VR 元数据
│   ├── spherical_injector.py    #   sv3d/st3d 注入（spatialmedia，有 ffmpeg fallback）
│   ├── upscaler.py              #   超分（含平滑 tile blending）
│   ├── streaming_pipeline.py    #   流式管线编排
│   ├── prompt_builder.py        # ★ VR180 友好 Prompt 包装层（wrap_prompt_for_vr180）
│   └── research/                #   研究原型
│       ├── ai_outpainter.py     #     AI 外绘（支持 SDXL / DALL·E 后端）
│       ├── temporal_outpainter.py#    时序传播外绘
│       ├── benchmark_upscale.py #     超分模型基准测试
│       └── test_inversion_matrix.py#  多假设翻转矩阵验证
│
├── workers/                     # Celery 异步任务队列
│   ├── celery_app.py
│   ├── convert_tasks.py         #   VR180 转换任务
│   └── upscale_tasks.py
│
├── web/                         # FastAPI 后端
│   ├── app.py                   #   REST API（任务、生成、健康检查等）
│   ├── auth.py                  #   API Key 认证
│   ├── task_store.py / quota.py / storage.py / schemas.py
│   └── static/                  #   前端 SPA
│       ├── index.html / app.js / styles.css
│
├── db/                          # SQLAlchemy 2.0 持久层
│   ├── engine.py                #   get_session_factory / SessionLocal / init_db
│   └── models.py                #   TaskRecord / QuotaRecord / ApiKey
├── alembic/                     # 数据库迁移
│
├── integrations/                # 视频生成 Provider 抽象
│   ├── base.py                  #   VideoGenProvider ABC
│   └── kling.py / seedance.py / veo.py
│
├── notifications/               # ★ 通知推送模块（Hermes Agent）
│   └── feishu.py                #   飞书 webhook 消息推送
│
├── scripts/
│   ├── run_pipeline.py          #   CLI 跑完整管线
│   ├── create_api_key.py        #   API Key 生成工具
│   ├── watch_and_notify.py      #   ★ 目录监听 + 飞书通知
│   └── download_models.py
├── tests/                       # pytest（CI 权威把关）
├── video/                       # 测试素材与输出（git 忽略大文件）
└── docs/                        # 详细文档（见下方索引）
```

---

## 四、快速上手

### 环境要求
- Python 3.10+
- ffmpeg（`brew install ffmpeg`）
- 转换管线：Mac（MPS）可跑 baseline；高质量立体/超分需 NVIDIA CUDA GPU（RTX 3060 12GB 起）
- 可选：Redis（异步队列）、Docker

### 安装
```bash
git clone https://github.com/DreamPortalOS/vr180-ai-pipeline.git
cd vr180-ai-pipeline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install pre-commit && pre-commit install   # ★ 必装：提交前自动 ruff
```

### 跑一次 2D→VR180 转换
```bash
PYTHONPATH=$(pwd) python scripts/run_pipeline.py \
  --input video/你的素材.mp4 \
  --output video/output_vr180.mp4 \
  --src-hfov 120 --model-size small --crf 18
```
输出为 7680×1920 SBS。传到 Quest，用 Skybox VR / DeoVR 选「180° 3D（SBS）」播放。

### 飞书通知（Hermes Agent）
```bash
# 设置飞书 webhook
export FEISHU_WEBHOOK_URL="https://open.feishu.cn/open-apis/bot/v2/hook/xxx"

# 手动通知当前目录下最新 VR180 文件
python scripts/watch_and_notify.py --once

# 持续监听 video/ 目录，新文件自动通知
python scripts/watch_and_notify.py --watch-dir video/
```

### 跑测试
```bash
pytest tests/ -q --ignore=tests/e2e     # 单元测试
ruff check . && ruff format --check .   # lint + 格式
```

---

## 五、开发协作模式（★ 接手前必读）

本项目采用「**主脑 + 执行者 + 真人验收**」三角协作：

| 角色 | 职责 |
|------|------|
| **主脑（Claude Code / claude.ai/code）** | 分析调研、设计任务、写任务规格、审查 Cline 的 PR、做架构决策。**不直接写大量业务代码**。 |
| **执行者（VS Code 里的 Cline 插件）** | 按 `CLINE_TASK_BOARD.md` 的规格自主编码、自测、commit、push、开 PR。 |
| **真人（项目所有者）** | 戴 Quest 实测视频效果，给立体感/畸变/致晕反馈；决定方向；合并 PR。 |

### 工作循环
1. 主脑在 `CLINE_TASK_BOARD.md` 写下一个任务的**详细规格**（文件清单、接口签名、验收标准）。
2. 真人在 VS Code 对 Cline 说「读取 CLINE_TASK_BOARD.md，执行 XX 任务」。
3. Cline 自主完成 → push → 开 PR。
4. 主脑 review PR + 看 CI，绿了通知真人合并。
5. 真人合并；涉及视频效果的，戴 Quest 实测后反馈。

### 铁律（写在 `.clinerules`，血泪教训）
- **依赖任务必须串行**：若任务 B 依赖 A 的代码（如共用 `db/`），B 必须**从 A 的分支或合并后的 main 建**，绝不并行从旧 main 建——否则会重复造模块、rebase 冲突。
- **每个任务开始前 `git checkout main && git pull`**，基于最新已合并代码。
- **新依赖必须加进 requirements.txt**（passlib、sqlalchemy 等漏加都导致过 CI 红）。
- **绝不 `--no-verify` 跳过 pre-commit**（除非主脑确认是 hook 环境假阳性）。

---

## 六、工程规范（CI / PR）

- **所有改动走 PR**，不直接 push `main`（main 受保护）。
- **CI 是权威把关**（`.github/workflows/ci.yml`）：
  - `ruff check` + `ruff format --check`（lint 与格式）
  - `pytest tests/ -q --ignore=tests/e2e`（单元测试）
- **本地 pre-commit** 只做 ruff（快、可靠）；**pytest 不放 pre-push**（环境隔离会假阳性，交给 CI）。
- ruff 版本必须与 CI 一致（当前 `v0.15.18`），否则本地过、CI 红。

---

## 七、当前进展与路线图

### 已完成（已合并 main）
- ✅ 基础管线：SBS 检测、方向矩阵、深度、立体、等距投影、VR 元数据、超分、流式管线
- ✅ FastAPI REST API + 任务存储 + 配额 + 存储
- ✅ **修正 VR180 上下颠倒 + fps 继承 + FOV**（基于 Quest 实测）
- ✅ **Prompt 包装层** `pipeline/prompt_builder.py`（VR180 友好 prompt）
- ✅ **T1 数据库持久化**（SQLAlchemy 2.0 + SQLite + Alembic）
- ✅ **T2 API Key 认证**（PR #8 已提交，CI test 排查中）
- ✅ **T3 视频生成抽象层**（Kling/Seedance/Veo provider + 工厂）
- ✅ CI 健康化（移除假依赖、ruff 全绿、pre-commit、串行纪律）
- ✅ **研究原型**：多假设翻转矩阵、时序 AI 外绘、超分基准测试
- ✅ **Housekeeping**：清理 stale 分支、pycache、DS_Store

### 正在进行
- 🔄 **Hermes 通知 Agent**：本地 VR180 制作完成后，自动通过飞书推送通知（含视频信息、状态），实现快速测试反馈闭环
- 🔄 用 Google Veo/Gemini 生成的新 FPV 素材跑 pipeline，Quest 实测

### 待办（路线图）
- ⏸️ **T2 Auth PR #8**：CI test 失败，等待 GitHub Actions 日志排查后修复
- 🔜 修正预存测试失败（mapper shape assertion、spherical_injector import）
- 🔜 **Phase Q 画质提升（需远程 GPU）**：DepthCrafter + StereoCrafter + AI 外绘 + 8K 超分
- 🔜 **Phase C 前端工作流平台**（类 ComfyUI）全流程 UI
- 🔜 商业化：Convert / Generate / Studio 三层产品

---

## 八、关键文档索引

| 文档 | 内容 |
|------|------|
| [CLINE_TASK_BOARD.md](CLINE_TASK_BOARD.md) | ★ 当前任务看板，Cline 执行依据 |
| [.clinerules](.clinerules) | ★ Cline 自主开发协议与工程纪律 |
| [docs/STRATEGY_AI_VR180.md](docs/STRATEGY_AI_VR180.md) | AI→VR180 技术路线调研（为何走转换路线） |
| [docs/PROMPT_GUIDE_VR180.md](docs/PROMPT_GUIDE_VR180.md) | VR180 友好 Prompt 设计 + 可直接用的模板 |
| [docs/COMPETITOR_AND_BUSINESS.md](docs/COMPETITOR_AND_BUSINESS.md) | buildvr.ai 竞品逆向 + 商业模式规划 |
| [docs/architecture.md](docs/architecture.md) | 系统架构 |
| [docs/PRD-v2-vr180-studio.md](docs/PRD-v2-vr180-studio.md) | VR180 Studio 产品需求文档 v2 |
| [docs/session-summary-and-dev-guide.md](docs/session-summary-and-dev-guide.md) | 开发会话总结与指引 |
| [docs/archive/OVERNIGHT_RD_REPORT.md](docs/archive/OVERNIGHT_RD_REPORT.md) | 通宵研发报告（存档） |

---

## 九、给「接手的 Claude Code」的快速指引

如果你是另一台设备上的 Claude Code，按以下步骤上手：

1. **读三个文件**：本 `README.md` → `CLINE_TASK_BOARD.md`（当前在做什么）→ `.clinerules`（协作纪律）。
2. **明确你的角色**：你是**主脑**——分析、设计任务、审查 PR；具体编码交给 VS Code 的 Cline。
3. **看当前状态**：`git log --oneline -8`、`gh pr list`（看开放 PR）、`gh pr checks <n>`（看 CI）。
4. **派活方式**：在 `CLINE_TASK_BOARD.md` 写详细任务规格，再让真人在 Cline 里启动。
5. **审查 PR**：拉失败日志 `gh run view <run-id> --log-failed`；常见坑——漏加依赖、分支 base 错、缺 pytest-asyncio 配置。
6. **红线**：不直接 push main（走 PR）；不破坏现有测试；不改 `pipeline/` 现有转换逻辑除非任务要求。
7. **视频效果验收靠真人 Quest 实测**，不要自己判定「立体感够不够」。

---

## License
MIT — 见 [LICENSE](LICENSE)。
