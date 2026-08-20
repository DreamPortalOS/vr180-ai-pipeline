# VR180 / Fulldome AI Pipeline

> 把普通 2D 视频（含 AI 生成的 FPV 素材）转换成**沉浸式视频**。当前聚焦**转换工作流与画面清晰度**。

本文件是项目总入口。详细规划见 [docs/SOLUTION_ARCHITECTURE.md](docs/SOLUTION_ARCHITECTURE.md) 与 [docs/ROADMAP.md](docs/ROADMAP.md)；当前任务见 [GitHub Issues](https://github.com/DreamPortalOS/vr180-ai-pipeline/issues)（`stage:ready` = 待开发队列）。

---

## 两条交付路线（同一条共享管线，只在渲染层分叉）

| | **路线 1 · 球幕 / Fulldome**（进行中） | **路线 2 · VR180**（差异化，较难） |
|---|---|---|
| 观看 | 球幕投影，**无需眼镜** | VR 头显（Quest / Vision Pro） |
| 立体 | 单目（无重影、不致晕） | 真立体（双眼 SBS） |
| 投影 | domemaster 圆形鱼眼（方形 4K²） | SBS 等距投影 + `sv3d`/`st3d` |
| 难点 | 投影/分辨率 | + 立体融合、重影、致晕 |

共享前端：`ingest/生成 → SeedVR2 升采样 → 可插拔渲染器 → 编码`。

---

## 技术管线

```
2D 视频
  └─（可选）SeedVR2 源片超分  ← 清晰度关键，见 docs/SEEDVR2_SETUP.md
       ├── 路线1 Fulldome：ffmpeg v360 → 圆形鱼眼 domemaster（单目，无深度）
       └── 路线2 VR180：深度估计(Depth-Anything-V2) → 立体视差 → 等距投影 → sv3d/st3d 注入
            └→ 方形每眼 SBS（如 2880²/眼）H.264/H.265
```

清晰度的主因是**源片分辨率**（720p 直接映射会糊）。根治靠 SeedVR2 把源片升采样后再转换。

---

## 代码架构（平台层已归档）

```
vr180-ai-pipeline/
├── CLAUDE.md                    # ★ 自主 worker 行为规范（cockpit 派单执行）
├── pipeline/                    # 核心转换管线
│   ├── depth_estimator.py       #   深度估计（Depth-Anything-V2，路线2）
│   ├── depth_crafter.py         #   DepthCrafter 时序深度（可插拔后端）
│   ├── stereo_renderer.py       #   立体视差渲染（路线2）
│   ├── stereo_crafter.py        #   StereoCrafter 补遮挡（可插拔后端）
│   ├── equirectangular_mapper.py#   等距投影（含 map_sequence 批处理）
│   ├── fulldome_mapper.py       #   路线1 球幕渲染器（v360 鱼眼）
│   ├── outpainter.py            #   180° 边界外绘
│   ├── video_upscaler.py        #   SeedVR2 源片超分（CUDA）
│   ├── spherical_injector.py    #   sv3d/st3d 注入（spatialmedia）
│   ├── vr_metadata.py / spatial_converter.py / streaming_pipeline.py / upscaler.py
│   └── prompt_builder.py        #   VR180 友好 prompt 包装
├── integrations/                # 生成层（Kling / Seedance / Veo）
├── scripts/run_pipeline.py      #   CLI 跑完整管线
├── tests/                       # pytest（CI 把关）
├── docs/                        # 文档（见下方索引）
└── video/                       # 测试素材与输出（git 忽略）
```

> 平台层（`web/ db/ auth/ integrations/ notifications/ workers/` + 前端/配额）已**归档到分支 `archive/platform-layer`**，主线不含。需要时从该分支取回。

---

## 快速上手

### 环境
- Python 3.10+ · ffmpeg（Windows：`choco install ffmpeg` 或 `scoop install ffmpeg`；macOS：`brew install ffmpeg`）
- 路线1 球幕在任意机器可跑（纯 ffmpeg）。路线2 立体的深度估计 Mac(MPS)/CUDA 均可；高质量超分（SeedVR2）需 NVIDIA CUDA（RTX 3060/4070S 12GB 起）。

### 安装
```bash
git clone https://github.com/DreamPortalOS/vr180-ai-pipeline.git
cd vr180-ai-pipeline
python -m venv .venv
# Windows:  .venv\Scripts\activate     macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
pip install "git+https://github.com/google/spatial-media.git#egg=spatialmedia"  # 路线2 VR 元数据需要
```

### 路线 1 · 球幕 domemaster（最快、单目、无需眼镜）
```bash
ffmpeg -i video/输入.mp4 \
  -vf "v360=input=flat:output=fisheye:ih_fov=120:iv_fov=75:h_fov=180:v_fov=180:w=4096:h=4096" \
  -c:v libx265 -crf 18 -pix_fmt yuv420p video/输出_fulldome.mp4
```
（`fulldome_mapper.py` 会把它封装成 `--projection fulldome`，见看板 R-5。）

### 路线 2 · VR180 立体
```bash
# Windows PowerShell:  $env:PYTHONPATH="."   |  macOS/Linux:  export PYTHONPATH=.
python scripts/run_pipeline.py \
  --input video/输入.mp4 --output video/输出_vr180.mp4 \
  --src-hfov 150 --max-disparity 0.02 --model-size small --codec h265 --crf 18
```
输出为方形每眼 SBS（如 5760×2880），传 Quest 用 Skybox/DeoVR 选「180° 3D (SBS)」。
质量档位 `--quality {preview,standard,high}`（默认 `standard` = 2880²/眼流式，`high` = 3840²/眼流式，`preview` = 1920²/眼快速迭代）；码率随像素面积自适应缩放。

### 提升清晰度（强烈推荐）
先用 SeedVR2 把源片升采样到 ~2K–4K，再跑上面的转换。部署见 **[docs/SEEDVR2_SETUP.md](docs/SEEDVR2_SETUP.md)**（Windows/4070S 走 ComfyUI）。

### 测试
```bash
# Windows PowerShell:  $env:PYTHONPATH="."   |  macOS/Linux:  export PYTHONPATH=.
pytest -q
```

---

## 协作模式（2026-08 起：cockpit 派单）
**lead（Claude）** 作为 PM/架构师写详细任务卡到 **GitHub Issues**（打 `stage:ready` + `model:cheap` 标签）；
本地 **cockpit lane** 轮询领卡，派 worker（当前引擎 **kimi-k3**，网关探活失败自动回落 deepseek）在隔离
worktree 里实现、自测、开 PR；自动评审（CI 绿 + AI 评审）squash 合并；**项目所有者** 通过仓根
`WORKLOG.md`（本地异步工作台）实测打分、定方向。worker 行为规范见 [CLAUDE.md](CLAUDE.md)。

## 当前进展（2026-08-21）
- ✅ 转换管线全链路跑通：SeedVR2 超分 → 深度/立体 → 等距/鱼眼投影 → `sv3d`/`st3d` 注入（PR #1–#32 全部合并）。
- ✅ 路线1 球幕渲染器（fulldome）、180° 外绘、批处理等距投影（~10×）、生成层（Kling/Seedance/Veo）落地。
- ▶ **画质冲刺**：Quest 实测暴露两大主因 —— 每眼 1920px 太低（→ 流式 3840/眼）+ 视差 0.02 太弱且深度噪声大
  （→ 参数扫描 + Mac 上 DepthCrafter/StereoCrafter）。当前任务队列见 GitHub Issues。
- 🔜 跨机工作流：Windows（SeedVR2/CUDA）↔ Mac M2 Max（重深度/立体模型，MPS）分段接力。

## 文档索引
| 文档 | 内容 |
|------|------|
| [CLAUDE.md](CLAUDE.md) | ★ 自主 worker 行为规范 |
| [docs/SOLUTION_ARCHITECTURE.md](docs/SOLUTION_ARCHITECTURE.md) | ★ 两路线系统方案 |
| [docs/ROADMAP.md](docs/ROADMAP.md) | 执行路线图（里程碑） |
| [docs/SEEDVR2_SETUP.md](docs/SEEDVR2_SETUP.md) | SeedVR2 在 4070S 上的部署 |
| [docs/COMPETITOR_AND_BUSINESS.md](docs/COMPETITOR_AND_BUSINESS.md) | buildvr.ai 竞品逆向（技术+商业） |
| [docs/STRATEGY_AI_VR180.md](docs/STRATEGY_AI_VR180.md) · [docs/PROMPT_GUIDE_VR180.md](docs/PROMPT_GUIDE_VR180.md) | 技术路线 / Prompt 指南 |
| [docs/FULLDOME_USAGE.md](docs/FULLDOME_USAGE.md) | fulldome 输出定位与预览（⚠️ 非头显格式） |
| [docs/archive/](docs/archive/) | 历史过程文件（cline 看板/协议等） |

## License
MIT — 见 [LICENSE](LICENSE)。
