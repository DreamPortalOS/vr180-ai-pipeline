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

### 单图 → VR180（一张图直接生成 VR180）

用 `scripts/image_to_vr180.py` 一条命令把单图变成 Quest 可播的 VR180（I2V 生成 → [超分] → 转换 → QA）：

```bash
# Windows PowerShell:  $env:PYTHONPATH="."   |  macOS/Linux:  export PYTHONPATH=.
python scripts/image_to_vr180.py --image cat.png --provider mock --quality preview --workdir build/cat_vr180
pytest tests/test_image_to_vr180.py -q          # 干净仓库（无模型/key）验证 mock 全链路
```

> `--provider mock` 用 ffmpeg 合成视频，不联网不要 key。配真实生成用 `--provider seedance|kling|veo`
> （见 [docs/IMAGE_TO_VR180.md](docs/IMAGE_TO_VR180.md)）。`convert` 阶段依赖 Depth-Anything-V2
> 深度模型（`transformers`），干净环境未装时用上面的 `pytest` 验证编排器接线。

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

## 当前进展（2026-08-21 晚）
- ✅ **画质冲刺 V-1…V-9 全部交付**（kimi-k3 编码 + lead 实机验收）：质量预设（3840²/眼流式默认路径）、
  流式 SBS 布局/元数据注入/编码器回退三连修、立体参数扫描工具、VR180 QA 校验器、跨机 job manifest。
- ✅ 3840²/眼 · 视差 0.06 · hfov150 样片已交付 Quest 真机（等 owner 打分）。
- ▶ **阶段 G：单图 → 视频 → VR180 工作流**（见 [docs/PRD-image-to-vr180.md](docs/PRD-image-to-vr180.md)）：
  provider 层加 image-to-video 入口 → 图片预处理 → 一键编排 `image_to_vr180`。真实生成待 owner 提供 API key。
- 🔜 Mac M2 Max 上 DepthCrafter/StereoCrafter（治重影的根治路径，owner 部署中）。

## 文档索引
| 文档 | 内容 |
|------|------|
| [CLAUDE.md](CLAUDE.md) | ★ 自主 worker 行为规范 |
| [docs/DEV_PROCESS.md](docs/DEV_PROCESS.md) | ★ 开发协作机制（owner/lead/kimi-k3 分工与验收流程） |
| [docs/PRD-image-to-vr180.md](docs/PRD-image-to-vr180.md) | ★ 阶段 G PRD：单图→视频→VR180 |
| [docs/SOLUTION_ARCHITECTURE.md](docs/SOLUTION_ARCHITECTURE.md) | ★ 两路线系统方案 |
| [docs/ROADMAP.md](docs/ROADMAP.md) | 执行路线图（里程碑） |
| [docs/SEEDVR2_SETUP.md](docs/SEEDVR2_SETUP.md) | SeedVR2 在 4070S 上的部署 |
| [docs/COMPETITOR_AND_BUSINESS.md](docs/COMPETITOR_AND_BUSINESS.md) | buildvr.ai 竞品逆向（技术+商业） |
| [docs/STRATEGY_AI_VR180.md](docs/STRATEGY_AI_VR180.md) · [docs/PROMPT_GUIDE_VR180.md](docs/PROMPT_GUIDE_VR180.md) | 技术路线 / Prompt 指南 |
| [docs/IMAGE_TO_VR180.md](docs/IMAGE_TO_VR180.md) | ★ 单图 → VR180 使用指南（mock 试跑 / 真实 key / 断点续跑 / 排查） |
| [docs/FULLDOME_USAGE.md](docs/FULLDOME_USAGE.md) | fulldome 输出定位与预览（⚠️ 非头显格式） |
| [docs/archive/](docs/archive/) | 历史过程文件（cline 看板/协议等） |

## License
MIT — 见 [LICENSE](LICENSE)。
