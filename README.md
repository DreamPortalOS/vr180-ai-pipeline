# VR180 / Fulldome AI Pipeline

> 把普通 2D 视频（含 AI 生成的 FPV 素材）转换成**沉浸式视频**。当前聚焦**转换工作流与画面清晰度**。

本文件是项目总入口。详细规划见 [docs/SOLUTION_ARCHITECTURE.md](docs/SOLUTION_ARCHITECTURE.md) 与 [docs/ROADMAP.md](docs/ROADMAP.md)；当前任务见 [CLINE_TASK_BOARD.md](CLINE_TASK_BOARD.md)。

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
├── CLINE_TASK_BOARD.md          # ★ 任务看板（lead 派活、cline 执行）
├── .clinerules                  # ★ cline 自主开发协议
├── pipeline/                    # 核心转换管线
│   ├── depth_estimator.py       #   深度估计（路线2）
│   ├── stereo_renderer.py       #   立体视差渲染（路线2）
│   ├── equirectangular_mapper.py#   等距投影（路线2）
│   ├── spherical_injector.py    #   sv3d/st3d 注入（spatialmedia）
│   ├── vr_metadata.py / spatial_converter.py / streaming_pipeline.py / upscaler.py
│   ├── prompt_builder.py        #   VR180 友好 prompt 包装
│   └── (待建) fulldome_mapper.py #   路线1 球幕渲染器（看板 R-5）
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

### 提升清晰度（强烈推荐）
先用 SeedVR2 把源片升采样到 ~2K–4K，再跑上面的转换。部署见 **[docs/SEEDVR2_SETUP.md](docs/SEEDVR2_SETUP.md)**（Windows/4070S 走 ComfyUI）。

### 测试
```bash
# Windows PowerShell:  $env:PYTHONPATH="."   |  macOS/Linux:  export PYTHONPATH=.
pytest -q
```

---

## 协作模式
**lead（Claude Code）** 分析/规划/审查/QA + 写任务规格到 `CLINE_TASK_BOARD.md`；**cline** 按看板自主编码、自测、开 PR；**项目所有者** 实测效果、定方向、合并 PR。lead 的 git 操作走独立 `git worktree`，避免与 cline 抢工作区。详见 [.clinerules](.clinerules)。

## 当前进展
- ✅ 转换管线（深度/立体/等距/VR 元数据）跑通；VR180 输出格式正确（方形每眼 + `sv3d`/`st3d`）。
- ✅ 仓库精简：平台层归档，主线聚焦转换。
- ▶ **路线1 球幕渲染器（R-5）** 进行中；**SeedVR2 源片超分** 部署中（清晰度关键）。
- 🔜 路线2 立体画质：DepthCrafter + StereoCrafter（治重影/致晕，需 GPU）。

## 文档索引
| 文档 | 内容 |
|------|------|
| [CLINE_TASK_BOARD.md](CLINE_TASK_BOARD.md) | ★ 当前任务看板 |
| [docs/SOLUTION_ARCHITECTURE.md](docs/SOLUTION_ARCHITECTURE.md) | ★ 两路线系统方案 |
| [docs/ROADMAP.md](docs/ROADMAP.md) | 执行路线图（里程碑） |
| [docs/SEEDVR2_SETUP.md](docs/SEEDVR2_SETUP.md) | SeedVR2 在 4070S 上的部署 |
| [docs/COMPETITOR_AND_BUSINESS.md](docs/COMPETITOR_AND_BUSINESS.md) | buildvr.ai 竞品逆向（技术+商业） |
| [docs/STRATEGY_AI_VR180.md](docs/STRATEGY_AI_VR180.md) · [docs/PROMPT_GUIDE_VR180.md](docs/PROMPT_GUIDE_VR180.md) | 技术路线 / Prompt 指南 |
| [.clinerules](.clinerules) | cline 自主开发协议 |

## License
MIT — 见 [LICENSE](LICENSE)。
