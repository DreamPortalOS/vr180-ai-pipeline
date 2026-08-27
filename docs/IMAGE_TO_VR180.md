# 单图 → VR180 使用指南（阶段 G）

> 一张图（AI 生成图 / 照片 / 概念画）→ Quest 可播的 VR180 立体视频，一条命令搞定。
>
> 编排器：[`scripts/image_to_vr180.py`](../scripts/image_to_vr180.py)（issue #56 / G-3）。
> 设计背景见 [docs/PRD-image-to-vr180.md](PRD-image-to-vr180.md)。

## 整体流程

```
单图 ──prepare──► 归一化图 ──generate──► 生成视频 ──streamcheck──► [upscale] ──convert──► VR180 ──qa──► 交付
       (image_prep)              (I2V provider)        (ffprobe)    (SeedVR2 可选)  (run_pipeline)  (vr180_qa)
```

六个阶段，每跑完一个就写进 job manifest（断点文件），下游失败时 `--resume-from` 可跳过已完成的阶段——
**生成一次很贵，绝不能因为转换阶段崩了而重新生成**。

| 阶段 | 做什么 | 代码位置 |
|------|--------|----------|
| `prepare` | 校验 + EXIF 矫正 + 缩放/letterbox 到 16:9 @1280 宽 | `pipeline/image_prep.py` |
| `generate` | 调 I2V provider 生成视频（本地图片 base64 编码上传） | `integrations/<provider>` |
| `streamcheck` | ffprobe 读流信息，分辨率/fps/时长异常即 fail-fast | `scripts/vr180_qa.py` |
| `upscale` | 可选 SeedVR2 视频超分（`--upscale seedvr2`） | `pipeline/video_upscaler.py` |
| `convert` | VR180 转换：深度 → 立体 → 等距投影 → 元数据（`--quality` 预设） | `scripts/run_pipeline.py` |
| `qa` | `vr180_qa` 机器验收；不通过则**退出码 3**，不交付 | `scripts/vr180_qa.py` |

---

## 快速试跑（mock 全链路，无需 API key）

mock provider 用 ffmpeg `testsrc2` 合成一段真实可播的 mp4——不联网、不加载模型、不要 key。
这是在干净仓库里验证编排器接线的标准方式。

### 前置
- ffmpeg 在 `PATH`（Windows：`choco install ffmpeg`；macOS：`brew install ffmpeg`）。
- `convert` 阶段默认走 `run_pipeline` 全链路，依赖 **Depth-Anything-V2 深度模型**
  （`transformers` + 需下载权重，见下方「配真实后端」）。**干净环境（未装 transformers）跑不到 convert 就会报
  `ModuleNotFoundError: No module named 'transformers'`**——这是预期行为，不是 bug。
  若只想验证 mock 生成 + streamcheck + 断点续跑（不跑深度转换），用下面两种方式之一。

### 方式 A：直接跑命令（已装 transformers + 深度模型的环境）

```bash
# Windows PowerShell:  $env:PYTHONPATH="."   |  macOS/Linux:  export PYTHONPATH=.
python scripts/image_to_vr180.py \
  --image cat.png \
  --provider mock \
  --quality preview \
  --workdir build/cat_vr180 \
  --manifest build/cat_job.json
```

产物落在 `--workdir`：

```
build/cat_vr180/
  cat_prep.png          # prepare 归一化图
  cat_generated.mp4     # mock 生成的视频（testsrc2）
  cat_vr180.mp4         # ★ 最终 VR180 输出
build/cat_job.json      # 断点 manifest
```

末尾会打印 `📦 VR180 output: build/cat_vr180/cat_vr180.mp4`，退出码 `0` 表示 QA 通过。

> `--quality preview` 用 1920²/眼，跑得最快，适合试流程。交付质量用 `standard`(2880²/眼) 或 `high`(3840²/眼)。

### 方式 B：在干净仓库里验证 mock 接线（CI 同款，无需任何模型/key）

仓库的单元测试已用注入的伪 converter 跑通了「真 prepare + 真 mock 生成 + 真 streamcheck + 真 QA」全链路——
CI（ubuntu、CPU-only、无模型无 key）就用这套。这是干净环境里**唯一**能端到端跑通的 mock 路径：

```bash
# Windows PowerShell:  $env:PYTHONPATH="."   |  macOS/Linux:  export PYTHONPATH=.
pytest tests/test_image_to_vr180.py -q                  # 快速层（伪后端，not slow）
pytest tests/test_image_to_vr180.py -q -m "slow"          # mock 端到端（真 ffmpeg + 真 QA）
```

`-m slow` 的 `TestMockProviderEndToEnd` 会真渲染一段 SBS mp4、注入 sv3d/st3d 盒子、
过真实 `vr180_qa` 扫描——等价于「mock 生成 → 真转换 → QA 绿」。

---

## 断点续跑

每次运行带 `--manifest job.json`，每完成一个阶段就写入（含产物文件 SHA256）。
下游失败后用 `--resume-from` 续跑：已完成阶段会被**哈希校验后跳过**（图片被换或产物被改都会被抓住并拒绝）。

```bash
# 第一次：跑到 convert 崩了（比如显存不够）
python scripts/image_to_vr180.py --image cat.png --provider seedance --manifest build/cat_job.json

# 修好环境后续跑：generate 不会重跑（省钱），从 convert 继续
python scripts/image_to_vr180.py --image cat.png --provider seedance --resume-from build/cat_job.json
```

- `--manifest PATH`：新跑时写断点文件。
- `--resume-from MANIFEST`：从已有断点续跑；省略 `--manifest` 则只读不写。
- 续跑要求**同一张输入图**（按源文件哈希校验）。

---

## 配真实 I2V 后端

三个云 provider 各自的鉴权环境变量：

| Provider | `--provider` | 环境变量 | 端点 / 控制台 |
|----------|--------------|----------|--------------|
| Seedance（火山方舟） | `seedance` | `ARK_API_KEY` | `https://ark.cn-beijing.volces.com/api/v3` · [方舟控制台](https://console.volcengine.com/ark/) |
| Kling（可灵） | `kling` | `KLING_API_KEY` | `https://api.klingai.com` · [docs.klingai.com](https://docs.klingai.com/) |
| Veo（Vertex AI） | `veo` | `VEO_API_KEY`（+ `GCP_PROJECT_ID`） | `https://us-central1-aiplatform.googleapis.com/v1` · [GCP 凭据](https://console.cloud.google.com/apis/credentials) |

```bash
# 示例：Seedance 真实生成
export ARK_API_KEY="<your-volcengine-ark-key>"
python scripts/image_to_vr180.py \
  --image cat.png \
  --provider seedance \
  --prompt "镜头缓慢推进，画面静止" \
  --duration 5 \
  --quality standard \
  --manifest build/cat_job.json
```

**Seedance 默认模型** `doubao-seedance-2-0-fast-260128`（quota 受限，故选低成本档）；
分辨率默认 `480p`、比例 `adaptive`。模型未在方舟开通时会报 `ModelNotOpen`，
按提示去方舟控制台开通对应模型。

> ⚠️ 真实生成前请先在 [docs/VIDEOGEN_SETUP.md](VIDEOGEN_SETUP.md) 了解各 provider 的额度与计费。
> 路线 B（本地 SVD，`LocalSVDProvider`）见 PRD §3.1，为 P2 未实现。

### 转换阶段的后端依赖（必读）

`convert` 阶段调用 `run_pipeline` 的完整深度→立体→等距→元数据链路，**这些不是 mock**：

- **Depth-Anything-V2**（`pipeline/depth_estimator.py`）：需 `transformers`（已在
  `requirements.txt`）+ 首次运行下载模型权重（~25M/97M/335M 参数，按 `--model-size`）。
- **SeedVR2 超分**（`--upscale seedvr2`）：需 CUDA GPU + 已部署的 SeedVR2，见
  [docs/SEEDVR2_SETUP.md](SEEDVR2_SETUP.md)。
- **VR 元数据**（sv3d/st3d）：需 `spatialmedia`（`pip install "git+https://github.com/google/spatial-media.git#egg=spatialmedia"`）。
  未装时注入器回落到 ffmpeg udta（V1 XML），**V2 盒扫描器看不到，QA 会判为 plain 2D**。

干净仓库（未装这些）直接跑 `--provider mock` 会在 convert 报 `transformers` 缺失——
这是转换阶段的真实模型依赖，不是编排器问题。验证编排器本身用上面「方式 B」的测试。

---

## 参数解释

| 参数 | 默认 | 说明 |
|------|------|------|
| `--image` / `-i` | （必填） | 输入图片路径；支持 jpg/png/webp。I2V 约束：每边 300–6000px、宽高比 0.4–2.5、≤30MB、扩展名 jpg/jpeg/png/webp（`validate_image_for_i2v`）。`http(s)` URL 直接透传给 provider。 |
| `--prompt` | `""` | **只描述运动**（镜头推进/环绕/静止），画面内容由图决定；留空=纯图驱动。运动幅度默认保守（大运动 → 深度噪声 → 重影）。 |
| `--provider` | `mock` | `mock` / `kling` / `seedance` / `veo`。 |
| `--duration` | `5` | 生成视频秒数。 |
| `--upscale` | `none` | `none` / `seedvr2`（SeedVR2 超分，需 CUDA）。 |
| `--quality` | `preview` | `preview`(1920²/眼) / `standard`(2880²/眼) / `high`(3840²/眼)。码率按像素面积自适应。 |
| `--manifest` | （无） | 断点 manifest 写入路径。 |
| `--resume-from` | （无） | 从已有 manifest 续跑，跳过已完成阶段（哈希校验）。 |
| `--workdir` | `<图目录>/<图名>_vr180` | 中间产物目录。 |

---

## 常见失败与排查

| 症状 | 原因 | 排查 |
|------|------|------|
| `ModuleNotFoundError: No module named 'transformers'`（convert 阶段） | 深度模型依赖未装 | `pip install -r requirements.txt`；convert 阶段非 mock，见「转换阶段的后端依赖」。干净环境验证编排器用 `pytest tests/test_image_to_vr180.py`。 |
| `Image ... unsupported format` / `size ... exceeds the 30 MB limit` / `aspect ratio ... outside` | 输入图不满足 I2V 约束 | 按报错调整图：jpg/png/webp、每边 300–6000px、宽高比 0.4–2.5、≤30MB。 |
| `Could not decode image ... (not a valid image)` | 损坏文件或非图片（含路径含特殊字符导致 cv2 读不到） | 确认文件可被图片查看器打开；Windows 下路径用正斜杠或绝对路径。 |
| `ffmpeg not found on PATH` | ffmpeg 未装 / 不在 PATH | 装 ffmpeg，或设 `FFMPEG_BINARY` 指向可执行文件。 |
| `Generated video stream check failed: video too small` / `too short` | provider 返回了退化视频 | mock 不会触发；真实 provider 返回异常流时 fail-fast，换图/重试/查 provider 状态。 |
| `QA failed ... refusing to deliver`（退出码 3） | 最终产物未通过 `vr180_qa`（缺 sv3d/st3d、非 SBS、分辨率不足） | 多半是 `spatialmedia` 未装 → 元数据注入回落 V1 → QA 判 plain 2D。装 spatialmedia 重跑 convert（`--resume-from` 可跳过 generate）。 |
| `Seedance task ... ModelNotOpen` | 模型未在方舟开通 | 去方舟控制台开通 `doubao-seedance-2-0-fast-260128`。 |
| `Cannot resume from ... hash` | 输入图被换 / 产物被改 | 续跑要求同一源图；若有意换了图，删掉 manifest 重跑。 |
| provider 轮询超时（`did not complete within 300s`） | 生成排队久 | 默认 5 分钟超时；重试即可，已完成的 generate 阶段可 `--resume-from` 跳过。 |

### 退出码
- `0` — 全链路通过，QA 绿。
- `3` — QA 失败（`EXIT_QA_FAILED`），不交付非 VR180 产物。
- `1` — 其他运行时错误（图校验、ffmpeg、provider 调用等）。

---

## 相关文档
- [docs/PRD-image-to-vr180.md](PRD-image-to-vr180.md) — 阶段 G 产品需求与任务卡拆解
- [docs/VIDEOGEN_SETUP.md](VIDEOGEN_SETUP.md) — 各 I2V provider 额度与计费
- [docs/SEEDVR2_SETUP.md](SEEDVR2_SETUP.md) — SeedVR2 超分部署（4070S / ComfyUI）
- [README.md](../README.md) — 项目总入口与快速上手
