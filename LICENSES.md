# 许可证清单（LICENSES）

本文件汇总**本仓自身**、`third_party/` 三个项目、以及 `models/` 下各权重的许可证与**准确出处**
（文件路径 + 行号），供逐条复核。

> **本文件只转述许可证原文与可验证事实，不构成法律意见。**
> 任何商用/生产决策请以各许可证原文为准，必要时咨询律师。

> `third_party/` 与 `models/` **不在 git 仓库内**（`.gitignore:72` 与 `.gitignore:63` 已忽略），
> 是各机器本地 clone / 下载的副本。下表行号以 **2026-09-15 本机检出的副本**为准；
> 上游更新后请用文末「如何复核」一节重新核对。

---

## 一览表

| 组件 | 许可证 | 商用 / 生产使用 | 出处（路径:行号） |
|---|---|---|---|
| **本仓 vr180-ai-pipeline** | MIT | 许可证原文未限制用途 | `LICENSE:1` |
| `third_party/DepthCrafter/`（推理代码） | Tencent 自定义 | **原文写明「不得商用或生产使用」** | `third_party/DepthCrafter/LICENSE:10` |
| `third_party/StereoCrafter/`（推理代码） | Tencent 自定义 | **原文写明「不得商用或生产使用」** | `third_party/StereoCrafter/License-Code.txt:8` |
| `third_party/seedvr2_videoupscaler/`（代码） | Apache License 2.0 | 许可证原文未限制用途 | `third_party/seedvr2_videoupscaler/LICENSE:1-2`、`:189-191` |
| `models/DepthCrafter/`（权重） | Tencent 自定义 + SVD 非商业社区许可 | **原文写明「不得商用或生产使用」** | `models/DepthCrafter/LICENSE:11`、`models/DepthCrafter/NOTICE:16` |
| `models/StereoCrafter/`（权重） | Tencent 自定义 + SVD 非商业社区许可 | **原文写明「不得商用或生产使用」** | `models/StereoCrafter/LICENSE:9`、`models/StereoCrafter/NOTICE:14` |
| `models/svd-img2vid-xt-1-1/`（权重） | 本地**无**许可证文件，见下方说明 | 需查上游 | 本仓无；上游见 `models/StereoCrafter/NOTICE:21` 指向的 HF 仓库 |
| `models/SEEDVR2/`（权重） | 本地**无**许可证文件，见下方说明 | 需查上游 | 下载地址见 `docs/SEEDVR2_SETUP.md:156-159` |
| Depth-Anything-V2（运行时从 HF 拉取） | 本仓不分发权重，见下方说明 | 需查上游 | 模型 id 见 `pipeline/depth_estimator.py:34-36` |

---

## 逐条出处

### 1. 本仓自身 — MIT

`LICENSE:1`：

> MIT License

`LICENSE:3`：`Copyright (c) 2025 DreamPortalOS`。

### 2. `third_party/DepthCrafter/` — 仅限学术/研究/教育

`third_party/DepthCrafter/LICENSE:10`：

> You agree to use the DepthCrafter only for academic, research and education purposes（后接禁止商用或生产使用的措辞）

同文件 `:14` 界定 “Software” 的范围为 DepthCrafter 推理代码与权重；
`:3` 的版权归属为 Tencent（`:1` 说明原 “THL A29 Limited” 主体已注销）。
同文件 `:21-34` 另列其依赖的 Stability AI 代码为 MIT。

### 3. `third_party/StereoCrafter/` — 仅限学术/研究/教育

`third_party/StereoCrafter/License-Code.txt:8`：

> You agree to use the StereoCrafter only for academic, research and education purposes（后接禁止商用或生产使用的措辞）

同文件 `:1` 版权归属 THL A29 Limited（Tencent）；`:12` 界定 “Software” 范围；
`:19-32` 另列 Stability AI 代码为 MIT。

### 4. `third_party/seedvr2_videoupscaler/` — Apache License 2.0

`third_party/seedvr2_videoupscaler/LICENSE:1-2`：

> Apache License / Version 2.0, January 2004

`:189-191` 为版权与适用声明（`Copyright 2025 seed`，`Licensed under the Apache License, Version 2.0`）。
上游 README 亦确认：`third_party/seedvr2_videoupscaler/README.md:1053`。
Apache-2.0 的再分发/通知/归属等义务以许可证原文第 4 节（`:89` 起）为准。

### 5. `models/DepthCrafter/`（权重）— 双重限制

- `models/DepthCrafter/LICENSE:11`：与推理代码相同的「仅限学术、研究、教育」措辞。
- `models/DepthCrafter/LICENSE:3`：声明本模型是在 **Stable Video Diffusion (SVD)** 协助下微调的，
  受 **SVD Non-Commercial Community License** 约束。
- `models/DepthCrafter/NOTICE:16`、`:21-23`：列出 SVD 模型及其原始地址
  （`https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt`）；
  `NOTICE:26` 起为 SVD 非商业社区许可证全文。

### 6. `models/StereoCrafter/`（权重）— 双重限制

- `models/StereoCrafter/LICENSE:9`：与推理代码相同的「仅限学术、研究、教育」措辞。
- `models/StereoCrafter/LICENSE:1`：同样声明受 **SVD 非商业社区许可**约束。
- `models/StereoCrafter/NOTICE:14`、`:19-21`：SVD 模型条目与原始地址；`NOTICE:24` 起为其全文。

### 7. `models/svd-img2vid-xt-1-1/`（权重）— 本地无许可证文件

该目录当前只有模型文件与子目录（`feature_extractor/`、`image_encoder/`、`model_index.json`、
`scheduler/`、`unet/`、`vae/`），**没有 LICENSE / NOTICE 文件**，因此本仓无法给出带行号的原文出处。
与之相关的可验证事实：`models/StereoCrafter/NOTICE:20` 写明
「Stable Video Diffusion is licensed under the Stable Video Diffusion Research License」，
`NOTICE:21` 给出上游地址。**使用前请到上游仓库核对该权重自带的许可证。**

### 8. `models/SEEDVR2/`（权重）— 本地无许可证文件

该目录只有 `seedvr2_ema_3b_fp8_e4m3fn.safetensors` 与 `ema_vae_fp16.safetensors`，
**没有 LICENSE 文件**。下载地址见 `docs/SEEDVR2_SETUP.md:156-159`（HF 仓库 `numz/SeedVR2_comfyUI`）。
注意：`third_party/seedvr2_videoupscaler/LICENSE` 的 Apache-2.0 覆盖的是**该仓库的代码**
（`third_party/seedvr2_videoupscaler/README.md:1053`：“The code in this repository is released under
the Apache 2.0 license”），**权重的许可证需另行到上游核对**。

### 9. Depth-Anything-V2（VR180 默认深度模型）— 本仓不分发权重

`pipeline/depth_estimator.py:34-36` 通过 `transformers` 在运行时拉取
`depth-anything/Depth-Anything-V2-{Small,Base,Large}-hf`。本仓不包含其权重与许可证文件，
故此处不列行号出处；**请到对应 HF 仓库核对**。

---

## 路线分界：球幕单目链路不经过 DepthCrafter / StereoCrafter

这条分界决定了「哪条链路会碰到上述非商用许可证」（相关讨论见 #368）：

- **路线 1 · Fulldome（单目球幕）**：`scripts/run_pipeline.py:3489-3506` 在
  `args.projection == "fulldome"` 时直接调用 `FulldomeMapper` 并 `return`，
  日志原文为「bypassing depth/stereo/equirect/metadata stages」（`:3493`）。
  `pipeline/fulldome_mapper.py:1-3` 亦写明该渲染器是纯 ffmpeg `v360` 鱼眼投影、
  “No depth/stereo/spherical metadata”。
  → **该链路不加载 DepthCrafter / StereoCrafter 的代码或权重。**
- **路线 2 · VR180（双眼立体）**：`scripts/run_pipeline.py:63` / `:84` 引入
  `DepthCrafterEstimator` / `StereoCrafterRenderer`，由 `--depth-model depthcrafter`
  与 `--stereo-backend stereocrafter`（`:820`）显式选用；未选用时走
  Depth-Anything-V2（第 9 条）。
- **SeedVR2 超分**（`docs/SEEDVR2_SETUP.md`）是两条路线共用的前置步骤，
  其**代码**为 Apache-2.0（第 4 条），**权重**许可证见第 8 条。

---

## 如何复核

上游更新后行号可能漂移。逐条复核（在各文件所在机器执行）：

```bash
grep -n "academic, research and education" third_party/DepthCrafter/LICENSE
grep -n "academic, research and education" third_party/StereoCrafter/License-Code.txt
grep -n "Apache License" third_party/seedvr2_videoupscaler/LICENSE
grep -n "academic, research and education" models/DepthCrafter/LICENSE models/StereoCrafter/LICENSE
grep -n "Non-Commercial Community License" models/DepthCrafter/NOTICE models/StereoCrafter/NOTICE
```
