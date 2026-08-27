# PRD — 单图 → 视频 → VR180 工作流（阶段 G，2026-08-21）

_Owner 拍板的下一阶段目标；lead（Claude）规划，kimi-k3 worker 实现，验收标准见各任务卡。_

## 1. 产品目标

用户给**一张图**（AI 生成图 / 照片 / 概念画），产出一条**可直接放进 Quest 的 VR180 立体视频**：

```
单图 ──► I2V 生成（image-to-video, 5–10s） ──► [SeedVR2 超分] ──► VR180 转换 ──► vr180_qa ──► 交付
        （云 API / 本地模型，可插拔）          （已有，可选）      （已有全链路）    （已有）
```

只有第一段（图→视频）是新建；其余全部复用现有管线（`--quality` 预设、流式渲染、
sv3d/st3d 注入、跨机 job manifest、QA 校验器）。

## 2. 现状盘点（复用什么）

| 能力 | 位置 | 状态 |
|---|---|---|
| 生成层 provider 抽象（Kling/Seedance/Veo，text→video） | `integrations/base.py` + factory | ✅ 已合并（#32），**缺 image 入口** |
| 生成 CLI | `scripts/generate.py` | ✅ 可用（text→video） |
| SeedVR2 源片超分 | `pipeline/video_upscaler.py` | ✅ 本机已部署（CUDA） |
| 2D→VR180 全链路（质量预设/流式/元数据/回退） | `scripts/run_pipeline.py` | ✅ V-1/V-7/V-9 实机验收通过 |
| 跨机分段 + 断点续跑 | `pipeline/job_manifest.py`（V-3） | ✅ 已合并 |
| 输出机器验收 | `scripts/vr180_qa.py`（V-5/V-8） | ✅ 实机验收通过 |

## 3. 技术方案

### 3.1 I2V 后端（可插拔，两条路）
- **路 A（默认，先做）：云 API**。Kling / Seedance(即梦) / Veo 均有 image-to-video 端点。
  扩展现有 `VideoGenProvider`：新增抽象方法 `generate_from_image(image, prompt, duration, …)`，
  三个 provider 各自实现（HTTP 细节以各家官方 API 文档为准；**图片以 base64 或 URL 上传**，
  沿用各 provider 现有的鉴权/轮询/下载骨架）。**CI 一律 mock**（repo 铁律）。
  ⚠️ 真实调用需 owner 提供任一家的 API key（WORKLOG §3 待决策）——工作流先以 mock 全线打通。
- **路 B（P2，后做）：本地模型**。SVD img2vid（diffusers）：12GB 上仅低分辨率可行，
  M2 Max（MPS）可跑全分辨率——封装成与云 provider 同接口的 `LocalSVDProvider`，
  后端可插拔 + mock，真实推理走 Mac（复用 V-3 跨机机制）。

### 3.2 图片预处理（生成质量的地基）
I2V 对输入图的分辨率/比例敏感。新增 `pipeline/image_prep.py`：
- 校验：最小分辨率告警（<1024 宽 warn）；损坏/非图片报错。
- 归一化：letterbox/crop 到目标比例（默认 16:9，provider 可覆盖）；EXIF 旋转矫正。
- 可选放大：cv2 Lanczos 到 provider 推荐输入尺寸（不引入新模型依赖）。

### 3.3 一键工作流编排
新增 `scripts/image_to_vr180.py`（复用，不复制逻辑）：
```
--image cat.png [--prompt "..."] [--provider kling|seedance|veo|mock] [--duration 5]
[--upscale seedvr2|none] [--quality standard|high] [--manifest job.json]
```
流程：image_prep → provider.generate_from_image →（可选 SeedVR2）→ run_pipeline 全链
→ vr180_qa 自动验收（❌ 则非零退出）→ 打印产物路径 + QA 摘要。
每步写入 job manifest（V-3 机制）支持 `--resume-from` 断点续跑（生成一次很贵，绝不能因下游失败重新生成）。

### 3.4 I2V 专用注意点（写进实现）
- **生成参数偏置**：i2v 场景 prompt 只描述**运动**（镜头推进/环绕/静止），画面内容由图决定；
  运动幅度默认保守（大运动 → 深度估计噪声 → 重影）。
- **首帧一致性**：转换阶段的深度/立体参数沿用现有默认；生成的视频先过 `vr180_qa` 的
  stream-info 检查（分辨率/fps 达标才继续），不合格 fail-fast。

## 4. 任务卡拆解（派发顺序）

| 卡 | 内容 | 优先级 | 依赖 |
|---|---|---|---|
| **G-1** | provider 层 image-to-video：`generate_from_image` 抽象 + 三 provider 实现 + mock + 测试 | P0 | 无 |
| **G-2** | `pipeline/image_prep.py` 图片预处理 + 测试 | P0 | 无（与 G-1 并行，不同文件） |
| **G-3** | `scripts/image_to_vr180.py` 一键编排 + manifest 断点 + QA 自动验收 + 测试（mock 全部重活） | P0 | G-1、G-2 合并后放行 |
| **G-4** | `LocalSVDProvider` 本地 i2v 骨架（diffusers/SVD，CUDA+MPS 设备探测，mock 后端）| P2 | G-1 |
| **G-5** | 文档：README 快速上手加"单图→VR180"一节 + `docs/IMAGE_TO_VR180.md` 使用指南 | P2 | G-3 |

验收铁律（每张卡）：CI 无模型无 key 全绿；mock 覆盖成功/失败/超时路径；
lead 合并后实机跑 mock 全链路 + （拿到 key 后）一次真实生成端到端。

## 5. 阶段门（owner 真机测试点）

- **门 G-α**：G-1..G-3 合并 + lead 用 mock provider 跑通全链路（假视频→真转换→QA 绿）。
- **门 G-β**：owner 提供任一 API key → lead 真实生成 1 张图 → Quest 真机看片。
- 之后再决定：本地模型（G-4）值不值得做、要不要批量化。

## 6. 技术债（顺手记录，不在本阶段做）

- `pipeline/upscaler.py`（Real-ESRGAN，`--upscale` 默认关）已被 SeedVR2 取代，待废弃。
- NVENC 需 NVIDIA 驱动 ≥610（owner 待升级），当前 8K 输出走软编 x265（慢 ~5×）。
- `--src-hfov` 默认 70° 偏保守，交付样片实际用 150°；等 Quest 打分回来再定新默认。
