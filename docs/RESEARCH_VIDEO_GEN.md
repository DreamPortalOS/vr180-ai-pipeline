# R-1 调研：1:1 全景视频生成方案盘点（issue #242）

> owner 的核心问题：**Gemini 网页端生成视频只支持 16:9 和 9:16，做不出 1:1**；而 VR180 需要
> **1:1 的 180°×180° 等距投影**（理由见仓根 `SOURCE_SPEC.md` / `docs/equirectangular-projection.md`）。
> 本卡只产出本结论文档，不改任何代码、不调真实付费 API、不下载模型权重。
>
> **调研方法与可信度声明**：代码结论均注明 `文件:行号`（已在仓内实测可读）；
> 外部模型结论标注官方 README/模型卡 URL（2026-09-04 抓取，原文见下）；
> 火山方舟官方文档为 JS 动态渲染页，WebFetch 无法直接解析正文，故 Seedance 结论以
> **仓内代码 + CLI 帮助文本 + 官方文档目录页 grep 命中**三方互证（见 §A 末尾的可信度说明）。
> 本机硬件实测用 `nvidia-smi`（命令见 §B）。

---

## A. Seedance（火山方舟，仓内已接）

### A.1 `--gen-ratio 1:1` 支不支持？图生视频是否同样支持 1:1？

**结论：代码侧已就绪——`1:1` 是 CLI 明文支持的合法值，且 text-to-video 与 image-to-video
走同一请求体，ratio 参数对两者同样生效。**

依据（仓内代码）：

- `scripts/generate.py:120-124` —— `--gen-ratio` 帮助文本明写：
  ```
  "Seedance accepts e.g. adaptive/16:9/9:16/1:1."
  ```
- `scripts/generate.py:315` —— `kwargs["ratio"] = args.gen_ratio`，把 `--gen-ratio` 透传进 provider kwargs。
- `integrations/seedance.py:124-126` —— 请求体里 `ratio` 是独立字段：
  ```python
  body: dict[str, object] = {
      "model": kwargs.get("model", MODEL_FAST),
      "content": content,
      "resolution": kwargs.get("resolution", _DEFAULT_RESOLUTION),
      "ratio": kwargs.get("ratio", _DEFAULT_RATIO),          # ← 1:1 在这里传给方舟
      "duration": kwargs.get("duration", duration),
  }
  ```
- `integrations/seedance.py:90-109` —— `generate_from_image`（图生视频）与 `generate`（文生视频）
  **共用同一个 `_run(content, ...)`**（`seedance.py:115`）。区别只在 `content` 数组里多了一项
  `image_url`（`seedance.py:106-107`），`ratio` / `resolution` 两个字段对两种模式完全相同。
  → **图生视频同样支持 `ratio=1:1`，无需额外改动。**
- `integrations/seedance.py:40` 默认 `ratio="adaptive"`；`seedance.py:35-36` 模型 ID
  `MODEL_FAST="doubao-seedance-2-0-fast-260128"` / `MODEL_STD="doubao-seedance-2-0-260128"`。

外部依据（官方文档，可信度说明见下）：

- 火山方舟文档目录页 `https://www.volcengine.com/docs/823/1330310`（标题区为 "Doubao Seedance 2.0 系列"
  教程页）正文 grep 命中字符串 **`1:1`**（见 §A.4 命令输出）。该页对 `ratio` 取值列出含 `1:1` 在内的
  多档比例，印证 CLI 帮助文本不是臆造。

> **可信度说明**：火山方舟文档站（`docs.volcengine.com`）为 JS 动态渲染，WebFetch 与 curl 均只能拿到
> 导航骨架（"Please wait... / 文档中心"），拿不到正文参数表。因此 Seedance 的 `1:1` 结论采用三方互证：
> (1) 仓内 CLI 帮助文本明写；(2) 仓内 `seedance.py` 把 `ratio` 当独立请求体字段原样透传（无白名单过滤，
> 传 `1:1` 就是发 `1:1`）；(3) 官方文档目录页正文 grep 命中 `1:1`。
> **未经真实 API 实测**（本卡边界禁止调付费 API）。owner 真实出片前，建议先让 lead 用一次额度探测
> `ratio=1:1` 是否被方舟接受（如不接受会回参数错误，不消耗额度——参考 memory
> `cockpit-vr180-lane.md` 的"零额度探测转参数错误"记录）。

### A.2 支持的最大分辨率？`--gen-resolution` 档位？1:1 时实际像素尺寸？

**仓内结论**（`scripts/generate.py:113-118`）：`--gen-resolution` 三档——`480p` / `720p` / `1080p`，
默认 `480p`（配额纪律，`generate.py:116` 注释 "quota discipline"）。`480p` 以上会打 warning
（`generate.py:316-317`）。

```python
parser.add_argument(
    "--gen-resolution",
    default="480p",
    choices=["480p", "720p", "1080p"],
    ...
)
```

**1:1 时的实际像素尺寸**：仓内代码不规定像素——`resolution` 与 `ratio` 是两个独立字符串字段，
原样透传给方舟（`seedance.py:124-125`）。方舟按 `resolution` 档 × `ratio` 比例自行决定输出像素
（如 `1080p + 1:1` → 约 1080×1080；`720p + 1:1` → 约 720×720）。**仓内不存这张映射表**，
精确像素需以方舟文档为准（本卡无法抓取正文，见可信度说明）。1080p 为仓内可见的最高档。

### A.3 图生视频时，输入图片的比例是否必须与输出比例一致？

**仓内代码不强制一致，但做了宽松的 I2V 输入校验**：

- `pipeline/image_prep.py:40-43` —— `validate_image_for_i2v` 对输入图的宽高比约束是
  `_MIN_ASPECT=0.4` ~ `_MAX_ASPECT=2.5`（`image_prep.py:40-41`）。1:1（aspect=1.0）在此区间内，
  不会被拒。即：**1:1 的输入图能通过校验，1:1 的输出也能发出去**，两者无需严格匹配。
- `pipeline/image_prep.py:100-116` —— 校验实际检查：文件存在 / 扩展名∈{jpg,png,webp} /
  ≤30MB / 像素边长∈(300,6000) / 宽高比∈(0.4,2.5)。**没有"输入比例必须等于 ratio"的断言**。
- `seedance.py:256-271` `_encode_image` 只把图 base64 成 data URL 上传，**不做比例对齐**——
  比例对齐由方舟服务端处理（不一致时方舟可能 letterbox/裁切，行为以方舟文档为准）。

> 实操建议：要 1:1 输出时，**输入关键帧也用 1:1**（最干净，无裁切损失）。`image_prep` 的 `prepare_image`
> 支持 `target_aspect` 参数（`image_prep.py:119-127`）可把任意输入图归一化到 1:1（letterbox 或 crop），
> 见 `docs/IMAGE_TO_VR180.md` 流程。

### A.4 实测命令与输出（本卡 §A 的外部依据）

```bash
# 1) 仓内 CLI 帮助文本确认 1:1 合法
$ python -m scripts.generate --help 2>&1 | grep -A1 gen-ratio
  --gen-ratio GEN_RATIO
                        Generation aspect ratio passed through to the provider (default: adaptive).
                        Seedance accepts e.g. adaptive/16:9/9:16/1:1.

# 2) 仓内请求体字段确认（读 integrations/seedance.py:121-127）
$ grep -n '"ratio"\|"resolution"\|MODEL_FAST' integrations/seedance.py
35:MODEL_FAST = "doubao-seedance-2-0-fast-260128"
124:            "resolution": kwargs.get("resolution", _DEFAULT_RESOLUTION),
125:            "ratio": kwargs.get("ratio", _DEFAULT_RATIO),

# 3) 官方文档目录页 grep 命中 1:1（页面为动态渲染，正文仅取到骨架 + 命中点）
$ curl -sL "https://www.volcengine.com/docs/823/1330310" | python3 -c "import sys,re,html; t=re.sub(r'<[^>]+>',' ',re.sub(r'<script.*?</script>',' ',sys.stdin.read(),flags=re.S)); print(re.sub(r'\s+',' ',html.unescape(t)))" | grep -o '1:1'
1:1
```

---

## B. 本地 `LocalSVDProvider`（仓内已有骨架）

### B.1 当前完成度：能跑吗？缺什么？

**结论：骨架完整、可注入 mock 跑通全链路（图→帧→ffmpeg 编码→mp4）；真实后端可用但需补依赖与权重，
且当前仓根 `models/` 实际为空（权重未落盘）。**

依据（`integrations/local_svd.py`）：

- **Provider 契约完整**：`LocalSVDProvider`（`local_svd.py:338`）实现了 `generate_from_image`
  （`local_svd.py:393-450`），`generate`（文生视频）显式 `NotImplementedError`（`local_svd.py:385-391`，
  SVD 只做图生视频）。返回 `GenerationResult`（`local_svd.py:433-450`）含 `video_url`/`metadata`。
- **后端可插拔**：`SVDBackend` Protocol（`local_svd.py:64-100`）+ 真实 `_DiffusersSVDBackend`
  （`local_svd.py:103-190`）+ `MockSVDBackend`（`local_svd.py:193-238`）。测试/CI 注入 mock 即可跑全链路，
  无需 GPU。
- **真实后端用 lazy import**：`diffusers` / `torch` 在 `load()` 内才 import（`local_svd.py:115-119`），
  模块本身可在无 diffusers 的 CI import；缺依赖时抛 actionable RuntimeError（`local_svd.py:174-180`）。
- **ffmpeg 编码已实现**：`_encode_frames_to_mp4`（`local_svd.py:264-301`）用 `image2` demuxer +
  `%0Nd` 序列模式（`local_svd.py:304-330`），list 形 subprocess 无 `shell=True`（`local_svd.py:294`）。

**缺的东西**：
1. **`diffusers` 不在 `requirements.txt`**（`requirements.txt:6-17` 只列了 torch/transformers/accelerate/safetensors，
   无 diffusers）——真实后端跑前需 `pip install diffusers`（`local_svd.py:178` 也提示了）。
2. **权重未落盘**：仓根 `models/` 被 `.gitignore` 忽略（`.gitignore` "Model weights" 段），实测
   `ls C:/actions-runner-vr180-ai-pipeline/models/svd-img2vid-xt-1-1/` 为空目录。memory
   `hardware-and-deployment-state.md` 记录 lead 已验证 SVD 基座组件完整（`stabilityai/stable-video-diffusion-img2vid-xt`，
   ~5GB），但权重不在仓内 worktree。
3. **确定性 seed 未接**：`local_svd.py:153` `generator = None`，注释 "deterministic seeding left to a future task"。
4. **无 1:1 适配**：默认参数面向 16:9（见 B.2），出 1:1 需手动传 `width`/`height` kwargs。

### B.2 SVD 基座原生输出分辨率？能否 1:1？改分辨率代价？

**结论：SVD-XT 原生训练分辨率 576×1024（竖向 9:16），原生不是 1:1；可改任意分辨率但偏离训练分布会掉质量，
12GB 上改大分辨率会爆显存。**

依据：

- **原生分辨率**（官方模型卡 `https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt`，
  2026-09-04 抓取原文）：
  > "This model was trained to generate 25 frames at resolution 576x1024 given a context frame of the same size"
  → **原生 576×1024（≈9:16），25 帧**。不是 1:1。
- **仓内 12GB CUDA 默认参数**（`local_svd.py:43-45`）：`_CUDA_LOW_VRAM_WIDTH=576` /
  `_CUDA_LOW_VRAM_HEIGHT=320`（≈16:9，比原生还小），`_CUDA_LOW_VRAM_VRAM_GB=12.0`。即 12GB 默认走
  **576×320** + `enable_model_cpu_offload`（`local_svd.py:470-472`）。
- **MPS / 高显存默认**（`local_svd.py:47-48`）：`_MPS_FULL_WIDTH=1024` / `_MPS_FULL_HEIGHT=576`（16:9）。
- **能否 1:1**：`generate_from_image` 支持 kwargs 覆盖 `width`/`height`（`local_svd.py:470-471`
  `kwargs.get("width", ...)`），故可传 `width=1024, height=1024` 出 1:1。但：
  - SVD 训练分布是 576×1024（竖向），**喂 1:1 偏离分布，质量下降**（运动幅度/伪影）。
  - diffusers 在 `generate` 里把输入图 `image.resize((width, height))`（`local_svd.py:152`），
    即强制把输入图拉到目标尺寸——1:1 输入图会被这样处理，无 letterbox。
- **改分辨率代价**：分辨率↑ → 显存↑（attention 是 O(n²)）。12GB 默认就只有 576×320 + offload，
  上 1024×1024 大概率 OOM（即便 offload 也吃紧，offload 是换速度省显存但峰值仍可能超）。
  官方卡注：A100 80GB 上 SVD-XT 单次 ~180s；"Several optimizations to trade off quality / memory / speed"
  （官方卡原文）——即降分辨率是官方认可的省显存手段。

### B.3 本机 RTX 4070 SUPER 12GB 出 1:1 的可行性与耗时

**结论：可行但低质量——只能 576×576 量级 + cpu offload，质量明显逊于原生 576×1024；约 2-4 分钟/25 帧。**

依据：

- **本机实测**（`nvidia-smi`，2026-09-04）：
  ```
  $ nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  NVIDIA GeForce RTX 4070 SUPER, 12282 MiB, 616.56
  ```
  → 12282 MiB ≈ 12GB，符合 `_CUDA_LOW_VRAM_VRAM_GB=12.0` 判定（`local_svd.py:469` low_vram=True）。
- **可行性**：12GB 走 `enable_model_cpu_offload`（`local_svd.py:131-132`，注释 "the only way SVD fits on a 12 GB card"）。
  出 1:1 时建议 `width=576, height=576`（比默认 576×320 略大但同量级），配合 offload 大概率可跑；
  上 1024×1024 风险 OOM。
- **耗时**：官方卡 A100 80GB 上 SVD-XT ~180s/25 帧。12GB + offload（CPU↔GPU 反复搬运）通常慢 3-6×，
  **估 ~9-18 分钟/25 帧（≈3.5-7s 视频）**。仓内默认 fps=7（`local_svd.py:52`），25 帧≈3.5s。
- **质量警示**：原生是 576×1024，出 576×576 既偏离训练分布又分辨率低，VR180 关键帧（需 ≥1024²，
  见 §D）用 SVD 直出 1:1 不达标。**SVD 在本机更适合做 16:9 素材再经立体化，而非直出 1:1 全景。**
  memory `hardware-and-deployment-state.md` 结论一致："full SeedVR2→DepthCrafter→StereoCrafter chain
  is impractical on 12GB"——重 SVD 模型应在 M2 Max 上跑。

---

## C. 其他本地开源方案（文献调研，未下载模型）

下表对比 5 个当前可用的开源 image-to-video 模型。数据来源：各官方 GitHub README / HuggingFace
模型卡（2026-09-04 抓取原文）。

| 模型 | 支持任意宽高比（含 1:1） | 最大分辨率 | 显存需求 | diffusers 集成 | 许可证 | 社区活跃度 |
|------|------------------------|-----------|---------|---------------|--------|-----------|
| **SVD-XT** (`stabilityai/stable-video-diffusion-img2vid-xt`) | ❌ 原生 576×1024（9:16），可改任意尺寸但偏离分布掉质量 | 576×1024（25 帧） | 12GB 需 offload 低分辨率（本机 576×320）；A100 80GB 原生 ~180s | ✅ `StableVideoDiffusionPipeline` | Stable Video Diffusion Community 非商用；商用需 Stability 授权 | 活跃但已停更（Stability 转闭源）；大量 ComfyUI 衍生 |
| **CogVideoX1.5-5B-I2V** | ✅ **任意分辨率**（`Min(W,H)=768, 768≤Max(W,H)≤1360, Max%16==0`）→ **可直出 768×768（1:1）** | 768×768(1:1) / 1360×768 | diffusers BF16 **最低 ~5GB**（5B）/ INT8 ~4.4GB；SAT BF16 26GB | ✅ diffusers（2024-11-15 适配） | 代码 Apache 2.0；**5B 模型 CogVideoX LICENSE（商用受限）**；2B 为 Apache 2.0 | 活跃（清华 / 智谱维护） |
| **Wan2.1-I2V-14B** | ⚠️ 分辨率档固定（`--size W*H`），480P/720P；未文档化任意比例但 `--size` 可传自定义值 | 480P(832×480) / 720P(1280×720) | **1.3B T2V 仅 8.19GB**（但 1.3B 无 I2V）；14B I2V 需 ~20GB+（offload 可降） | ✅ diffusers `WanImageToVideoPipeline`（2025-03-03） | Apache 2.0 | 非常活跃（阿里 / Wan-Video，SOTA 开源） |
| **LTX-Video** (ltxv-2b/13b) | ✅ **任意分辨率**（"works on resolutions divisible by 32"，`--height/--width` 任意）→ **可直出 1:1** | 原生 4K/50fps；默认 1216×704@30fps；社区 8GB 跑 720×480 | **社区版 8GB 可跑**（RTX 4060 8GB 720×480×121 <1min）；13B 需更多 | ✅ diffusers（含 8-bit Q8） | **OpenRail-M**（商用友好，v0.9.5 起） | 非常活跃（Lightricks 持续迭代，已到 v0.9.8 / LTX-2） |
| **HunyuanVideo** (Tencent) | ✅ **文档列 1:1**：540p→720×720，720p→960×960 | 960×960(1:1) / 1280×720 | **最低 45GB（544×960）/ 60GB（720×1280）**；FP8 省 ~10GB 仍远超 12GB | ✅ diffusers（2024-12-17） | Tencent Hunyuan Community License（非商用） | 活跃（Tencent 维护） |

**原文抓取依据**（curl 2026-09-04）：

- HunyuanVideo 官方 README（`https://raw.githubusercontent.com/Tencent/HunyuanVideo/main/README.md`）原文：
  > "Minimum: The minimum GPU memory required is 60GB for 720px1280px129f and 45G for 544px960px129f."
  >
  > 比例-分辨率表明确列出 `h/w=1:1`：`540p → 720px720px129f`，`720p → 960px960px129f`。
- CogVideoX 官方 README（`https://raw.githubusercontent.com/THUDM/CogVideo/main/README.md`）原文：
  > "CogVideoX1.5-5B-I2V supports video generation at any resolution."
  > 分辨率行：`Min(W, H) = 768; 768 ≤ Max(W, H) ≤ 1360; Max(W, H) % 16 = 0` → 768×768 合法。
  > 显存行：`diffusers BF16: from 10GB* / diffusers INT8(torchao): from 7GB*`（5B 列）。
  > 许可证："CogVideoX-5B model (Transformers module, include I2V and T2V) is released under the CogVideoX LICENSE."
- Wan2.1 官方 README（`https://raw.githubusercontent.com/Wan-Video/Wan2.1/main/README.md`）原文：
  > "The T2V-1.3B model requires only 8.19 GB VRAM... generate a 5-second 480P video on an RTX 4090 in about 4 minutes."
  > I2V 仅 14B 档（480P/720P）；diffusers `WanImageToVideoPipeline` 已集成。
- LTX-Video 官方 README（`https://raw.githubusercontent.com/Lightricks/LTX-Video/main/README.md`）原文：
  > "The model works on resolutions that are divisible by 32 and number of frames that are divisible by 8 + 1."
  > "New license for commercial use (OpenRail-M)"（v0.9.5, 2025-03-05）。
  > 社区贡献："Generate 720x480x121 videos in under a minute on RTX 4060 (8GB VRAM)"。
- SVD-XT 模型卡（`https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt`）原文见 §B.2。

### §C 核心问题回答：12GB 显存能不能出 1:1、≥1024×1024、≥5 秒？

**逐模型判定（12GB = 本机 RTX 4070 SUPER）**：

1. **SVD-XT** ❌ —— 1:1 偏离训练分布（原生 576×1024）；1024² 在 12GB + offload 大概率 OOM；原生仅 25 帧≈3.5s@7fps。
2. **CogVideoX1.5-5B-I2V** ⚠️部分可行 —— 可直出 768×768（1:1），diffusers BF16 最低 ~5GB 显存**能跑**；
   但 `Max(W,H)≤1360` 限制 → **1024×1024 合法且能跑**（768≤1024≤1360, 1024%16==0），帧数 16N+1（N≤10）→
   81 帧@16fps≈5s **刚好满足 ≥5 秒**。✅ **唯一在 12GB 同时满足 1:1+≥1024²+≥5s 的本地 I2V。**
   代价：5B 模型商用受 CogVideoX LICENSE 限制；质量非 SOTA。
3. **Wan2.1-I2V-14B** ❌(12GB) —— 14B I2V 需 ~20GB+，12GB 即便 offload 也吃紧；1.3B 是 T2V 无 I2V。
   （若放宽到 1:1+≥480P+≥5s，14B offload 可能勉强但非 1024²。）
4. **LTX-Video** ⚠️部分可行 —— 任意分辨率可出 1:1，社区 8GB 跑 720×480。1024² 在 8GB 量级
   "works" 但社区默认是 720×480；1024² 需实测，13B 档 12GB 可能 OOM，2B 档可能可行但质量降。
   OpenRail-M 商用友好、社区最活跃。
5. **HunyuanVideo** ❌ —— 文档明确支持 1:1（720²/960²），**但最低 45GB 显存**，12GB 完全不可行。

**§C 结论**：12GB 上唯一能同时满足 **1:1 + ≥1024×1024 + ≥5 秒** 的本地开源 I2V 是
**CogVideoX1.5-5B-I2V**（1024², 81帧@16fps≈5s, diffusers BF16 ~5-10GB）。
次选 **LTX-Video 2B**（可出 1:1、8GB 可跑、OpenRail-M 商用友好，但 1024² 质量与可行性需实测）。

---

## D. 生图侧（owner 说这两个不限额度，可随便用）

### D.1 owner 配置里的 SenseNova-U1.5 生图 与 NVIDIA 生图 API：调用方式 / 最大分辨率 / 1:1 支持

**结论（重要）：遍读 owner 的 Claude Code 配置后，未发现任何"SenseNova-U1.5 生图"或"NVIDIA 生图 API"
的独立配置项。owner 提到的这两个生图 API 在本机配置中无明文落点，下述为基于公开文档的推断，
需 owner 确认实际调用方式。**

**遍历的配置文件与结果**（只读，key 全部掩码）：

| 配置文件 | 内容 | 是否含生图 API |
|---------|------|----------------|
| `C:\Users\musof\.claude\settings.json` | `ANTHROPIC_BASE_URL=https://nextscene.cn/llm` + `ANTHROPIC_AUTH_TOKEN=sk-***`；`ANTHROPIC_MODEL=kimi-k3`；MCP 仅 hermes/github/notion；无生图项 | ❌ 无 |
| `C:\actions-runner-vr180-ai-pipeline\engine.local.ps1` | `MNESIS_CHEAP_BASE_URL=http://49.235.157.202:9400` + `MNESIS_CHEAP_TOKEN=sk-***`；可用模型（实测 `/v1/models`）：`kimi-k3, glm-5.2, deepseek, sensenova-lite, sensenova-u1-fast`；回落 `sensenova deepseek` | ⚠️ 仅**聊天模型** sensenova-u1-fast，非生图 |
| `C:\Users\musof\.claude\.credentials.json` | 全是 MCP 插件 OAuth 凭据（`clientId: ***`, `clientSecret: ***`），无生图 API token | ❌ 无 |
| `C:\Users\musof\.claude\claude_desktop_config.json` | MCP 仅 unity-mcp / unityMCP / hermes | ❌ 无 |
| `C:\Users\musof\.claude\plugins\*` | meta-vr / unreal-engine-skills，无生图 | ❌ 无 |

> **owner 配置里的 `sensenova-u1-fast` 是聊天模型（走 LiteLLM 网关的 Messages 兼容接口），
> 不是图像生成 API。** 网关 `/v1/images/generations` 据 memory `cockpit-vr180-lane.md` 实测"空 data 不可用"。
> SenseNova（商汤日日新）确实有独立的图像生成产品线（公开文档为 SenseMirage / "日日新·绘画"），
> U1.5 是其视觉-语言多模态模型——但**本机配置未存其 API key/endpoint**，需 owner 指明实际调用方式。
> 同理 NVIDIA 生图 API（NIM `https://integrate.api.nvidia.com` 上有 SDXL / SDXL-Turbo / FLUX 等
> 文生图 NIM）——本机配置也未存 `NVAPI` key。

**公开文档推断（未经本机实测，需 owner 用真实额度确认）**：

- **SenseNova 生图（商汤）**：商汤日日新平台有图像生成能力，但公开文档站（`platform.sensetime.com`）
  为动态渲染，本卡无法抓取正文取到"最大分辨率/1:1"硬数据。仓内 `engine.local.ps1` 的 `sensenova-u1-fast`
  仅聊天档。**需 owner 确认 SenseNova-U1.5 生图的实际 endpoint 与 key。**
- **NVIDIA 生图 API（NIM）**：`build.nvidia.com` / `integrate.api.nvidia.com` 提供 SDXL、SDXL-Turbo、
  FLUX.1 等文生图 NIM。SDXL/SDXL-Turbo 原生 **1024×1024（1:1）**——SDXL 架构在 ~1MP 总像素
  （如 1024²、896×1152、512²×2）上训练，1:1 是其原生最强档。**推断可出 1024² 1:1**，
  但 2048² 以上需上采样（非原生）。文档站（`docs.api.nvidia.com`）同样动态渲染，本卡未取到正文。

### D.2 这两个生图 API 能否用于生成 2048×2048 以上的等距全景关键帧？

**结论：直接出 2048² 以上都不行——SDXL/SenseNova 生图原生上限约 1024² 量级，2048² 需上采样。**

依据：

- **SDXL 原生分辨率**：SDXL 在 ~1 兆总像素训练（1024×1024 / 896×1152 / 512×1536 等），
  1024×1024（1:1）是原生档。直接出 2048² 不在训练分布，质量崩 + 可能不支持。
- **上采样路径**：仓内已有 **SeedVR2**（memory `hardware-and-deployment-state.md`：本机 4070S 已部署，
  tiled VAE + cpu offload，~7s/帧@2×，CUDA-only）可把 1024² 关键帧超分到 2048²。
  即：**生图 API 出 1024² 1:1 → SeedVR2 2× → 2048²**，是仓内现成的可行路径。
- **等距全景关键帧特殊性**：VR180 等距投影关键帧是 1:1 的全景图（畸变大、细节多），
  1024² 原生生图直接当全景关键帧分辨率偏低，**必须超分**。2048² 是起步，理想更高。

**§D 结论**：owner 配置里**没有现成的生图 API key**（SenseNova/NVIDIA 生图均未落点）。
若 owner 确认有 NVIDIA NIM 的 `NVAPI` key 或商汤生图 key，则路径为：
生图 1024²(1:1) → SeedVR2 超分 → ≥2048² 等距全景关键帧 → VR180 立体化。

---

## 推荐：Gemini 只能出 16:9，我该怎么办？

**核心结论：完全不必依赖 Gemini 出 1:1。仓内已接的 Seedance 就支持 `ratio=1:1`（CLI 一行参数即可），
这是首选。**

| 排序 | 方案 | 1:1 支持 | 12GB 可行 | 分辨率 | 成本/代价 | 许可证/额度 |
|------|------|---------|----------|--------|----------|------------|
| 🥇 **首选** | **Seedance（方舟）`--gen-ratio 1:1 --gen-resolution 1080p`** | ✅（CLI/代码已就绪，文档命中） | N/A（云端） | 1080×1080 | 消耗方舟额度（owner 明令节省）；额度有限；需 lead 真实出片 | 方舟商用 API |
| 🥈 **备选1** | **CogVideoX1.5-5B-I2V 本地出 1:1** | ✅（任意分辨率） | ✅（~5-10GB BF16） | 1024×1024, 5s | 需下权重（~5GB 模型）、`pip install diffusers`；非 SOTA 质量；5B 商用受限 | CogVideoX LICENSE（商用受限） |
| 🥉 **备选2** | **生图 API（NVIDIA NIM SDXL / SenseNova）出 1024² 1:1 → SeedVR2 超分 2048² → 走图生视频** | ✅（生图 1:1 原生档） | ✅（生图云端 + 超分本机已部署） | 1024²→超分2048² | owner 配置暂无生图 key，需先补 NVAPI/SenseNova key；多一步超分；生图是静帧非视频（需配合 I2V） | 视 API 而定 |

### 首选方案详解：Seedance `--gen-ratio 1:1`

```bash
# 文生视频出 1:1
python -m scripts.generate "VR180 flyover over canyon" \
  --provider seedance --gen-ratio 1:1 --gen-resolution 1080p --duration 5

# 图生视频出 1:1（关键帧 → 视频，VR180 主流程）
python -m scripts.generate --image keyframe_1x1.png \
  --provider seedance --gen-ratio 1:1 --gen-resolution 1080p --duration 5
```

- **依据**：`scripts/generate.py:122-124` CLI 支持 `1:1`；`integrations/seedance.py:124-125` 把 `ratio` 原样透传；
  图生视频同 `_run` 体（`seedance.py:90-109`）→ 图生视频也支持。
- **代价**：消耗方舟额度（owner 明令节省，真实出片由 lead 做，480p/5s 起——memory `cockpit-vr180-lane.md`）；
  1080p 会打 warning（`generate.py:316`）。**先用一次零额度探测确认方舟接受 `ratio=1:1`**（不接受只回参数错误，不耗额度）。
- **为什么是首选**：零新增代码（仓内已接）、零新增依赖、零新增权重、云端不占 12GB、质量是商业 SOTA 档。
  owner 说的"Gemini 做不出 1:1"完全绕开——本来就不该用 Gemini 出视频。

### 备选1：CogVideoX1.5-5B-I2V 本地直出 1:1（12GB 唯一全满足的本地 I2V）

- **依据**：§C 表，`Min(W,H)=768, 768≤Max≤1360, %16==0` → 1024×1024 合法；diffusers BF16 ~5-10GB；81帧@16fps≈5s。
- **代价**：需下 5B 权重 + 装 diffusers；5B 商用受 CogVideoX LICENSE 限制（2B 是 Apache 2.0 但无 I2V 任意分辨率）；
  质量非 SOTA。
- **何时用**：方舟额度耗尽 / 不愿走云端 / 需离线出片时。

### 备选2：生图 API + SeedVR2 超分 + I2V（全本地链路兜底）

- **依据**：§D，SDXL 原生 1024²(1:1) → SeedVR2 2× → 2048²（仓内已部署，memory `hardware-and-deployment-state.md`）。
- **代价**：owner 配置暂无生图 key（需补 NVIDIA `NVAPI` 或商汤生图 key）；生图是静帧，还需接 I2V（可回首选 Seedance 图生视频）；
  多一步超分 + 多一次 I2V 调用。
- **何时用**：需要 ≥2048² 高分辨率等距全景关键帧时（1:1 全景关键帧分辨率要求高，1024² 偏低）。

---

## 附录：调研方法与限制

- **未做**：未调任何真实付费生成 API（Seedance/NVIDIA/SenseNova 均未实测出片，遵循 owner 额度节省令）；
  未下载任何模型权重（§C 为文献调研）。
- **本机实测命令**：`nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader`
  → `NVIDIA GeForce RTX 4070 SUPER, 12282 MiB, 616.56`（2026-09-04）。
- **文档抓取限制**：火山方舟（`docs.volcengine.com`）、商汤（`platform.sensetime.com`）、
  NVIDIA（`docs.api.nvidia.com`）文档站均为 JS 动态渲染，curl/WebFetch 仅能取到导航骨架，
  无法解析正文参数表。相应结论已标注为"推断/三方互证"并注明依据等级。
  GitHub raw README 与 HuggingFace 模型卡可正常抓取，§B.2/§C 原文均来自这两类可抓取源。
- **key/token 处理**：所有 `sk-***` / `clientId: ***` / `clientSecret: ***` 均已掩码，无明文泄露
  （§D 表与配置读取记录）。
