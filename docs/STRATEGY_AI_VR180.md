# 战略分析：AI 原生生成 VR180 的可行性与技术路线

> 调研日期：2026-06-23
> 调研人：Claude Code（主脑）
> 结论级别：**项目核心架构决策**

---

## 一、核心问题

能否通过 Prompt 直接让 Seedance / Kling / Google Veo 生成"适合合成 VR180"甚至"直接是双眼立体"的视频，从而跳过或简化后处理？

---

## 二、调研结论（一句话）

**目前没有任何主流 AI 视频生成模型支持原生立体（stereoscopic）或 VR180 输出。**
"AI 原生生成 VR180" 这条路当前不可行。质量提升的真正杠杆在于：**用 SOTA 的 2D→3D 转换模型替换现有简陋实现**。

---

## 三、三大模型现状（实证）

| 模型 | 最高输出 | 立体支持 | 相机控制 | 结论 |
|------|---------|---------|---------|------|
| **Google Veo 3.1** | 4K / 24fps / 16:9·9:16 | ❌ 无 | 电影级运镜 | 仅 2D |
| **Kling 3.0** | 4K / 多镜头 | ❌ 无 | pan/tilt/zoom/dolly | 仅 2D，无精确 IPD 位移 |
| **Seedance 2.0** | 当前素材来源 | ❌ 无 | 有 | 仅 2D |

### 为什么"双机位生成双眼"路线（路线 C）不可行

设想"同一 Prompt 生成左眼 + 右移 6.4cm 生成右眼"：
1. **内容不一致**：扩散模型同一 Prompt 两次生成内容不同，左右眼会是两个不同的世界 → 无法融合
2. **无精确位移控制**：Kling 的相机控制是语义级（"向左平移"），不是厘米级 IPD 控制
3. 即便用 image-to-video 锁定首帧，运动过程仍会发散

→ **放弃路线 C。**

---

## 四、正确的技术路线

```
Seedance/Kling/Veo（2D 视频，优化 Prompt：广角/低空/FPV）
        │
        ▼
[阶段1] 时序一致深度估计   ← 升级：DepthCrafter / Video-Depth-Anything
        │                    （替换现有逐帧 Depth Anything V2）
        ▼
[阶段2] 高保真立体对生成   ← 升级：StereoCrafter
        │                    （替换现有简单视差位移 stereo_renderer）
        ▼
[阶段3] 等距投影           ← 保留（已修复方向 + FOV）
        │                    equirectangular_mapper.py
        ▼
[阶段4] 边缘补全           ← 新增：上下黑边 AI outpainting
        │
        ▼
[阶段5] 超分               ← 保留 4K→8K
        │
        ▼
     VR180 (Quest)
```

### 关键工具选型（均为开源，腾讯系，CVPR 2025）

**1. StereoCrafter（最高优先级）**
- 仓库：https://github.com/TencentARC/StereoCrafter
- 原理：depth-based video splatting（视差扭曲）+ **stereo video inpainting**（基于 Stable Video Diffusion 补全遮挡区域）
- 价值：直接解决当前"左右眼背景色不一致 / 视差空洞"问题——它内部用扩散模型补全空洞，而非简单填色
- 明确支持 **AIGC 视频**输入（我们的场景）

**2. DepthCrafter / Video-Depth-Anything**
- 仓库：https://github.com/Tencent/DepthCrafter ｜ https://github.com/DepthAnything/Video-Depth-Anything
- 价值：时序一致的视频深度，消除现有逐帧深度的闪烁/抖动

这两个是**对现有架构的 drop-in 升级**，不破坏 `streaming_pipeline.py` 的分阶段结构，只替换阶段1、2 的实现。

---

## 五、硬件现实（决定项目部署形态）

| 工具 | 硬件需求 | 本机 Mac (MPS) |
|------|---------|---------------|
| 现有 Depth Anything V2 small | 低 | ✅ 可跑 |
| StereoCrafter | NVIDIA CUDA，建议 ≥16GB VRAM | ❌ 跑不了 |
| DepthCrafter | NVIDIA CUDA | ❌ 跑不了 |

**结论**：质量提升必须依赖 NVIDIA GPU。这恰好与"平台化（类 ComfyUI 的在线工具）"目标一致——**核心算力放云端 GPU，本机只做开发与轻量预览**。这也使得 A1 的 Celery 异步队列成为正确的前置投资（云端 GPU worker 跑重任务）。

---

## 六、对当前项目逻辑的优化建议

1. **保留并固化现有 baseline**：Depth Anything + 简单立体 + 等距投影，作为"快速预览档"（Mac 可跑）
2. **新增"高质量档"**：DepthCrafter + StereoCrafter，跑在云端 CUDA worker
3. **Prompt 工程并行验证**（零成本）：生成时就要求广角、低空、FPV、画面主体居中，减少后期补全压力
4. **架构上抽象 depth/stereo 后端**：定义统一接口，baseline 与高质量档可切换

---

## 七、Prompt 工程建议（阶段0 优化）

生成 FPV 源视频时，Prompt 加入：
```
ultra-wide angle lens, ~120° field of view, low-altitude FPV drone flight,
fast forward motion, main subject centered, cinematic, stable horizon,
minimal extreme close-ups at frame edges
```
目的：让源视频的有效内容覆盖更大画面比例，减少等距投影后的黑边与后期 outpainting 负担。

---

## 八、待用户确认的关键决策

1. 是否有 NVIDIA GPU 资源（本地/云）用于 StereoCrafter？
2. 已有哪些视频生成 API 权限（Seedance / Kling / Veo）？
3. 优先级：先把"质量"做到极致，还是先把"平台流程"跑通？
