# VR180 友好的视频生成 Prompt 指南

> 目的：在 AI 视频**生成阶段**就让素材更适合后续 VR180 合成，从源头减少致晕、提升立体质量。
> 关联：用户反馈"速度快致晕、立体融合差"。这是从 Prompt 端的对策。

---

## 一、为什么 Prompt 决定 VR180 体验

VR180 致晕的根因不只是分辨率，更是 **双眼视差冲突（vergence-accommodation conflict）**：
- 画面运动太快 + 立体对齐不准 → 大脑无法融合左右眼 → 眩晕
- 镜头剧烈旋转/抖动 → 前庭觉与视觉不符 → 眩晕
- 画面边缘快速掠过的物体 → 视差极端 + 运动模糊 → 眩晕

→ **在生成阶段控制运动与构图，比后期补救有效得多。**

---

## 二、VR180 友好 Prompt 的 6 条原则

| # | 原则 | 正面写法 | 避免 |
|---|------|---------|------|
| 1 | **运动平稳** | smooth, steady, continuous forward glide, moderate speed | rapid, fast, sudden, jerky |
| 2 | **地平线稳定** | level horizon, stable altitude | rolling, tilting, barrel roll, spinning |
| 3 | **景深分层** | foreground / mid-ground / background layers, clear depth | flat, single-plane, everything in focus far away |
| 4 | **主体居中** | main subject centered, framed ahead | subjects rushing past frame edges |
| 5 | **广角适中** | wide cinematic ~120° FOV | extreme fisheye, ultra-fast zoom |
| 6 | **高画质** | 8K, ultra-detailed, sharp focus, photorealistic | motion blur, low detail |

---

## 三、Prompt 包装层逻辑（界面"VR180 优化"按钮的底层实现）

用户在界面输入任意创意，点击「优化为 VR180」按钮，系统在底层套用以下模板：

```
{用户原始创意}
, first-person FPV view, smooth continuous forward motion at moderate speed,
level and stable horizon, rich depth layers (foreground/mid-ground/background),
main subject centered, wide cinematic ~120° field of view,
soft natural lighting, ultra-detailed, sharp focus, 8K, photorealistic.
Negative: rapid turns, barrel rolls, camera shake, motion blur,
sudden cuts, extreme close-ups at frame edges, flat composition.
```

实现要点（给 Cline）：
- 函数 `wrap_prompt_for_vr180(user_prompt: str, scene_type: str = "fpv") -> str`
- 不同 `scene_type`（fpv / walkthrough / orbit / static）套不同运动模板
- 保留用户原意，仅**追加约束**，不删改用户内容
- 正向约束 + 负向约束（negative prompt）分离，适配各模型 API

---

## 四、针对当前恐龙 FPV 场景的优化 Prompt（可直接拿去 Seedance 重生成）

### 优化版（推荐）
```
First-person FPV drone flight gliding smoothly and steadily forward at a
moderate, controlled speed through a lush prehistoric valley. Wide cinematic
landscape: herds of brachiosaurus grazing in the mid-ground, a distant
snow-capped volcano gently smoking on the horizon, waterfalls cascading far
behind. The camera moves in one continuous smooth forward motion at a stable
low altitude, horizon perfectly level, no rolling or shaking. Clear depth
layers — ferns and grass in the foreground, dinosaurs in the mid-ground,
mountains far away. Warm golden-hour lighting. Ultra-detailed, razor-sharp
focus, 8K, photorealistic, cinematic.

Negative: rapid turns, barrel rolls, camera shake, motion blur, fast zoom,
sudden cuts, dinosaurs rushing past the frame edges, flat composition.
```

### 与原素材的关键差异
| 原素材问题 | 优化点 |
|-----------|--------|
| 速度过快致晕 | moderate controlled speed + smooth continuous motion |
| 边缘物体快速掠过 | 主体放 mid-ground，negative 排除 rushing past edges |
| 画面模糊 | razor-sharp focus, 8K, no motion blur |
| 立体层次不足 | 显式要求 foreground/mid/background 三层景深 |
| 地平线不稳 | horizon perfectly level, no rolling |

---

## 五、不同模型的 Prompt 适配差异

| 模型 | 运动控制 | Negative prompt | 备注 |
|------|---------|----------------|------|
| Seedance | 文本描述 | 支持 | 当前主力 |
| Kling | 专业运镜术语 + 相机控制参数 | 支持 | 运动控制最强，适合稳定运镜 |
| Veo 3 | 文本 + 原生音频 | 部分 | 最高 4K，画质好 |

→ Kling 的相机控制最适合"平稳运镜"需求，建议稳定性优先时用 Kling。
