# VR180 FPV Flight Video — Prompt Design Guide

## 为什么提示词设计很重要

Gemini Omni 生成的 2D 视频质量直接决定了 VR180 转换的效果。良好的提示词应当：
- **产生清晰的前景/背景分离** — 深度估计依赖于此
- **有明确的透视感和运动** — FPV 飞行的核心体验
- **生成 16:9 横屏视频** — 最佳兼容 VR180 等距柱状投影

---

## 提示词模板体系

### 基础模板（通用飞行）

```
[场景描述]，FPV 第一人称视角飞行，无人机穿越 [环境]，[动作/速度描述]。
[视觉风格]，[光照条件]，[色彩调性]。
16:9 横屏，平滑流畅，浅景深效果，前方视野开阔。
```

### 场景分类模板

#### 1. 自然风光飞行

```
FPV drone flying through [自然景观]，first-person view，
[动作描述] weaving between [元素1] and [元素2]，
[季节/天气]，[光照条件]，[色彩调性]，
16:9 landscape, cinematic, smooth motion, depth of field,
clear foreground-background separation for stereoscopic depth,
FPV racing drone perspective, wide open sky ahead.
```

**Prompt 示例 — 峡谷飞行：**
```
Cinematic FPV drone flying through a narrow red rock canyon,
first-person perspective, weaving between towering sandstone walls,
morning golden hour light casting long shadows,
warm earthy tones with deep blue sky visible above,
16:9 landscape, smooth flying speed, clear depth separation
between nearby rock walls and distant canyon opening,
dust particles illuminated in shafts of sunlight,
professional FPV racing drone camera view.
```

**Prompt 示例 — 森林飞行：**
```
FPV first-person drone flight through a misty ancient redwood forest,
weaving between massive tree trunks, dappled sunlight through canopy,
morning fog hugging the forest floor, emerald green and golden light,
16:9 cinematic landscape view, smooth arcing turns,
clear depth layers: nearby branches, middle tree trunks, distant foggy backdrop,
dynamic motion with stable horizon, professional cinematic drone perspective.
```

#### 2. 城市/建筑飞行

```
FPV drone flight over [城市类型]，first-person perspective，
[飞行轨迹] between [建筑特征]，
[时段] lighting，[色彩调性]，
16:9 landscape, smooth cinematic motion,
strong geometric depth cues from [建筑特征]，
modern cityscape FPV perspective.
```

**Prompt 示例 — 未来城市：**
```
FPV drone racing through a futuristic cyberpunk city at night，
first-person perspective weaving between neon-lit skyscrapers,
rain-slicked streets far below, holographic billboards,
deep blues and purples with warm neon accents,
16:9 cinematic landscape, smooth banking turns,
strong depth layers: nearby buildings, mid-cityscape, distant towers,
light trails from flying vehicles below,
professional cinematic drone FPV camera view.
```

#### 3. 抽象/梦幻飞行

```
Surreal FPV flight through [梦幻场景]，
first-person perspective, [运动描述]，
[视觉风格]，[光照氛围]，
16:9 landscape, smooth fluid motion,
dreamlike atmosphere, strong depth layering for 3D effect.
```

**Prompt 示例 — 云层之上：**
```
FPV drone flight above a sea of clouds at sunrise,
first-person perspective soaring through golden mist,
distant mountain peaks piercing through the cloud layer,
warm pink and orange gradients in the sky,
16:9 cinematic landscape, smooth gliding motion,
clear depth separation: nearby cloud wisps, mid-layer clouds, distant peaks,
sunlight creating volumetric rays through the clouds,
peaceful aerial drone perspective, professional cinematic quality.
```

---

## Prompt 优化技巧（针对 VR180 转换）

### ✅ 增强深度感的关键词

| 目的 | 提示词短语 |
|------|-----------|
| 强化前景 | `nearby [元素] in foreground` |
| 分层深度 | `clear depth layers: foreground, midground, background` |
| 运动视差 | `weaving between objects` |
| 景深效果 | `shallow depth of field, bokeh` |
| 空间感 | `wide open space, dramatic scale` |

### ✅ 选择有利于深度估计的运动

- `smooth arcing turns` — 平滑弧线转弯，产生运动视差
- `weaving between objects` — 在物体间穿行，产生前后景分离
- `gliding, soaring` — 滑翔，稳定的运动速度
- `dynamic but smooth motion` — 动态但平滑

### ❌ 避免

- `fast cuts, quick edits` — 快速剪辑，帧间不连续
- `extreme fisheye` — 极端鱼眼，扭曲深度
- `rapid spinning` — 快速旋转，产生运动模糊
- `complicated multi-person interaction` — 复杂人物交互，深度难以处理

---

## Gemini Omni 平台操作指南

### 步骤

1. 访问 `gemini.google.com` → 选择 Gemini Omni（需 AI Plus/Pro 订阅）
2. 在提示词输入框中选择 **Text to Video** 模式
3. 粘贴上述提示词（推荐从自然风光模板开始）
4. 设定输出为 **16:9, 10秒**
5. 生成后下载 MP4 文件

### 参数建议

| 参数 | 建议值 | 说明 |
|------|--------|------|
| 时长 | 10秒 | Gemini Omni Flash 最大长度 |
| 宽高比 | 16:9 | VR180 最佳兼容 |
| 分辨率 | 1080p | 当前 Omni Flash 最高 |
| 帧率 | 30fps | 平衡质量与计算量 |

### 首轮推荐提示词（用于首次测试）

```
Cinematic FPV drone flying through a narrow red rock canyon,
first-person perspective, weaving between towering sandstone walls,
morning golden hour light, warm earthy tones,
16:9 landscape, smooth flying speed,
clear foreground-background separation,
professional FPV racing drone camera view.
```

```
FPV drone flight above a sea of clouds at sunrise,
first-person perspective soaring through golden mist,
distant mountain peaks piercing through the cloud layer,
16:9 cinematic landscape, smooth gliding motion,
clear depth layers, dramatic scale.
```

---

## 视频质量检查清单

在将视频传入流水线前，检查以下内容：

- [ ] 视频是 16:9 横屏格式
- [ ] 视频长度 ≤ 10秒（Omni Flash 限制）
- [ ] 无快速剪辑或场景切换
- [ ] 有明显的前景物体（增强立体感）
- [ ] 运动平滑，无剧烈抖动
- [ ] 色彩对比适中，避免全黑/全白场景
- [ ] 下载的 MP4 可以正常播放