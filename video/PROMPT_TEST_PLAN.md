# Prompt 测试任务清单（照着做）

> 目标：找出"让 AIGC 生成什么样的素材"最适合后续 180/360 转换。
> 用法：把下面的 prompt 直接粘进你的 AIGC 工具（Veo / Kling / Seedance），各生成一条 5-10s 片子，
> 按文件名存到 `video/`，戴 Quest 或肉眼按"评分表"打分，把最好的 1-2 条发给 Claude 转 VR180/球幕实测。

生成更多变体：`PYTHONPATH=. python scripts/prompt_lab.py --prompt "你的创意" --out video/my_prompts.json`

---

## 实验 A — 目标取景（核心：180 vs 360）

同一个创意，只变"目标格式"，看哪种最适合。

### [ ] A1 — VR180 头显（前向 120°，立体）→ 存为 `test_A1_vr180.mp4`
```
First-person FPV view, smooth continuous forward motion at moderate speed, level and stable horizon, stable low altitude, rich depth layers (foreground/mid-ground/background), main subject centered, wide cinematic ~120° field of view, soft natural lighting, ultra-detailed, sharp focus, 8K, photorealistic
Negative: rapid turns, barrel rolls, camera shake, motion blur, sudden cuts, extreme close-ups at frame edges, flat composition, fast zoom, rushing past frame edges
```
（把你的场景描述加在最前面，例如 "FPV flight over a coastal cliff at sunrise, ..."）

### [ ] A2 — 球幕/飞行影院（前向 150-180°，单目）→ 存为 `test_A2_fulldome.mp4`
```
First-person FPV view, smooth continuous forward motion at moderate speed, level and stable horizon, stable low altitude, ultra-wide cinematic ~150-180° field of view, soft natural lighting, ultra-detailed, sharp focus, 8K, photorealistic, stable horizon throughout
Negative: rapid turns, camera shake, motion blur, sudden cuts, flat composition, fast zoom
```

### [ ] A3 — 360 全向（验证你的工具能不能做 360）→ 存为 `test_A3_360.mp4`
```
First-person FPV view, full 360° equirectangular spherical coverage, seamless wraparound environment, omni-directional field of view, smooth forward motion, level horizon, ultra-detailed, sharp focus, 8K, photorealistic, continuous 360° surround view
```
> ⚠️ 主流 2D 视频模型基本**做不出真 360 全景**。A3 主要是验证"你的工具到底能不能"。如果出来还是普通平面片，就证实了"360 要专门的 360 模型/多视角拼接/外绘"，近期先专注 180。

---

## 实验 B — 运动与致晕（舒适度）

固定 A1 的取景，只变运动，看哪种戴头显更舒服。

### [ ] B1 — 慢速平稳（已在 A1 里）→ 复用 `test_A1_vr180.mp4`

### [ ] B2 — 完全静止运镜（对照，最不晕）→ 存为 `test_B2_static.mp4`
```
Locked-off camera position, completely static camera, no camera movement, rich depth layers (foreground/mid-ground/background), main subject centered, wide cinematic field of view, cinematic lighting, ultra-detailed, sharp focus, 8K, photorealistic
Negative: camera movement, camera shake, motion blur, sudden cuts, zooming, panning, flat composition
```

### [ ] B3（可选）— 故意快速运镜（反例，看会不会更晕）
把 A1 的 negative 去掉，prompt 里写 "fast dynamic flight, quick turns"，生成一条对照。

---

## 评分表（每条片子打分，1-5）

| 文件 | 清晰度 | 运动舒适(不晕) | FOV/覆盖够 | 景深分层(前中后景分明) | 适合立体(边缘无快速掠过) | 备注 |
|---|---|---|---|---|---|---|
| test_A1_vr180 | | | | | | |
| test_A2_fulldome | | | | | | |
| test_A3_360 | | | | | | |
| test_B2_static | | | | | | |

> 重点观察：
> - **景深分层**好不好 = 立体感的根源（前景近物 + 中景主体 + 远景背景）。
> - **边缘有没有物体快速掠过** = VR180 立体最容易"重影/晕"的地方，越少越好。
> - **A2 是不是明显比 A1 更广** = 球幕更沉浸但更费"周边像素"。

---

## 做完之后

1. 把评分最高的 1-2 条（连同文件）发给 Claude。
2. Claude 用 SeedVR2 升清 → 转 VR180 + 球幕两版 → 你戴 Quest 实测前后对比。
3. 据此定"标准 prompt 模板"，固化进 `prompt_builder` 的默认。
