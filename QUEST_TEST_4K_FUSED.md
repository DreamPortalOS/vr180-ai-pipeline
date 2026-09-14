# 🥽 Quest 融合测试单 · 4K 三主体对照（2026-09-12，最新）

> 主干 `3d6de9e` · 本单融合了 `QUEST_TEST_1X1.md` / `QUEST_TEST_1X1_R2.md` /
> `QUEST_TEST_FEATHER.md` / `ACCEPTANCE_2026-09-09.md` 四份旧单的**未决问题**。
> 旧单结论已沉淀进 `SOURCE_RULES.md`，不再单独使用。

## 0. 本次测什么（一句话）

三条 **4K / 10s / 5760×2880 SBS / sv3d+st3d QA 全绿** 的同管线成片，
只有**锚点主体**不同。全部走 `feather` 默认（165→180° 渐隐）+
`--input-projection fisheye --fisheye-fov 180`（黑区 4.0%，安全转头 ±38°）。

## 1. 三条样片（都在 `video/`，已验证）

| 编号 | 文件 | 源（只读对照） | 规格确认 |
|---|---|---|---|
| **F-plain** | `quest_4k_fe180.mp4`（56MB） | `gen_1x1_4k.mp4`（2880² 10bit） | 5760×2880 hevc 24fps 10.0s，QA 6/6 pass |
| **F-crystal** | `quest_4k_crystal_fe180.mp4`（62MB） | `gen_1x1_4k_crystal.mp4` + `seed_1x1_crystal.png` | 同上，QA 6/6 pass。⚠️ **透明形态已判出局**（SOURCE_RULES 六），此条不再测 |
| **F-drone** | `quest_4k_anchor_fe180.mp4`（61MB） | ⚠️ 标签纠正：源是 `gen_1x1_4k_anchor.mp4`（鸟 prompt V2 系），不是 `gen_1x1_4k_drone.mp4`。**画面是鸟，不用重测**。真无人机源 `gen_1x1_4k_drone.mp4`（V4，9/9）尚无 quest 成片 | 同上，QA 6/6 pass |

> ⚠️ 本单 Q1/Q2 已被 owner 反馈替代（见 §3）：F-drone 实为旧鸟、F-crystal 透明出局、
> Q3/Q4/Q5 已通过。**不要再按本单旧 Q1–Q5 打扰 owner**；新测试只测 V4 无人机成片。
>
> 📌 G-7（2026-09-12）教训：转码时**不要手动传 `--output-width/height`**。
> `--quality high` 档默认每眼 2880²（总 5760×2880）；手动传 5760/2880 会被当作每眼尺寸，
> 出 11520×2880，QA 判 `width ≠ 2×height` 直接 fail。成片参数以 `quest_4k_anchor_fe180.json`
> 为准：preset=source + quality=high + fisheye-fov 180 + h265 24fps，不加分辨率覆盖。

推头显（二选一）：
```
adb push video/quest_4k_fe180.mp4 video/quest_4k_crystal_fe180.mp4 video/quest_4k_anchor_fe180.mp4 /sdcard/Movies/
```

## 2. 请回答（核心 5 问，预计 10 分钟）

**Q1. 三条里哪条最能待得住？**（只能选一个）
- [ ] F-plain（无主体纯风景） — [ ] F-crystal（发光水晶） — [ ] F-drone（无人机跟拍）

**Q2. 主体有没有崩？**（逐条打勾）
- F-crystal：[ ] 形状稳定 [ ] 透明空洞/变形/融化（有则描述：____）
- F-drone：[ ] 机身笔直刚性 [ ] 机臂弯曲/机身变形/旋翼异常（有则描述：____）

**Q3. 远景连贯吗？**（对应 SOURCE_RULES 三：白出口 vs 有景物）
- [ ] 远景有具体景物、推进连贯 — [ ] 远处出现白色无细节区/跳变（在哪条第几秒：____）

**Q4. 河水/植被动了吗？**（对应 SOURCE_RULES 四：环境必须写"动"）
- [ ] 河水奔流、浪花翻涌 — [ ] 河水静止像贴图（在哪条：____）

**Q5. 边缘与转头**（对应 feather 定版 + 180 摊满）
- 转头约 ____° 看到黑边（若看不到写"转头到底没看到"）
- 边缘渐隐：[ ] 自然 [ ] 能看到一圈渐变环 [ ] 其它：____

## 3. 你的反馈写这里

```
Q1 =
Q2 crystal =
Q2 drone =
Q3 =
Q4 =
Q5 =
其它（晕/立体感/清晰度，想到什么写什么）=
```

### ✅ owner 2026-09-12 反馈（已吸纳，不重测）

- F-drone（`quest_4k_anchor_fe180.mp4`）画面是鸟不是无人机；F-crystal 与鸟有透明合成
  bug 无法使用；Q3/Q4/Q5 之前已回过、无问题。
- 结论：旧三条不再测。Q3/Q4/Q5 通过归档；Q1/Q2 等 V4 无人机成片出来后只测新片。

## 4. 旧单结论（已定版，不用重测，只备查）

- 铺球：`fisheye-fov 180` 摊满（4.0%黑区，±38°）；`src-hfov 160` 矩形拉伸已否决；120忠实几何黑边60.9%已否决。
- 同源对照：拉伸会把仰角×1.9，"像在地底"是取景问题不是拉伸必然代价；`--pitch` 已就位（#323/#324）。
- 边缘：feather（纯黑+165→180°渐隐）定版；accum 废弃（重复树冠/同心弧纹）。
- 眩晕：横滚真值仅 1–2°（21°是ORB累积假象），不是眩晕主因；平移/俯仰测量待开发。

## 5. 测完之后我做什么（对应关系）

| 你的反馈 | 我的动作 |
|---|---|
| 某主体崩坏 | 按"刚体优先"换主体重生成（水晶/无人机二选一），不碰管线 |
| 远景白区/不连贯 | 回 prompt 加远景景物约束（SOURCE_RULES 三） |
| 河水静止 | 回 prompt 逐项写动效（SOURCE_RULES 四） |
| 边缘可见/渐变环刺眼 | 开管线卡调 feather 曲线，不重生成 |
| 都好、无短板 | 延长到 15–25s 多段拼接（`segment_concat` 需补 CLI 卡） |

> 2026-09-12 状态：远景（Q3）/河水（Q4）/边缘（Q5）owner 已确认通过，不再开卡；
> 主体线收敛到 **V4 无人机一条**（水晶透明出局、鸟出局），见 §D 新卡 G-7。
