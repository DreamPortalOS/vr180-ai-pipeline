# 双输出策略：VR180 头显 vs 投影/LED 黑暗骑乘（Dark Ride）

> lead 合成稿（2026-09-07）。依据两路独立调研：A=显示/格式/投影/同步侧，B=生成/管线适配侧。
> 每条结论标来源；**未证实**项集中在 §8。本文只定策略与分叉设计，不含实现代码。

## 0. 结论先行

1. **两种目标差别很大，但分叉点很靠后。** 源素材 → 超分 → concat → 深度 → 立体这五段**共享**；
   从「投影映射」开始各走各的（几何、立体封装、编码、元数据、QA）。
2. **"一份源两用"：部分可以。** 平稳镜头（慢推/环绕/缓降）用 **1:1、hfov≈126°、2880² 母版**两用；
   刺激镜头（压坡转弯、俯冲拉起、加速）**只为骑乘生成**——头显里视觉-前庭冲突无解，骑乘靠六轴平台补足。
   节目单按"平稳段共用 / 刺激段骑乘专属"分别生成。
3. **骑乘交付物 = 未 warp 的 domemaster（或柱面条带）帧序列 + 相机轨迹 CSV + 分轨音频**；
   切片/warp/blend/色彩统一由场馆的标定与媒体服务器（Pixera/7thSense/Disguise/Watchout）完成，
   **不要在内容侧预 warp**（IMERSA 规范 + 行业指南一致）。
4. **骑乘默认单目 2D、60p 起步。** 飞行影院主流不戴眼镜（Soarin'/FlyOver/i-Ride）；立体做显式开关，
   且必须是**波长分光（Infitec/6P）+ 左右眼两套鱼眼序列**，偏光在球幕不可用。
5. **六轴同步没有公开通用标准**：交付"每帧相机轨迹 sidecar（t, xyz, rpy）"给厂商编排员，
   **不要自造运动码格式**；同步靠 SMPTE LTC/PTP 主钟，帧 0 对齐。
6. **仓内 `--projection fulldome` 是正确起点但有三处硬伤**（B 报告，代码取证）：缺 `alpha_mask`+黑底（与 #255 白边同根因）、
   `iv_fov` 用线性比例而非针孔公式、成片静音；且**从未在球幕实机验证**。

## 1. 两种目标的差异对照

| 维度 | VR180（Quest 3） | 投影/LED 骑乘 | 来源 |
|---|---|---|---|
| 几何 | 每眼 180° 半等距（hequirect）SBS；观众可转头 | **domemaster**（1:1 画布内嵌 180/165/210/220° 等距方位鱼眼，圆心=天顶或倾斜球幕前方）/ 环幕柱面条带 / 每屏透视切片；观众固定朝向≈球心 | A §1 |
| 分辨率 | 2880²/眼（16 px/度） | 4096² 下限；Cosm LED 画布 8192²；Sphere 16K² | A §2.2 |
| 帧率 | 30/60 | **60p 原生**（Cosm/FoP/King Kong），投影机可 120/眼；24→60 直接插帧有伪影风险 | A §2.2 |
| 立体 | 必有，头中心 IPD 视差 | 默认无；有则 L/R 两套 ODS 鱼眼 + 波长分光；行业在拆 3D（King Kong 2026-03） | A §2.3 |
| 元数据 | `sv3d`+`st3d` | 无标准；slate + sidecar/文件名约定 | A §4 |
| 容器 | H.264/H.265 mp4 | 帧序列母版（PNG/TGA/TIFF16/DPX/EXR）+ HAP Q/NotchLC 播放 | A §4.1 |
| 色彩 | Rec.709 / 头显自适应 | Rec.709 γ2.4（Cosm 原生）；投影球幕低对比、忌亮背景；禁"dome grads" | A §4.1 |
| 同步 | 无 | show controller 主钟（LTC/PTP/genlock）；平台 60–400 Hz 位置帧 | A §3 |
| 运动数据 | 无 | 相机轨迹 sidecar → 厂商私有编排工具 | A §3.2, B §3.4 |
| 舒适度 | 禁视觉自运动加速/旋转（无前庭反馈） | 平台补前庭线索，允许更强 onset；受行程（±15°、±0.5 G 量级）与 washout 假线索约束 | A §3.3 |
| 源 FOV | ≥110°×96° 真内容 + 165→180 渐隐 | 只需覆盖屏幕张角（120–180°），无需头动余量 | B §2.2 |
| 相机运动 | 慢、恒速、地平线水平 | 允许俯冲/压坡/加速，但**低频主导、可预测** | B §2.2 |

## 2. 管线分叉设计

```
源素材 ─ 0 超分(SeedVR2) ─ 0b concat ─ 1 深度(DepthCrafter+cache) ─ 2 立体(L/R)
                                             │                            │
                                             └─[新] 1b 自运动估计 → 六轴曲线 CSV   （骑乘专属，只读共享产物）
═══════════════════════════════════ 分叉点 = 3 投影映射 ═══════════════════════════════════
 VR180 : hequirect 180×180 + alpha/黑底 + SBS + 165→180 羽化 → HEVC mp4 + sv3d/st3d → vr180_qa
 骑乘  : [新] ScreenMapper: v360 output=fisheye(球幕)/cylindrical(环幕)/flat(平面) + yaw/pitch 偏置
         → 单目单 pass（或 L/R 各一 pass）→ 帧序列母版(+HAP Q) → 不注元数据，写 sidecar → ride_qa
```

| 阶段 | VR180 | 骑乘 | 共享？ |
|---|---|---|---|
| 0 超分 / 0b concat | 同 | 同（骑乘要求段间相机状态连续） | ✅ |
| 1 深度 | DepthCrafter + cache | 可选（2.5D/被动立体/运动估计） | ✅ |
| 1b 运动曲线 | 无 | 光流+深度 → 6 参 LSQ → washout → CSV | ❌ 骑乘专属 |
| 2 立体 | Quest 舒适预设 | 单目跳过；被动立体复用 L/R + 新 `screen_passive` 预设 | ✅ 引擎 / ❌ 预设 |
| **3 投影** | hequirect SBS | **ScreenMapper** fisheye/cylindrical/flat | ❌ 分叉，复用 `_FfmpegV360Pipe`、`_BLACK_COMPOSITE`、`_calc_vertical_fov` |
| 3.5 羽化 | 165→180 默认 | 通常不需要 | 部分 |
| 4 编码 | HEVC + GOP 预设 + NVENC | 母版级帧序列 / 高码率；单目码率减半 | ✅ 机制 / ❌ 参数 |
| 5 元数据 | 注 sv3d/st3d | 不注；sidecar `projection=fisheye/cylindrical` | ❌ |
| 6 QA/命名 | `vr180_qa` | `--expect ride` 或 `ride_qa`；route `ride` | ❌ |

## 3. 骑乘输出的具体映射（已本地用 `testsrc2` 验证参数可被 ffmpeg 接受，A §5.3）

- 球幕 / 倾斜穹顶：`v360=input=flat:output=fisheye:ih_fov=<hfov>:iv_fov=<针孔vfov>:h_fov=180:v_fov=180:pitch=<倾角>:w=4096:h=4096:alpha_mask=1` → 黑底合成。
- 环幕：`v360=input=flat:output=cylindrical:ih_fov=<hfov>:iv_fov=<vfov>:h_fov=270:v_fov=60:w=5184:h=1152:alpha_mask=1` → 黑底合成。
- 每屏透视切片：`output=flat:h_fov=70:v_fov=45:yaw=<每屏偏航>`，视点固定于车辆。
- 立体球幕：`output=dfisheye` 一次出左右，或每眼一 pass 再 `hstack/vstack`。
- 已有 VR180 成片可直接再投影：`input=hequirect:in_stereo=sbs:output=fisheye:out_stereo=sbs`（sbs 时 `w/h` 是每眼尺寸）。

## 4. 生成侧差异

- 骑乘 prompt：横向优先宽画幅、显式**运动时间线**（"3 秒压坡左转后改平"），禁高频抖动/往复；保留"一镜到底 / 光照稳定"。
- 仓内 `prompt_builder.py` 已有 `fulldome_180` target（B §4，带行号），但 `generate.py` 把 target 写死为 `vr180_flight`（`generate.py:357`）——需加 `--target`。
- `prompt_library.py` 的 `_CONSTRAINT_TAIL`（"slow and constant-speed"）对骑乘不适用，需 `_RIDE_CONSTRAINT_TAIL` + 4–5 条骑乘模板 + `target` 字段。
- Provider：Seedance 支持 `1:1/16:9/21:9` 等、4k 1:1=2880²（需 `--model` 标准档）；Veo 只 16:9/9:16；Kling 支持 1:1。

## 5. 最小改动清单（每条一张卡；文件互不交叉的可并行）

| 卡 | 内容 | 只许改 | 依赖 |
|---|---|---|---|
| R-A ScreenMapper | 新建 `pipeline/screen_mapper.py`：fisheye/cylindrical/flat、fov/yaw/pitch/roll、`alpha_mask`+黑底、针孔 vfov、整文件单 pass + 逐帧 pipe、音频透传；顺带修 fulldome 三处硬伤 | `pipeline/screen_mapper.py`、`tests/test_screen_mapper.py`（可选 `pipeline/fulldome_mapper.py`） | 无 |
| R-D 合同扩展 | sidecar 加 `cylindrical` 与 `top_bottom/frame_seq/separate_lr`；naming 加 route `ride`；`vr180_qa --expect {vr180,fulldome,ride}` | `pipeline/sidecar.py`、`pipeline/naming.py`、`scripts/vr180_qa.py` + 测试 | 无 |
| R-E 运动曲线 | 新建 `pipeline/egomotion.py`（光流 + 6 参 LSQ + washout）+ `scripts/motion_curve.py`（→ CSV `t_sec,frame,surge,sway,heave,roll,pitch,yaw`，归一化 ±1）；合成流场做 CI 测试 | 两个新文件 + 测试 | 无 |
| R-F 生成侧 | `prompt_library` 加 `target`/`_RIDE_CONSTRAINT_TAIL`/骑乘模板；`prompt_builder` 加 `ride_screen`；`generate.py --target` | 三文件 + 测试 | 无 |
| R-B CLI 接线 | `--projection {vr180,fulldome,ride}` + `--ride-*` 参数；投影分发移到 streaming 分支之前；fulldome/ride 补音频；补 fulldome CLI 测试 | `scripts/run_pipeline.py` + 测试 | R-A、R-D |
| R-C 被动立体封装 | `pipeline/stereo_pack.py`：L/R → separate/sbs/tb/frameseq；`comfort_presets` 加 `screen_passive` | 两文件 + 测试 | R-A |
| R-G 模板曲线 | 由 `camera_motion` 标签生成关键帧样条，与 R-E 估计做时间对齐 | `egomotion.py`、`motion_curve.py` | R-E、R-F |

**顺序**：R-A ∥ R-D ∥ R-E ∥ R-F → R-B → R-C → R-G。

## 6. 骑乘交付规格（建议默认）

| 项 | 默认 | 备注 |
|---|---|---|
| 母版 | 4096² PNG 帧序列（8192² 作开关）+ slate | 8192² 16-bit TIFF ≈ 400 MB/帧，60p 一分钟 ≈ 1.4 TB |
| 播放格式（可选） | HAP Q `.mov`（ffmpeg `-c:v hap -format hap_q`） | Pixera/Watchout/7thSense 直接播 |
| 帧率 | 60p | 24p 源需光流插帧，先加运动模糊再插 |
| 色彩 | Rec.709 γ2.4，sidecar 记录 | 不做 dome grads |
| 立体 | 关 | 开则 `_L/_R` 两套鱼眼序列 |
| 同步 | 相机轨迹 CSV（60 Hz，帧 0 对齐）+ 分轨 WAV | 运动码转换交厂商 |
| 安全区 | IMERSA safe-action：经度 ±50°、纬度 10–60°；sweet spot 30–40° | QA 里叠加检查主体位置 |

## 7. 舒适度：两套阈值

- 无平台（VR180）：禁持续加速/旋转，地平线水平，慢恒速。
- 有平台（骑乘）：允许更强 onset 与短时旋转；**持续旋转/长时间加速仍受限**（平台行程 + washout 回中假线索）；
  相机轨迹与平台曲线**同源设计**（Brogent"先定 gondola 怎么飞"）。QA 加"每秒角速度/加速度/旋转持续时间"阈值告警，按目标切换。

## 8. 未证实 / 需向厂商确认

- 屏幕硬件形态（球幕/环幕/平面 LED、倾角、投影机数、warp 由谁做）**未定**，决定 ScreenMapper 全部参数。
- Simtec/Brogent/Simworx/D-BOX/4DX 运动曲线文件格式（列/单位/坐标系/采样率）与 "`.dbx`" 扩展名——需 ICD/NDA。
- ffmpeg `v360` 的 `h_offset/v_offset` 对 fisheye 是否生效；`cylindrical` 非方形输出下 `d_fov` 几何。
- fulldome 边缘涂抹缺陷从 #255 根因推断，未跑实测。
- AI 视频自运动估计精度（单目尺度歧义、场景内运动物体污染、DepthCrafter 窗口归一化跳变）——做原型才知道。
- 银幕被动立体的舒适视差预算；Gemini 网页端画幅选项；中国 LED 球幕厂商 120fps/HDR 宣称。
