# DEV_PROCESS — 本仓开发协作机制（2026-08-21 定稿）

_Owner 拍板的固定流程；此后全部开发严格按此执行。_

## 角色

| 角色 | 谁 | 职责 |
|---|---|---|
| **Owner（Muso）** | 人 | 定方向、拍板决策（API key/参数取舍）、**阶段门真机测试**（Quest/球幕） |
| **Lead（Claude）** | Claude（本会话/巡检唤醒） | PRD/架构/任务卡编写 · 验收（代码 + 实机）· CI/CD · 阶段门交付 · 排障定位 |
| **Worker（kimi-k3）** | Claude Code CLI 挂 kimi-k3 引擎（cockpit lane 自动派发） | **全部编码与调研实现**：按任务卡开发、自测、开 PR |

### 分工红线（2026-08-31 owner 拍板）

**lead 不直接写实现代码、不手工搭环境**。凡是「要动手做」的事——写代码、装依赖、搭第三方
环境、调研某个库怎么用——**一律写成任务卡交给 Claude Code CLI（kimi-k3）执行**，lead 只负责
把要求写清楚、事后验收。

lead 手上只保留三类动作：
1. **只读诊断**：跑测试/lint、真实跑一遍现有 CLI、抽帧看画质、读日志定位根因（结论写进任务卡）；
2. **交付物产出**：用**已合并的**管线渲染样片、推 Quest、写 WORKLOG；
3. **一行级热修**（例外，需在 PR 描述写明理由）：worker 交付后实机才暴露、且修改 ≤ 几行的
   阻塞性缺陷（如版本号写错）。**超过这个规模一律回写任务卡重派。**

判据：如果一件事需要「写超过几行代码」或「装/配环境」，它就是 worker 的活，不是 lead 的活。

## 流程（一张卡的生命周期）

```
Lead 写卡（GitHub Issue, stage:ready + model:cheap + agent-ok）
  → cockpit lane（≤8 分钟轮询）领卡 → kimi-k3 在隔离 worktree 实现 + 自测 → 开 PR
  → 自动评审门：CI 绿 + AI 评审 APPROVE → squash 合并 → 关卡
  → Lead 巡检验收（见下）→ 通过 → 派下一批卡；不通过 → 开修复卡（附诊断）回到顶部
  → 一个阶段（如 G-α）全卡通过 → 写 WORKLOG 通知 Owner 真机测试
```

## Lead 验收标准（每阶段必做，mock 测试拦不住的正是这些）

1. `pytest -m "not slow"` + `ruff` 在**本机（CUDA）**全绿——CI 无 GPU，机器相关缺陷只有这里能抓。
2. **实机跑真实路径**：真渲染/真文件/真 ffmpeg（历史教训：SBS 布局错乱、NVENC 驱动不符、
   元数据未注入，全部是 CI 绿但实机炸）。
3. 产物过 `scripts/vr180_qa.py`；视频类交付抽帧肉眼核对。
4. 验收结论写进 WORKLOG（发现的缺陷 → 修复卡附字节级诊断）。

## 监控机制（自动）

- **cockpit lane**：常驻轮询派卡/评审/合并（重启：`D:\Github\_ops\start-vr180-lane.ps1`；
  日志 `D:\Github\_ops\logs\lane-DreamPortalOS--vr180-ai-pipeline*.log`）。
- **Lead 定时巡检**：Claude 定时任务每小时唤醒 lead —— 检查新合并 PR → 执行验收 →
  阶段完成则派下一批卡/通知 Owner；lane 挂了自动拉起。Owner 无需干预。

## 任务卡模板（Lead 写卡用）

```
标题：G-N: <一句话>（优先级）
## 背景        ← 为什么做，引用 PRD 章节
## 要做什么     ← 精确到文件/类/函数签名/参数默认值
## 验收标准     ← 可勾选清单；测试要求（mock 范围）；CI 约束
## 边界（Do NOT）← 不许碰什么、不许新增什么
```

规则：一卡一 PR；互改同一文件的卡**不同时** ready（后置的压 `stage:backlog`，前置合并后放行）；
实现类卡必须可在**无 GPU 无模型无 key** 的 CI 上测试（mock/注入伪后端）。

## Owner 阶段门

阶段门到达时 lead 会：渲好样片 → `adb push` 到 Quest（`/sdcard/Movies/`）→ WORKLOG 列清单。
Owner 戴机打分回填 WORKLOG，lead 据此定下一阶段方向。
