# 今晚连续开发计划 — 2026-09-17 起 10 小时

> 编排：MiMo 主会话 + Claude Code CLI（`-p` / `--bg`，Windows 原生，无 tmux）
> 主线 PR：[#380](https://github.com/DreamPortalOS/vr180-ai-pipeline/pull/380) `feat/node-studio-m0`
> 原则：**先推分支再求完美** · 禁 `shell=True` · 测试 mock · 前台看全绿再 commit

---

## 0. 当前未完成任务盘点（20:00 基线）

| ID | 标题 | 状态 | 今晚策略 |
|---|---|---|---|
| #380 / #377–379 | Studio M0–M2 + 生产流程 + 3D | OPEN PR，代码已在分支 | **叠 M3/UX**，不阻塞等合并 |
| #376 | dome-preview WebGL | OPEN PR，可合并 | 记入合并队列；Studio 已内嵌同源算法 |
| #372 G-12 | anchor 面积门槛 VR180 反了 | OPEN | 派 Claude 调研+提案，不盲目改阈值 |
| #368 L-1 | Apache 链路顶替 DepthCrafter | OPEN | 派 Claude 做可行性笔记+测试骨架 |
| pending G-11 | texture window / barrier 重标 | 卡片就绪 | **派 Claude 按卡执行**（高优先级管线质量） |
| T2/T6 | 分支清理 | 隔离会话禁删 | 主检出跑 `tools/cleanup_branches.ps1`（人工/授权 shell） |
| SenseNova U1 | 免费档验证 | 未查实 | 文档+接入骨架；无 key 则 mock |
| MiniMax | #373 后 revert | 代码在历史分支 | M3 恢复 provider + 测试 |

---

## 1. 10 小时时钟（建议 20:00 → 06:00，可按实际开工平移）

| 时段 | 窗口 | 主线（Studio） | 并行（Claude CLI） | 产出门 |
|---|---|---|---|---|
| **H0** | +0:00–0:30 | 计划落盘、开卡、环境自检 | 启动 2–3 个 `--bg` 代理 | 计划文件 + agents id |
| **H1** | +0:30–1:30 | Studio 分镜图廊 + 运行缩略图 | **G-11** anchor 重标扫描 | Studio 测试绿；G-11 分支推送 |
| **H2** | +1:30–2:30 | 节点参数预览 / contact sheet 进检查器 | **#368** Apache 可行性笔记 | PRD 附录更新 |
| **H3** | +2:30–3:30 | **MiniMax 恢复**（从 `feat/issue-353-minimax`） | G-11 完成扫描表 + 测试 | MiniMax 单测 mock 绿 |
| **H4** | +3:30–4:30 | LiteLLM/SenseNova 配置 UI + 节点 | **#372** G-12 数据分析提案 | 配置不进 git 的文档 |
| **H5** | +4:30–5:30 | 脏节点重跑 + 工程磁盘保存 | 代理汇合、修红测 | `/api/run` 二次 cache hit |
| **H6** | +5:30–6:30 | SeedVR2 节点（CUDA 可选，默认 mock） | 清理僵尸分支脚本执行结果 | 节点注册 + 测试 |
| **H7** | +6:30–7:30 | 资产库/最近工程/导出目录配置 | 代码评审 `claude -p` review PR #380 | Review comments 清单 |
| **H8** | +7:30–8:30 | 全量 pytest + ruff + UI 截图回归 | 修 review 点 | **前台见 N passed** |
| **H9** | +8:30–9:30 | 文档：PRD/DEV_GUIDE/README Studio 章节 | 更新 issue 勾选 | 文档 PR 可审 |
| **H10** | +9:30–10:00 | 交接报告、未完成清单、明早第一卡 | 停止 agents、收集 session | `docs/OVERNIGHT_HANDOFF.md` |

---

## 2. 任务卡（今晚实现范围）

### S-1 Studio UX：分镜图廊与缩略图
- 运行后展示 `image.batch_stills` 的 contact sheet 与逐镜路径
- 检查器/结果区可点开路径说明
- 验收：生产流程跑完后 UI 能看到 sheet 路径与 shots 列表

### S-2 MiniMax 恢复（M3）
- 从 `origin/feat/issue-353-minimax` 取回 `integrations/minimax.py` + 测试
- factory 注册；Studio `video.seedance` 同级 provider 或新节点 `video.minimax`
- CI 必须 mock；费用表并入节点 meta
- 验收：`pytest tests/test_minimax.py` 等价用例绿（mock）

### S-3 LiteLLM / SenseNova 配置面
- 环境变量 + `studio_settings.json` 说明写进 PRD 附录
- 节点参数校验错误信息可读
- 验收：缺 key 时报错含配置提示；mock 润色仍可用

### S-4 脏子图重跑
- `run_graph(only_downstream_of=)` 实现过滤 skip
- 验收：改 video 参数后仅下游重算（cache 其余 hit）

### P-1 G-11（卡）anchor 常量重标
- 仅 `check_source_quality` + 测试
- 窗口×迭代扫描表；分离度 ≥55×；变异检验
- 真实素材 env 指向主检出 `video/`

### P-2 #368 Apache 链路笔记
- 不实现重模型；输出可行性矩阵 + mock 测试骨架
- 文档：`docs/APACHE_CHAIN_FEASIBILITY.md`

### P-3 #372 G-12 调研
- 读现有 anchor 面积逻辑与 owner 反馈
- 提案：鱼眼 vs VR180 分档门槛，附数据；**不直接改生产阈值** 除非有测试

---

## 3. Claude Code CLI 编排（Windows）

```powershell
# 状态
claude auth status
claude agents --json --all

# 后台任务模板（在独立 worktree 或明确分支上）
claude --bg "在分支 agent/overnight-g11 上按 .pending_cards/G-11-anchor-recalibration.md 完成…"
claude agents --json --all
claude logs <id>
claude attach <id>   # 需要时

# 一次性任务
claude -p "Review PR #380 for bugs" --max-turns 5 --allowedTools 'Read,Bash' --output-format json
```

**约束（写入每个代理 prompt）：**
1. 禁止 `git commit --no-verify`
2. 提交带路径；`git push -u origin <branch>` 尽早
3. 测试不调真实 API / 不下载模型 / 不写 `video/`
4. 完成后输出：Status / Summary / Files / 校验命令与结果
5. 只动卡片允许的文件

---

## 4. 优先级（若时间不够）

1. **P-1 G-11**（管线质量，卡片完整）
2. **S-2 MiniMax 恢复**（付费生成备选，代码已存在）
3. **S-4 脏重跑**（Studio 可用性）
4. **S-1 UX 图廊**
5. P-2 / P-3 文档调研
6. S-3 配置面打磨

---

## 5. 质量门（每轮合并前）

```powershell
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m ruff format --check .
.venv\Scripts\python.exe -m pytest tests/ -m "not slow" -q
```

Studio UI：`python -m studio.server` → 生产流程运行 → 截图存 `studio/static/_qa/`（gitignore）。

---

## 6. 交接物

- 更新本文件进度列
- `docs/OVERNIGHT_HANDOFF.md`：完成/未完成/阻塞/明早第一卡
- GitHub：PR #380 新 commit；相关 issue 评论勾选
- agents：`claude stop` / `rm` 清理
