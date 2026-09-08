# Claude/Cursor 任务调度与监控

## 事实源只有一个：GitHub + git

**不维护任何任务台账。** 曾经存在的 `docs/agent_tasks.json`（#296 引入）在 12 小时内就与现实脱节：
把已合并的 #291 记成 `blocked-needs-tests`、#292 记成 `ready-for-review`，记录的 worktree 路径也全被清理。
两份互相矛盾的真相比一份差。现在的规则是：

- **任务状态** = GitHub Issue（`stage:*` 标签、评论）
- **交付状态** = PR（是否开着、`mergeStateStatus` 是否绿）
- **本地进度** = `git worktree list` / `git status` / 分支是否推上 origin

想知道现状就**跑脚本现场查**，不要读、也不要更新任何 JSON。

## 跑这个脚本看现状

```powershell
python scripts/audit_agent_dispatch.py
```

脚本是只读、无状态的：每次运行现场采集 `git worktree list --porcelain`、
`git status --porcelain`（逐个 worktree）、`git ls-remote --heads origin`、
`gh pr list --state open --json number,headRefName,mergeStateStatus`、
`gh issue list --state open --json number,title`，然后报四类告警：

| 告警 | 含义 | 通常的下一步 |
|---|---|---|
| `no open PR for this branch` | 有 worktree/分支，但没有开着的 PR | 开 PR，或收掉这个 worktree |
| `CI not green — <mergeStateStatus>` | PR 开着但 GitHub 不认为可合 | 看 `gh pr checks <N>`，修到绿 |
| `uncommitted changes (N modified, M untracked)` | 工作只存在于这台机器的工作区 | 提交或明确丢弃 |
| `branch not pushed to origin` | 分支只存在于本地 | `git push -u origin <branch>` |

另外会以 `[INFO]` 列出脱离分支的 worktree、已被删除但仍注册的 worktree
（提示 `git worktree prune`），以及所有开放 issue 是否已有对应分支/PR。

退出码：

- `0` — 无告警
- `1` — 至少一条告警
- `2` — `git`/`gh` 不可用（未安装、未登录、查询失败）

**`gh` 缺失或未登录时脚本硬失败（退出码 2），不会降级成空报告。**
"没有告警"必须意味着 GitHub 说一切正常，而不是我们根本没问成。所以看到退出码 2
先修工具（`gh auth login`），不要把它当成"审计通过"。

## 心跳与收工规则

1. 派发前确认 issue 有 `stage:ready`；派发后在 Issue 留评论说明已开工，不另建台账。
2. 代理每完成一个阶段，在 GitHub Issue 留评论 —— 评论时间戳就是心跳，不需要 `last_checked_at` 字段。
3. 超过 4 小时没有新提交、没有 Issue 评论、没有 PR 更新，视为 stale：停止消耗额度并人工检查。
4. 代理进程退出不代表任务完成；必须同时存在提交、专项测试结果和 PR。
5. 分支必须先 push 再宣布完成 —— 上表第 4 类告警就是为了抓这个。
6. 合并前由主控运行专项测试、ruff 和完整非慢测试；合并后把 commit、CI URL 和结果回写 Issue。
7. 任务结束后清理 worktree（`git worktree remove` + `git worktree prune`），审计脚本下次运行自然不再列出它。

## GitHub 同步规则

- Issue 开始、阻塞、完成和移交评审时各留一条评论。
- 分支先 push，再请求 PR；不要在未同步远端的本地 worktree 上宣布完成。
- 不把 API key、模型路径或本地视频路径写入 Issue。
- 本地 `video/`、模型、缓存和 worktree 资产不提交 GitHub。
