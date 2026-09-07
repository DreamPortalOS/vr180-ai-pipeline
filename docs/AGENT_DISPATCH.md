# Claude/Cursor 任务调度与监控

## 在哪里看进度

- 当前任务台账：`docs/agent_tasks.json`
- 人类可读的任务计划：`docs/NEXT_TASK_PLAN.md`
- GitHub Issue：任务卡、评论、PR 和 CI 是远端事实源
- 代理 worktree：运行 `git worktree list` 查看路径、分支和 HEAD

检查命令：

```powershell
python scripts/audit_agent_dispatch.py
python scripts/audit_agent_dispatch.py --stale-hours 4
```

## 必填状态

每个代理任务必须有：Issue、独立 worktree、分支、HEAD、状态、最后检查时间、下一步和专项测试命令。状态只允许：

- `queued`
- `running`
- `blocked`
- `blocked-needs-tests`
- `ready-for-review`
- `ready-to-merge`
- `done`
- `stopped`
- `failed`

## 心跳与收工规则

1. 派发时立即登记 worktree、分支、启动时间和负责人。
2. 代理每完成一个文件或一个验证阶段，更新 `last_checked_at`，并在 GitHub Issue 留评论。
3. 超过 4 小时没有新提交、状态更新或测试结果，标记为 `stale`，停止继续消耗额度并人工检查。
4. 代理进程退出不代表任务完成；必须同时存在提交、专项测试结果和 GitHub 状态更新。
5. 代理达到轮数上限、无测试文件、测试无法收集或 worktree 脏状态异常，状态必须改为 `blocked`，禁止合并。
6. 合并前由主控运行专项测试、ruff 和完整非慢测试；合并后把 commit、CI URL 和结果回写 Issue。
7. 任务结束后保留最后一次审计记录，不删除台账条目。

## 当前快照（2026-09-07）

| Issue | 状态 | HEAD | 专项验收 |
|---|---|---|---|
| #290 | `ready-to-merge` | `ab323ad` | depth cache 27 passed |
| #291 | `blocked-needs-tests` | `567b928` | 缺少 `tests/test_calibrate_hfov.py` |
| #292 | `ready-for-review` | `04bbd7f` | e2e smoke 97 passed |

## GitHub 同步规则

- Issue 开始、阻塞、完成和移交评审时各留一条评论。
- 分支先 push，再请求 PR；不要在未同步远端的本地 worktree 上宣布完成。
- 不把 API key、模型路径或本地视频路径写入 Issue。
- 本地 `video/`、模型、缓存和 worktree 资产不提交 GitHub。
