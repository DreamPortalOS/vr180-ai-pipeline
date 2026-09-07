# 下一阶段开发任务计划

更新：2026-09-07

## 当前闸门

1. **1:1 真前进素材**：owner 先生成 480p/5s draft，确认持续前进、宽视角和主体不出戏；未通过前不消耗 4k 配额。
2. **代码质量**：所有任务必须有 CPU-only/mock 测试，不调用真实 API、不下载模型、不写入 `video/`。
3. **合并条件**：任务分支必须包含实现、回归测试、专项测试通过、ruff 通过，并在合并前通过完整 `pytest tests/ -m "not slow" -q`。

## 任务队列

### #290 深度缓存 mtime

状态：**已完成，待合并/同步到主线**。

验收：`tests/test_depth_cache.py` 已通过 27 项。提交为 `debccd3` / `ab323ad`，变更只涉及缓存 freshness 回归测试。

动作：将该提交纳入主线合并候选；合并后跑完整非慢测试。

### #292 10-bit HEVC e2e

状态：**已完成，待合并/同步到主线**。

验收：在 `feat/issue-292-e2e-10bit` worktree 中，`tests/test_e2e_smoke.py` 通过 97 项；相关 ruff 通过。提交链为 `7c81f97`、`04bbd7f`。

动作：先审阅测试 diff，确认只生成小型合成样本；合并后跑完整非慢测试。真实 4k Seedance 素材和 Quest 播放属于后续人工验收，不由 CI 替代。

### #291 calibrate_hfov

状态：**返工，不可合并**。

当前候选提交 `567b928` 只有 `scripts/calibrate_hfov.py`，没有 `tests/test_calibrate_hfov.py`；专项测试无法收集。工具还需要明确输入假设、失败边界和与 `run_pipeline --src-hfov` 的使用契约。

返工要求：

- 补充纯 CPU 单元测试，至少覆盖候选 hfov 计算、非法输入、ffmpeg/ffprobe 失败和 JSON/文本输出。
- 使用 list-form subprocess，禁止 `shell=True`。
- 明确这是“辅助校准建议”而非无条件自动真值；输出建议值、置信度/评分和人工复核提示。
- CLI 帮助中给出示例，并说明真实素材仍需抽帧复核。
- 专项测试和 ruff 通过后，才进入合并候选。

### 1:1 draft 验收

状态：**等待 owner 素材**。

输入：`GEN_TASK_1x1.md` 和 `GEN_PROMPT_1x1.txt`。产物放 `video/`，不得提交 Git。

验收顺序：尺寸/编码检查 → 人工确认真前进与宽覆盖 → #291 hfov 辅助校准 → 默认 feather 管线 → Quest 真机评分 → 决定是否生成 4k/10-15s。

## 代理调度规则

- Claude/Cursor 负责：窄测试修复、CPU-only 工具、静态分析、文档索引和 mock e2e。
- 主控负责：任务拆分、依赖排序、分支状态、合并门、完整 CI 和最终验收。
- 代理必须在独立 worktree 工作；完成后提交 conventional commit，并报告测试命令和结果。
- 代理达到轮数上限或没有测试交付时，状态记为“未完成”，不根据提交标题直接合并。
- 真实 API、模型推理、4k 长视频和 Quest 主观体验都不能由 CI 或代理结果代替。

## 下一步顺序

`#290 合并候选 -> #292 合并候选 -> #291 返工并专项验收 -> 480p 1:1 draft 人工闸门 -> hfov 校准 -> 4k/10-bit 小样 -> 完整管线 -> Quest 验收`
