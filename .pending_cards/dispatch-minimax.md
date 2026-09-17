You are executing overnight card issue #381 (MiniMax restore) for D:\Github\vr180-ai-pipeline.

READ: CLAUDE.md, DECISION_MINIMAY.md if present (DECISION_MINIMAX.md), docs/OVERNIGHT_2026-09-17.md.

Branch: `git switch -c agent/overnight-minimax`

Source code lives on branch feat/issue-353-minimax:
- integrations/minimax.py
- tests/test_minimax.py
- factory/seedance-style integration

Get files:
`git show origin/feat/issue-353-minimax:integrations/minimax.py > integrations/minimax.py`
`git show origin/feat/issue-353-minimax:tests/test_minimax.py > tests/test_minimax.py`

Then:
1. Wire factory registry for provider name "minimax"
2. Ensure tests use mock/httpx patches only — no network in CI
3. Add studio node OR document how video seedance-style provider=minimax works; prefer integrations first
4. ruff + pytest -m "not slow" foreground green
5. Commit paths only; push agent/overnight-minimax

Do not enable paid calls in tests. Keys via env MINIMAX_API_KEY.

Reply: Status / Summary / Files / Validation / Findings
