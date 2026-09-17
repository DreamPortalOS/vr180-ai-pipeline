You are executing overnight card P-1 / issue #384 for repo D:\Github\vr180-ai-pipeline (Windows).

READ FIRST: .pending_cards/G-11-anchor-recalibration.md and CLAUDE.md. Follow CLAUDE.md boundary locks.

Branch: `git switch -c agent/overnight-g11` then implement ONLY in:
- scripts/check_source_quality.py
- tests/test_check_source_quality.py

Do NOT touch video/ models/ .venv/. No real API. No model download.
Real-asset tests: set VR180_SEED_STILL / VR180_CANYON_STILL / VR180_SEED_CLIP to files under D:\Github\vr180-ai-pipeline\video\ if present; otherwise document skips.

Quality gate before commit:
.venv\Scripts\python.exe -m ruff check --fix scripts/check_source_quality.py tests/test_check_source_quality.py
.venv\Scripts\python.exe -m ruff format scripts/check_source_quality.py tests/test_check_source_quality.py
.venv\Scripts\python.exe -m pytest tests/test_check_source_quality.py -q
Then full: .venv\Scripts\python.exe -m pytest tests/ -m "not slow" -q  (must see passed count)

Commit with explicit paths (Conventional Commits), then `git push -u origin agent/overnight-g11`.

When done reply with:
**Status**: success|partial|failed
**Summary**: one line
**Files touched**: ...
**Validation**: test command + N passed
**Findings**: scan table summary or blockers
