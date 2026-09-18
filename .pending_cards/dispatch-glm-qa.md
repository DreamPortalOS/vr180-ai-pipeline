You are executing quality gate card for D:\Github\vr180-ai-pipeline.

Branch: git switch -c agent/glm-qa-suite

Tasks:
1) Run foreground: .venv\Scripts\python.exe -m pytest tests/ -m "not slow" -q
2) If any failure, fix ONLY tests or minimal code bugs you can prove; do not disable tests.
3) Run studio live smoke: start .venv\Scripts\python.exe -m studio.server, POST /api/run with production template shrunk sizes, assert gallery present.
4) Write docs/QA_SMOKE_2026-09-18.md with test counts and smoke result.

Commit any code/doc changes with paths; push agent/glm-qa-suite.
Do not commit secrets. Do not write into video/.

Reply: **Status** / **Summary** / **Files** / **Validation** (must include exact pytest passed line)
