# P-1: Prompt Builder — Target-Aware Routing

## Checklist

- [x] Analyze requirements & read existing code (prompt_builder.py, test_prompt_builder.py)
- [x] Git: checkout main → pull → create `feat/P1-prompt-target-aware`
- [x] Activate .venv, install pre-commit hooks
- [ ] Modify `pipeline/prompt_builder.py`:
  - [x] Add `wrap_prompt()` with `target` param (`vr180_flight`, `fulldome_180`, `vr360_dome`)
  - [x] Create fulldome_180 template with wider FOV, relaxed negative, 3-depth-layers, stable horizon
  - [x] Create vr360_dome template with equirect/360 coverage, notes field
  - [x] Add `notes` field to return dict (default `""`)
  - [x] Refactor `wrap_prompt_for_vr180` as backward-compat alias calling `wrap_prompt(target="vr180_flight")`
- [ ] Update `tests/test_prompt_builder.py`:
  - [ ] Test alias backward compat (returns same structure, contains VR180 constraints)
  - [ ] Test target routing: fulldome_180 positive contains wider FOV
  - [ ] Test vr360_dome notes non-empty and contains "360"
  - [ ] Test unknown target falls back to vr180_flight
- [ ] Run `ruff check . && ruff format --check . && pytest` — all green
- [ ] Commit, push, open PR (never --no-verify)
