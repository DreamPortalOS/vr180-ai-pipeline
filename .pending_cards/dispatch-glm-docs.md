You are executing pipeline/docs card for D:\Github\vr180-ai-pipeline.

READ: CLAUDE.md, docs/APACHE_CHAIN_FEASIBILITY.md, docs/LITELLM_LOCAL_NOTES.md, docs/PRD-node-studio.md.

Branch: git switch -c agent/glm-docs-merge

Tasks:
1) Update docs/PRD-node-studio.md §8 with live LiteLLM gateway notes: models glm-5.2/auto/internlm-s2 OK; deepseek-pro often 429; never paste secrets.
2) Update docs/APACHE_CHAIN_FEASIBILITY.md: Studio VR180 backend=apache already shipped (convert.vr180).
3) README: short「Immersive Node Studio」section — how to run python -m studio.server, production template, local litellm settings.
4) Do NOT invent untested claims; cite what exists in code/tests.

ruff not required for md-only; if you touch .py run tests.
Commit paths; push agent/glm-docs-merge.

Reply: **Status** / **Summary** / **Files** / **Validation**
