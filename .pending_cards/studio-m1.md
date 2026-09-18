Depends on Studio M0.

## Scope
- LLM polish node via configurable LiteLLM base_url/model/api_key
- SenseNova U1 feasibility card (model list, free tier, I2V?) — document result even if unsupported
- Seedance text/image-to-video node + cost estimate (mock in CI)
- Source quality check node wrapping `scripts/check_source_quality.py`

## Acceptance
- [ ] Polish node returns rewritten prompt in canvas
- [ ] SenseNova decision recorded in PRD appendix
- [ ] Seedance node shows pre-submit cost estimate; CI uses mock provider
- [ ] Quality node reproduces script judgments on fixtures

Refs: PRD §8, §8.1
