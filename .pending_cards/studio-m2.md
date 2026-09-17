Depends on Studio M1 and preferably PR #376 (dome-preview).

## Scope
- Dome conversion node (`fulldome_mapper` / CLI flags)
- VR180 conversion node (CLI subprocess, list-form, no shell=True)
- Coverage measurement + hemisphere WebGL preview
- Dual export: VR180 SBS + 4096² domemaster

## Acceptance
- [ ] Bad master (content to r≈0.6) shows low outer-ring coverage in preview
- [ ] Dome output is 4096² circular domemaster
- [ ] VR180 output is playable SBS with sv3d/st3d (or documented mock path)

Refs: PRD §6.2 D, §9, DECISION_DOME
