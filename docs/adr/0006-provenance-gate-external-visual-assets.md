---
status: superseded by ADR-0018
---

# Provenance-gate external visual assets

The generator may use external PBR textures and HDRI lighting assets only when
their asset receipt records a permitted license, source, and file SHA. These
assets may influence RGB realism but cannot provide geometry or labels; scene
truth remains procedural and generated in the local worker.

## Considered Options

- Procedural-only materials simplify provenance but limit visual diversity and
  the requested realism.
- Untracked downloaded assets improve appearance quickly but make dataset
  redistribution and reproducibility unsafe.
