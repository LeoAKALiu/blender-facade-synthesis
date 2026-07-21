# Use fingerprint-only visual assets for internal generation

The internal Web Studio will accept locally available PBR textures and HDRI
files without collecting source or license metadata. It will automatically
record the selected asset's local identifier and content SHA in the generation
brief and dataset receipt, so a batch remains reproducible without imposing an
asset-catalog workflow. Visual assets may influence RGB appearance but never
provide semantic labels.

## Consequences

- Asset-source and license entry are intentionally out of scope for this
  internal-only tool.
- A moved, missing, or content-changed local asset prevents the affected job
  from publishing until the user explicitly confirms a new brief.
