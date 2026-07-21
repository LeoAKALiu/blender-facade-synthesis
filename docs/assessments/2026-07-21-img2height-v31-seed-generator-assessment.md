# img2height V3.1 Synthetic-Facade Seed Generator Assessment

## Assessment outcome

The specified V3.1 generator is suitable as a locally absorbed **seed
generator**, not as the finished Web Studio backend. Its deterministic 3D scene
specification, window-instance geometry, projected floor regions, raster
labels, metadata validator, and contact-sheet/validation utilities are useful
starting points. The new project must own and evolve the imported code; it will
not keep an upstream dependency.

## Verified evidence

- The V3-focused pytest selection passed: **27 passed**.
- A one-sample direct Blender 4.2.1 render at 320x240 completed with
  `rendered_with_blender_count: 1`, `projection_fallback_count: 0`, and the
  upstream dataset validator reported no errors.
- The source's original command launched through the local BlenderProc 2.8.0
  wrapper did not reach rendering. It imports NumPy before importing and
  initializing BlenderProc, so Blender's Python 3.11 tried to load Anaconda
  Python 3.13 NumPy and failed with a NumPy C-extension ABI import error.
- BlenderProc 2.8.0 itself is healthy on this machine: a minimal official-style
  script (`import blenderproc as bproc`; `bproc.init()`) initialized the Metal
  renderer and imported Blender's bundled NumPy 1.24.3 successfully.
- A thin bootstrap that calls `bproc.init()` first, restores the project source
  path after that initialization, and only then imports the V3.1 runner
  completed a one-sample BlenderProc render at 320x240. The upstream validator
  reported no errors and the run reported zero projection-fallback samples.
- The successful real render visibly demonstrates 3D facade/window geometry,
  but uses simple flat materials and lighting; it is structurally useful rather
  than close to the requested post-2000 Chinese urban visual domain.

## Reuse candidates

- `synthetic/facade_mvp/generator/blender_scene.py` — deterministic base scene
  definitions, camera/projection helpers, and Blender RGB capture.
- `synthetic/facade_mvp/generator/blender_scene_v3.py` and
  `structure_scene.py` — three-dimensional facade grammar, recessed windows,
  frames, sills, balconies, podium logic, and per-window/floor identities.
- `projection.py`, `render_outputs.py`, and `render_outputs_v3.py` — scene
  truth projected into window masks, instance IDs, floorline/roofline/groundline
  heatmaps, depth, normal arrays, and metadata.
- `schema.py`, `validate_dataset.py`, and `preview_contact_sheet.py` — a useful
  pattern for strict validation and visual inspection, to be replaced with this
  project's expanded task-package contracts.

## Required replacement or extension

- Replace the V3 command runner with a thin BlenderProc bootstrap. It must call
  `bproc.init()` before importing NumPy or project generator modules, then add
  the project import root and invoke the generator. This fixes the ABI path
  leak without changing the global BlenderProc installation. The worker must
  also disable projection fallback and record actual backend/fallback state per
  sample. The old runner reported `render_backend` as
  `blender_with_projection_fallback` even on a direct real render with zero
  fallback samples, while its final manifest omitted those fields entirely.
- Replace the strict `synthetic_facade_mvp/v1` metadata schema. It has no
  building-use class, component-semantic masks, task curricula, generation
  brief, asset fingerprint, dataset receipt, or task-package manifests.
- Add the six agreed task-package contracts: window instances/count, floorline
  heatmap with polylines, visible floor count, four-class building use,
  component semantic segmentation, and package QA/release evidence.
- Add the agreed Chinese post-2000 facade grammar, daylight profiles, full
  facade view family, light controlled occlusion, foundation assets, optional
  local PBR/HDRI inputs, and task-specific visibility gates.
- Add the local Web Studio, serialized resumable worker, explicit review and
  publication lifecycle.

## Decision boundary

The project may move the reusable generator modules directly into this
repository after the implementation design is approved. It must not copy the
old runner unchanged or publish projection-fallback output as Blender training
data.
