# Synthetic Facade Training-Package Studio v1

## Problem Statement

The team needs internally produced, training-ready synthetic datasets for
facade analysis without hand-authoring labels or accepting non-rendered label
fallbacks. Existing V3.1 code provides a useful deterministic facade seed but
is not a production data studio: it lacks a BlenderProc-compliant launch path,
task-specific package contracts, the target visual domain, local job control,
and human publication control.

## Solution

Build a local Web Studio backed by one local BlenderProc Worker. A user creates
and explicitly confirms an immutable Generation Brief for one Task Dataset; the
worker renders and validates its Facade Samples one at a time; the user reviews
the package and explicitly publishes it. The deliverable is a Training Package
for exactly one learning target, with scene-truth labels, split manifests,
validation evidence, and an immutable Dataset Receipt.

## User Stories

1. As a dataset producer, I want to select one Task Dataset at a time, so that
   every published package has a single unambiguous training target.
2. As a dataset producer, I want to confirm the exact Output Target before a
   job starts, so that a completed package contains the requested sample count.
3. As a dataset producer, I want to adjust the train, validation, and test
   ratio, so that each Task Curriculum fits the intended experiment.
4. As a dataset producer, I want Building Recipe splits to be fixed before
   rendering, so that related views cannot leak between splits.
5. As a dataset producer, I want to choose the Building Use distribution, so
   that residential, office, commercial, and mixed-use data are intentional.
6. As a dataset producer, I want to choose a full View Family, so that frontal,
   light/medium-oblique, and strong-oblique facade views are represented.
7. As a dataset producer, I want to choose daylight conditions and controls,
   so that sunny and overcast conditions are diverse yet reproducible.
8. As a dataset producer, I want light controlled foreground occlusion, so
   that models learn realistic but bounded visibility variation.
9. As a dataset producer, I want Chinese post-2000 urban facade grammar and
   materials, so that the synthetic visual domain is relevant to target images.
10. As a dataset producer, I want optional local visual assets fingerprinted,
    so that visual variation is reproducible without manual license entry.
11. As a dataset producer, I want a Window Instance Task Dataset, so that
    visible windows have masks, boxes, stable identities, and derived counts.
12. As a dataset producer, I want a Floorline Heatmap Task Dataset, so that
    visible floor boundaries can train a heatmap model and retain QA polylines.
13. As a dataset producer, I want a Visible Floor Count Task Dataset, so that
    image-level floor-count training uses projected visibility truth.
14. As a dataset producer, I want a Building Use Task Dataset, so that a model
    can learn the four agreed scene-truth Building Use classes.
15. As a dataset producer, I want a Facade Component Segmentation Task Dataset,
    so that visible facade pixels train a stable component vocabulary.
16. As a reviewer, I want a contact sheet and quantitative QA summary, so that
    I can assess visual variety and label quality before publication.
17. As a reviewer, I want the system to reject invalid labels, non-Blender
    outputs, and projection fallback, so that no non-trainable package ships.
18. As a dataset producer, I want one serialized Worker queue with safe cancel
    and resume, so that local rendering is dependable and prior valid work is
    retained.
19. As a reviewer, I want to publish manually after validation, so that no job
    becomes training data solely because rendering finished.
20. As a downstream training workflow, I want self-describing manifests and a
    Dataset Receipt, so that each Training Package is reproducible and auditable.

## Implementation Decisions

- The deployment shape is a local browser-based Web Studio and one local
  BlenderProc Worker; the browser never owns rendering or filesystem writes.
- Reuse the evaluated V3.1 scene grammar, projection, validation, and preview
  concepts by moving selected source into this repository. It is owned local
  code, not a submodule or synchronized upstream dependency.
- The Worker must boot BlenderProc before importing NumPy or generator modules.
  A failed runtime preflight is an explicit environment failure, never a direct
  Blender or projection-fallback substitution.
- A Generation Brief is immutable after explicit confirmation and records the
  Task Dataset, Output Target, split ratio, visual distributions, seeds, and
  task-specific visibility thresholds.
- Every Task Dataset is generated and versioned independently. The first
  release supports Window Instance/Count, Floorline Heatmap, Visible Floor
  Count, Building Use, and Facade Component Segmentation packages.
- Window count derives from visible Window Instance truth. Floorline Heatmap is
  the sole floorline training target; Floorline Polyline is QA/evaluation data.
- Component masks target visible raster pixels only. Complete scene geometry is
  metadata, not an amodal training label.
- The target visual domain is Chinese urban buildings constructed or materially
  renovated from the 2000s onward, with four Building Use classes, procedural
  foundation assets, optional local PBR/HDRI assets, selected daylight profiles,
  the complete facade View Family, and bounded Occlusion Profiles.
- A Trainable Package requires real BlenderProc/Blender output, valid labels,
  required asset fingerprints, complete Output Target, and a final review.
- Jobs are serialized. Cancellation is safe at sample boundaries; resume may
  retain only per-sample validated outputs and never bypass final validation or
  Publication Confirmation.
- A published package receives an immutable Dataset Receipt containing the
  brief hash, code and renderer identities, assets, seeds, actual parameters,
  validation, and publication decision.

## Testing Decisions

- The primary acceptance seam is the external Training Package publication
  contract: a confirmed Generation Brief reaches a manually published package
  through the local Studio and Worker, or produces an explicit failed state.
- Acceptance tests use a small real BlenderProc job and verify package layout,
  split ownership, task-native labels, validation status, review state, and
  Dataset Receipt. They must never accept direct Blender or projection fallback
  as a Trainable Package.
- Focused tests cover deterministic scene truth, label/image alignment, schema
  validation, task visibility gates, lifecycle transitions, cancellation/resume,
  and API validation. Tests assert externally observable contracts rather than
  internal implementation structure.
- Existing V3.1 projection, render-output, scene, and dataset-validator tests
  are reference prior art. Imported behavior is covered by local tests before
  it becomes a Worker capability.

## Out of Scope

- Training, evaluating, or serving machine-learning models inside the Studio.
- Night scenes, rain, snow, and unconstrained weather degradation.
- A distributed render farm, parallel local Worker execution, cloud storage, or
  a native macOS application package.
- Manual source and license fields for internal visual assets.
- Amodal component masks as a first-release segmentation target.
- Automatic publication, upstream synchronization, or retaining a Git submodule
  relationship with img2height.

## Further Notes

The first-release render resolution is fixed within each package, defaulting to
1024×768 (4:3), with RGB and pixel-aligned labels at the same resolution.
The user confirms Output Target, view, lighting, Building Use, and other brief
parameters before a job; split proportions remain adjustable. Diagnostic output
may explain a failure but may never be published as training data.
