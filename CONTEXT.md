# Blender Facade Synthesis

The project creates procedurally rendered facade datasets whose labels are
derived from the generated scene, for training and evaluating facade-analysis
models.

## Language

**Synthetic facade dataset**:
A versioned collection of rendered facade samples and their scene-truth training
labels.
_Avoid_: render batch, image set

**Facade sample**:
One RGB render together with every label, annotation, and provenance record
produced from the same generated scene.
_Avoid_: image, render

**Building use**:
The categorical scene-truth label describing the target building's intended
facade program. The initial vocabulary is `residential`, `office`,
`commercial`, and `mixed_use`.
_Avoid_: building type, style

**Target domain**:
The real-image population that the synthetic facade grammar approximates. The
first-release target domain is Chinese urban buildings constructed or materially
renovated from the 2000s onward.
_Avoid_: generic architecture, global style

**Mixed use**:
A building use whose visible facade intentionally combines more than one
program, initially represented as a distinct class rather than floor-level
multi-labels.
_Avoid_: commercial, residential

**Scene truth**:
Labels computed from the procedural scene specification or its render passes,
rather than inferred from the RGB render.
_Avoid_: predicted label, pseudo-label

**Asset fingerprint**:
The automatically captured local path or asset identifier and content SHA of a
visual asset used by an internal generation job. It supports reproducibility but
does not require a user to enter source or license metadata; an asset never
supplies scene truth.
_Avoid_: asset receipt, license record

**Foundation asset library**:
The built-in procedural materials and daylight-environment presets that let the
Web Studio render a trainable package without external files. User-selected
local PBR textures and HDRIs are optional visual extensions.
_Avoid_: required asset catalog, external dependency

**Seed generator**:
The evaluated subset of the `img2height` V3.1 synthetic-facade generator moved
under this repository's ownership as a starting point for the Web Studio. It is
not a git submodule, runtime remote dependency, or synchronised mirror.
_Avoid_: upstream dependency, submodule

**Task dataset**:
A separately generated, versioned, training-ready dataset for one learning
target, such as window instances, floor lines, floor count, or building use.
_Avoid_: export profile, view of a master dataset

**Task curriculum**:
The task-specific sampling distribution, class balance, and visibility
thresholds that determine which scene recipes become samples in one task
dataset. It reuses the shared scene grammar without requiring identical renders
across tasks.
_Avoid_: shared batch, generic sampler

**Dataset split**:
The generation-time allocation of building recipes to `train`, `validation`,
or `test`. A recipe and all renders derived from it belong to exactly one split.
_Avoid_: random image split, post-generation split

**Output target**:
The exact user-confirmed sample count for a task dataset, including its
generation-time allocation across train, validation, and test splits.
_Avoid_: estimated batch size, default count

**Training package**:
The self-describing deliverable of one task dataset: its images, annotations,
split manifests, validation result, and versioned provenance. It is consumed by
external training code, not trained inside the Web Studio.
_Avoid_: model package, training project

**Trainable package**:
A training package whose samples were rendered by BlenderProc/Blender and pass
asset-fingerprint and label validation. Projection fallbacks may never be part
of it.
_Avoid_: successful job, preview package

**Diagnostic preview**:
An inspectable non-trainable render or projection output retained to explain a
failed job or support development.
_Avoid_: sample dataset, fallback dataset

**Generation brief**:
The user-confirmed, immutable specification for one dataset job. It states the
task, required output count, camera-angle distribution, lighting-intensity
distribution, building-use distribution, and split ratio before rendering
starts.
_Avoid_: job form, transient settings

**Generation job**:
The worker's execution of one immutable generation brief. It moves through
queued, running, ready-for-review, published, failed, or cancelled states.
_Avoid_: render command, background process

**Worker queue**:
The ordered local execution queue for generation jobs. The first-release worker
processes exactly one BlenderProc job at a time.
_Avoid_: parallel renderer, browser queue

**Resumable job**:
A generation job whose immutable brief and per-sample validation records let a
worker preserve completed valid samples and render only missing samples after an
interruption. It remains non-publishable until the full output target and final
validation pass.
_Avoid_: partial dataset, restarted job

**View family**:
The complete set of facade camera-angle bands rendered from one building recipe.
The first release includes frontal, light/medium-oblique, and strong-oblique
facade views; all resulting samples remain in the recipe's one dataset split.
_Avoid_: random camera, duplicate sample

**Daylight profile**:
A user-confirmable distribution of daylight scene conditions. The first release
offers `daylight_diverse` (clear, overcast, warm low-angle, and backlit) and
`controlled_daylight` (clear and overcast only); neither includes night or
weather-degradation scenes.
_Avoid_: generic lighting, all-weather profile

**Lighting recipe**:
The exact per-sample Blender lighting controls: sun elevation, relative sun
azimuth, sun energy, world strength, exposure EV, and colour temperature. A
human-readable weak/normal/strong label may guide the UI but is not a physical
illuminance claim.
_Avoid_: lux value, lighting preset only

**Visibility truth**:
The per-view scene-truth determination of which generated facade components,
windows, floors, and boundaries project with sufficient evidence to label a
sample.
_Avoid_: predicted visibility, assumed visibility

**Visible raster label**:
A training mask containing only pixels directly visible to the rendered camera.
It is the first-release segmentation target.
_Avoid_: complete-object mask, amodal mask

**Amodal geometry record**:
The scene-truth identity and complete geometry of a component, retained in
metadata but not emitted as the first-release segmentation target.
_Avoid_: hidden-pixel label

**Occlusion profile**:
The user-confirmed distribution of non-target foreground occlusion in a task
dataset. The first release uses `light_controlled_occlusion`: clear, light
(0–15%), and moderate (15–30%) target-facade occlusion bands.
_Avoid_: random clutter, unbounded occlusion

**Task visibility threshold**:
The minimum scene-truth projected evidence required to publish a sample for a
particular training target. It may differ between window instances, floorlines,
floor count, building use, and component segmentation.
_Avoid_: global visibility rule, object presence

**Publication confirmation**:
The explicit human decision to publish a validated training package after
reviewing its visual and quantitative quality summary.
_Avoid_: automatic release, completed job

**Dataset receipt**:
The immutable provenance record issued with a published training package. It
binds the generation brief, code revision, renderer versions, asset
fingerprints, seeds, actual render parameters, validation result, and
publication confirmation.
_Avoid_: run log, dataset note

**Package manifest**:
The JSONL index of a training package. Each record identifies one sample, its
split, annotation paths, scene-truth labels, and provenance needed by that
task's loader.
_Avoid_: master manifest, dataset index

**Render resolution**:
The fixed, user-confirmed width, height, and aspect ratio of every RGB image and
pixel-aligned label in one training package. The first-release default is
1024×768 (4:3).
_Avoid_: display resolution, independent mask resize

**Floorline heatmap**:
The raster supervision target for visible facade floor boundaries, produced from
scene-truth floorline geometry.
_Avoid_: floorline mask, floor count label

**Floorline polyline**:
The pixel-coordinate scene-truth path of a visible floor boundary. It supports
quality assurance, evaluation, and post-processing; it is not a separate
first-release training target.
_Avoid_: detected floorline

**Visible floor count**:
The number of building floors with sufficient projected facade evidence in a
rendered sample. It is an image-level scene-truth label and a distinct training
target from the floorline heatmap.
_Avoid_: true floor count, inferred floor count

**Window instance**:
One individually identified visible window opening, with its spatial label and
stable instance identifier. Its inclusion in a sample determines the derived
window count.
_Avoid_: window pixel, window detection

**Window count**:
The number of visible window instances in a facade sample; it is derived from
the instance annotation rather than created as an independent scene label.
_Avoid_: estimated count

**Facade component segmentation**:
The independent semantic-segmentation task dataset for visible facade pixels.
Its first-release vocabulary is `facade_wall`, `window_glass`, `window_frame`,
`door`, `balcony`, `floor_band`, `podium_storefront`, `roof_parapet`, and
`background`.
_Avoid_: generic facade mask, component detection
