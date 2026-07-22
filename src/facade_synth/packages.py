"""Task-native Training Package construction on top of BlenderProc seed output."""

from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

from .contracts import (
    BUILDING_USES,
    COMPONENT_CLASSES,
    OCCLUSION_BANDS,
    GenerationBrief,
    GenerationJob,
    RenderedPackage,
    TaskKind,
)
from .runtime import BlenderProcRuntime, RuntimeGateError, validate_render_summary
from .seed_v31.validate_dataset import DatasetValidationError, validate_dataset


class RenderCancelled(RuntimeGateError):
    """The Worker stopped safely before beginning the next Facade Sample."""


@dataclass(frozen=True)
class PlannedSample:
    sample_id: str
    recipe_id: str
    seed: int
    split: str
    building_use: str
    structure_variant: str
    view_band: str
    daylight_condition: str
    lighting_intensity_scale: float
    occlusion_band: str


class BlenderProcRenderer:
    """Real Worker renderer: BlenderProc evidence first, task package second."""

    identity = "blenderproc/v3.1-local-seed"

    def __init__(self, runtime: BlenderProcRuntime | None = None, *, render_samples: int = 16) -> None:
        self.runtime = runtime or BlenderProcRuntime()
        self.render_samples = render_samples

    def render(
        self,
        job: GenerationJob,
        package_dir: Path,
        *,
        cancellation_requested: Callable[[], bool] | None = None,
    ) -> RenderedPackage:
        validate_local_assets(job.brief)
        evidence = self.runtime.preflight()
        code_revision = _code_revision()
        if code_revision == "unknown":
            raise RuntimeGateError("cannot publish output without a source code revision")
        provenance = {
            "brief_hash": job.brief.brief_hash,
            "renderer_identity": self.identity,
            "code_revision": code_revision,
            "blender_version": str(evidence["blender_version"]),
            "blenderproc_version": str(evidence.get("blenderproc_version", "unknown")),
        }
        package_dir.mkdir(parents=True, exist_ok=True)
        plan = plan_samples(job.brief)
        records: list[dict[str, Any]] = []
        for sample in plan:
            if _cancel_requested_at_sample_boundary(job, cancellation_requested):
                raise RenderCancelled("job cancelled at a Facade Sample boundary")
            validate_local_assets(job.brief)
            sample_root = package_dir / "seed_samples" / sample.sample_id
            validated_record = sample_root / "validated_record.json"
            if validated_record.exists():
                cached_record = _load_cached_record(
                    validated_record,
                    planned=sample,
                    provenance=provenance,
                )
                if cached_record is not None:
                    try:
                        validate_dataset(sample_root)
                        validate_task_records(
                            (cached_record,),
                            brief=job.brief,
                            package_dir=package_dir,
                            allow_partial_recipe=True,
                        )
                        validate_task_annotations((cached_record,), package_dir=package_dir, brief=job.brief)
                    except (DatasetValidationError, OSError, RuntimeGateError, TypeError, ValueError, KeyError):
                        _quarantine_invalid_resume_sample(sample_root, package_dir)
                    else:
                        records.append(cached_record)
                        continue
                else:
                    _quarantine_invalid_resume_sample(sample_root, package_dir)
            sample_root.parent.mkdir(parents=True, exist_ok=True)
            arguments: list[str] = [
                    "--output-dir",
                    str(sample_root),
                    "--count",
                    "1",
                    "--width",
                    str(job.brief.render_width),
                    "--height",
                    str(job.brief.render_height),
                    "--seed",
                    str(sample.seed),
                    "--structure-variants",
                    sample.structure_variant,
                    "--view-band",
                    sample.view_band,
                    "--lighting-variant",
                    _lighting_variant(sample.daylight_condition),
                    "--lighting-intensity-scale",
                    str(sample.lighting_intensity_scale),
                    "--material-variant",
                    _material_variant(sample.building_use, sample.seed),
                    "--occlusion-band",
                    sample.occlusion_band,
                    "--render-samples",
                    str(self.render_samples),
                ]
            for asset_path in job.brief.asset_paths:
                arguments.extend(("--asset-path", asset_path))
            summary = self.runtime.run_generator(tuple(arguments))
            validate_render_summary(summary, expected_count=1)
            validate_dataset(sample_root)
            record = build_task_record(package_dir, sample_root, sample, job.brief)
            validated_record.write_text(
                json.dumps(
                    {"provenance": provenance, "planned_sample": asdict(sample), "record": record},
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            records.append(record)

        validate_task_records(records, brief=job.brief, package_dir=package_dir)
        manifest_path = package_dir / "manifest.jsonl"
        manifest_path.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
        )
        write_contact_sheet(package_dir, records)
        write_qa_summary(package_dir, records, job.brief)
        return RenderedPackage(
            package_dir=str(package_dir),
            validated_sample_count=len(records),
            renderer_identity=self.identity,
            code_revision=code_revision,
            blender_version=str(evidence["blender_version"]),
            blenderproc_version=str(evidence.get("blenderproc_version", "unknown")),
            sample_records=tuple(records),
        )


def _cancel_requested_at_sample_boundary(
    job: GenerationJob, cancellation_requested: Callable[[], bool] | None
) -> bool:
    """Consult durable Worker state so a different Studio process can cancel safely."""

    return job.cancelled_requested or (cancellation_requested is not None and cancellation_requested())


def plan_samples(brief: GenerationBrief) -> list[PlannedSample]:
    """Allocate recipes first, then render every requested view in the same split."""

    recipe_count = brief.output_target // len(brief.view_family)
    uses = _balanced_labels(brief.building_use_distribution, recipe_count, brief.seed)
    splits = _balanced_labels(brief.split_ratio, recipe_count, brief.seed + 1)
    daylight_conditions = _balanced_labels(brief.daylight_distribution, recipe_count, brief.seed + 2)
    occlusion_bands = _balanced_labels(brief.occlusion_distribution, recipe_count, brief.seed + 3)
    plan: list[PlannedSample] = []
    for recipe_index, (building_use, split, daylight_condition, occlusion_band) in enumerate(
        zip(uses, splits, daylight_conditions, occlusion_bands, strict=True)
    ):
        recipe_seed = brief.seed + recipe_index
        intensity_rng = random.Random(recipe_seed ^ 0x5A17)
        lighting_intensity_scale = round(
            intensity_rng.uniform(
                float(brief.lighting_intensity_range["min"]),
                float(brief.lighting_intensity_range["max"]),
            ),
            6,
        )
        recipe_id = f"recipe_{recipe_seed}"
        for view_band in brief.view_family:
            plan.append(
                PlannedSample(
                    sample_id=f"facade_{recipe_index:06d}_{view_band}",
                    recipe_id=recipe_id,
                    seed=recipe_seed,
                    split=split,
                    building_use=building_use,
                    structure_variant=_structure_variant(building_use, recipe_index),
                    view_band=view_band,
                    daylight_condition=daylight_condition,
                    lighting_intensity_scale=lighting_intensity_scale,
                    occlusion_band=occlusion_band,
                )
            )
    return plan


def build_task_record(
    package_dir: Path,
    seed_root: Path,
    planned: PlannedSample,
    brief: GenerationBrief,
) -> dict[str, Any]:
    source_metadata_path = seed_root / "metadata" / "facade_000000_metadata.json"
    metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    render_summary = _load_validated_render_summary(seed_root)
    source_prefix = seed_root.relative_to(package_dir).as_posix()
    rgb_path = f"{source_prefix}/{metadata['image']['rgb_path']}"
    labels = metadata["labels"]
    annotation = _task_annotation(package_dir, seed_root, planned, brief, metadata, source_prefix)
    _validate_task_annotation(annotation, brief.task)
    annotation_path = package_dir / "annotations" / f"{planned.sample_id}.json"
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    annotation_path.write_text(json.dumps(annotation, indent=2, sort_keys=True), encoding="utf-8")
    visible_floor_count = (
        int(annotation["visible_floor_count"])
        if brief.task is TaskKind.VISIBLE_FLOOR_COUNT
        else int(metadata["building"]["floor_count_visible"])
    )
    window_count = (
        int(annotation["facade_synth"]["window_count"])
        if brief.task is TaskKind.WINDOW_INSTANCE_COUNT
        else int(metadata["windows"]["instance_count"])
    )
    record = {
        "sample_id": planned.sample_id,
        "recipe_id": planned.recipe_id,
        "split": planned.split,
        "task": brief.task.value,
        "rgb_path": rgb_path,
        "annotation_path": annotation_path.relative_to(package_dir).as_posix(),
        "source_metadata_path": f"{source_prefix}/metadata/facade_000000_metadata.json",
        "source_artifact_sha256": source_artifact_sha256(seed_root),
        "render_backend": render_summary["render_backend"],
        "used_projection_fallback": render_summary["used_projection_fallback"],
        "building_use": planned.building_use,
        "view_band": planned.view_band,
        "daylight_condition": planned.daylight_condition,
        "lighting_intensity_scale": planned.lighting_intensity_scale,
        "occlusion_band": planned.occlusion_band,
        "occlusion_ratio": metadata["building"]["occlusion_ratio"],
        "visible_floor_count": visible_floor_count,
        "window_count": window_count,
        "scene_truth": metadata.get("scene_truth", {}),
        "render_parameters": metadata["generation_params"],
        "visibility_score": _task_visibility_score(brief.task, metadata),
        "source_labels": {
            "window_instance_mask_path": f"{source_prefix}/{labels['window_instance_mask_path']}",
            "floorline_heatmap_path": f"{source_prefix}/{labels['floorline_heatmap_path']}",
        },
    }
    record["task_artifact_sha256"] = task_artifact_sha256(package_dir, record)
    return record


def _task_annotation(
    package_dir: Path,
    seed_root: Path,
    planned: PlannedSample,
    brief: GenerationBrief,
    metadata: Mapping[str, Any],
    source_prefix: str,
) -> dict[str, Any]:
    labels = metadata["labels"]
    geometry = metadata["geometry"]
    if brief.task is TaskKind.WINDOW_INSTANCE_COUNT:
        return _window_instance_coco_annotation(
            package_dir,
            seed_root,
            planned,
            brief,
            metadata,
            source_prefix,
        )
    if brief.task is TaskKind.FLOORLINE_HEATMAP:
        return {
            "task": brief.task.value,
            "floorline_heatmap_path": f"{source_prefix}/{labels['floorline_heatmap_path']}",
            "floorline_polylines_px": geometry["floorline_polylines_px"],
        }
    if brief.task is TaskKind.VISIBLE_FLOOR_COUNT:
        fractions = geometry.get("floor_visibility_fraction", [])
        visible_floor_count = sum(
            isinstance(value, (int, float)) and float(value) >= brief.task_visibility_threshold
            for value in fractions
        )
        return {
            "task": brief.task.value,
            "visible_floor_count": visible_floor_count,
            "visibility_fraction": fractions,
            "visibility_threshold": brief.task_visibility_threshold,
        }
    if brief.task is TaskKind.BUILDING_USE:
        return {"task": brief.task.value, "building_use": planned.building_use}
    if brief.task is TaskKind.FACADE_COMPONENT_SEGMENTATION:
        mask_path, class_pixel_counts = _component_scene_truth_annotation(
            package_dir,
            seed_root,
            metadata,
            source_prefix,
        )
        return {
            "task": brief.task.value,
            "semantic_mask_path": mask_path,
            "classes": {name: index for index, name in enumerate(COMPONENT_CLASSES)},
            "class_pixel_counts": class_pixel_counts,
            "target": "visible_raster_only",
        }
    raise ValueError(f"unsupported task: {brief.task}")


def _window_instance_coco_annotation(
    package_dir: Path,
    seed_root: Path,
    planned: PlannedSample,
    brief: GenerationBrief,
    metadata: Mapping[str, Any],
    source_prefix: str,
) -> dict[str, Any]:
    """Write task-native visible masks and a one-image COCO annotation document."""

    labels = metadata["labels"]
    source_instance = seed_root / str(labels["window_instance_mask_path"])
    instance_mask = np.asarray(Image.open(source_instance).convert("L"))
    allowed_instances = [
        instance
        for instance in metadata["windows"]["instances"]
        if float(instance["visible_fraction"]) >= brief.task_visibility_threshold
    ]
    allowed_ids = {int(instance["id"]) for instance in allowed_instances}
    task_instance_mask = np.where(np.isin(instance_mask, list(allowed_ids)), instance_mask, 0).astype(np.uint8)
    task_semantic_mask = np.where(task_instance_mask > 0, 255, 0).astype(np.uint8)

    output_dir = package_dir / "annotations"
    output_dir.mkdir(parents=True, exist_ok=True)
    instance_path = output_dir / f"{planned.sample_id}_window_instance_mask.png"
    semantic_path = output_dir / f"{planned.sample_id}_window_semantic_mask.png"
    Image.fromarray(task_instance_mask, mode="L").save(instance_path)
    Image.fromarray(task_semantic_mask, mode="L").save(semantic_path)

    annotations = []
    for annotation_id, instance in enumerate(allowed_instances, start=1):
        instance_id = int(instance["id"])
        mask = task_instance_mask == instance_id
        if not np.any(mask):
            continue
        x_min, y_min, x_max, y_max = _mask_bbox(mask)
        annotations.append(
            {
                "id": annotation_id,
                "image_id": 1,
                "category_id": 1,
                "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                "area": int(np.count_nonzero(mask)),
                "iscrowd": 0,
                "segmentation": _coco_rle(mask),
                "facade_synth_instance_id": instance_id,
            }
        )
    return {
        "images": [
            {
                "id": 1,
                "file_name": f"{source_prefix}/{metadata['image']['rgb_path']}",
                "width": int(metadata["image"]["width"]),
                "height": int(metadata["image"]["height"]),
            }
        ],
        "annotations": annotations,
        "categories": [{"id": 1, "name": "window", "supercategory": "facade"}],
        "facade_synth": {
            "task": TaskKind.WINDOW_INSTANCE_COUNT.value,
            "window_count": len(annotations),
            "instance_mask_path": instance_path.relative_to(package_dir).as_posix(),
            "semantic_mask_path": semantic_path.relative_to(package_dir).as_posix(),
            "visibility_threshold": brief.task_visibility_threshold,
        },
    }


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        raise RuntimeGateError("visible window instance has no raster evidence")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _coco_rle(mask: np.ndarray) -> dict[str, Any]:
    """Return COCO's uncompressed column-major RLE for one visible instance."""

    flattened = np.asarray(mask, dtype=np.uint8).reshape(-1, order="F")
    counts: list[int] = []
    current = 0
    length = 0
    for value in flattened:
        item = int(value)
        if item == current:
            length += 1
        else:
            counts.append(length)
            current = item
            length = 1
    counts.append(length)
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": counts}


def _component_scene_truth_annotation(
    package_dir: Path,
    seed_root: Path,
    metadata: Mapping[str, Any],
    source_prefix: str,
) -> tuple[str, dict[str, int]]:
    """Reuse the Worker-written visible Blender material-pass raster, never projections."""

    labels = metadata["labels"]
    relative = str(labels.get("component_semantic_mask_path", ""))
    if not relative:
        raise RuntimeGateError("segmentation package requires a Blender component semantic pass")
    source = seed_root / relative
    if not source.exists():
        raise RuntimeGateError("segmentation package is missing its Blender component semantic pass")
    component = np.asarray(Image.open(source).convert("L"))
    expected_shape = (int(metadata["image"]["height"]), int(metadata["image"]["width"]))
    if component.shape != expected_shape:
        raise RuntimeGateError("component semantic pass dimensions do not match RGB")
    vocabulary = metadata.get("scene_truth", {}).get("component_class_ids")
    expected_vocabulary = {name: index for index, name in enumerate(COMPONENT_CLASSES)}
    if vocabulary != expected_vocabulary:
        raise RuntimeGateError("component semantic pass vocabulary is not the versioned public contract")
    if np.any(component > len(COMPONENT_CLASSES) - 1):
        raise RuntimeGateError("component semantic pass contains an unknown class id")
    counts = {name: int(np.count_nonzero(component == index)) for index, name in enumerate(COMPONENT_CLASSES)}
    return f"{source_prefix}/{relative}", counts


def validate_task_records(
    records: Sequence[Mapping[str, Any]],
    *,
    brief: GenerationBrief,
    package_dir: Path,
    allow_partial_recipe: bool = False,
) -> None:
    if not allow_partial_recipe and len(records) != brief.output_target:
        raise RuntimeGateError("package records do not match confirmed output_target")
    planned_by_id = {sample.sample_id: sample for sample in plan_samples(brief)}
    recipes: dict[str, tuple[str, set[str]]] = {}
    for record in records:
        if record.get("task") != brief.task.value:
            raise RuntimeGateError("package mixes task datasets")
        if record.get("render_backend") != "blenderproc_blender" or record.get("used_projection_fallback") is not False:
            raise RuntimeGateError("package contains non-BlenderProc or fallback output")
        split = record.get("split")
        if split not in {"train", "validation", "test"}:
            raise RuntimeGateError("package record has an invalid split")
        recipe_id = str(record.get("recipe_id"))
        existing_split, views = recipes.setdefault(recipe_id, (str(split), set()))
        if existing_split != split:
            raise RuntimeGateError("a Building Recipe crosses dataset splits")
        view_band = str(record.get("view_band"))
        if view_band not in brief.view_family:
            raise RuntimeGateError("package record has an unexpected camera view band")
        if view_band in views:
            raise RuntimeGateError("a Building Recipe has duplicate camera view bands")
        views.add(view_band)
        score = record.get("visibility_score")
        if not isinstance(score, (int, float)) or float(score) < brief.task_visibility_threshold:
            raise RuntimeGateError("sample does not meet the confirmed task visibility threshold")
        truth = record.get("scene_truth")
        if not isinstance(truth, Mapping) or truth.get("component_mask_origin") != "blender_object_index_pass":
            raise RuntimeGateError("sample lacks Blender object-index scene truth")
        render_parameters = record.get("render_parameters")
        if not isinstance(render_parameters, Mapping) or "lighting_recipe" not in render_parameters:
            raise RuntimeGateError("sample lacks actual Blender lighting recipe evidence")
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str):
            raise RuntimeGateError("package record has an invalid sample identifier")
        planned = planned_by_id.get(sample_id)
        if planned is None:
            raise RuntimeGateError("package record does not belong to the confirmed sample plan")
        _validate_lighting_intensity(record, render_parameters, planned, brief)
        _validate_occlusion_band(str(record.get("occlusion_band")), record.get("occlusion_ratio"))
        for key in ("rgb_path", "annotation_path", "source_metadata_path"):
            if not (package_dir / str(record[key])).exists():
                raise RuntimeGateError(f"package record references missing {key}")
        source_metadata_path = package_dir / str(record["source_metadata_path"])
        expected_source_digest = record.get("source_artifact_sha256")
        if not isinstance(expected_source_digest, str) or len(expected_source_digest) != 64:
            raise RuntimeGateError("package record lacks a source artifact digest")
        render_summary = _load_validated_render_summary(source_metadata_path.parent.parent)
        if (
            record.get("render_backend") != render_summary["render_backend"]
            or record.get("used_projection_fallback") is not render_summary["used_projection_fallback"]
        ):
            raise RuntimeGateError("package record disagrees with persisted BlenderProc render evidence")
        if source_artifact_sha256(source_metadata_path.parent.parent) != expected_source_digest:
            raise RuntimeGateError("sample source artifacts changed after validation")
        expected_task_digest = record.get("task_artifact_sha256")
        if not isinstance(expected_task_digest, str) or len(expected_task_digest) != 64:
            raise RuntimeGateError("package record lacks a task artifact digest")
        if task_artifact_sha256(package_dir, record) != expected_task_digest:
            raise RuntimeGateError("sample task artifacts changed after validation")
    if not allow_partial_recipe:
        expected_views = set(brief.view_family)
        for _recipe_id, (_split, views) in recipes.items():
            if views != expected_views:
                raise RuntimeGateError("each Building Recipe must contain the complete confirmed view family")


def _validate_lighting_intensity(
    record: Mapping[str, Any],
    render_parameters: Mapping[str, Any],
    planned: PlannedSample,
    brief: GenerationBrief,
) -> None:
    """Bind the Blender-written light recipe to the immutable plan and brief range."""

    lighting_recipe = render_parameters.get("lighting_recipe")
    if not isinstance(lighting_recipe, Mapping):
        raise RuntimeGateError("sample lacks actual Blender lighting recipe evidence")
    actual = lighting_recipe.get("intensity_scale")
    recorded = record.get("lighting_intensity_scale")
    if (
        isinstance(actual, bool)
        or isinstance(recorded, bool)
        or not isinstance(actual, (int, float))
        or not isinstance(recorded, (int, float))
    ):
        raise RuntimeGateError("sample lacks a numeric actual lighting intensity")
    actual_value = float(actual)
    recorded_value = float(recorded)
    if not math.isfinite(actual_value) or not math.isfinite(recorded_value):
        raise RuntimeGateError("sample has a non-finite lighting intensity")
    minimum = float(brief.lighting_intensity_range["min"])
    maximum = float(brief.lighting_intensity_range["max"])
    if not minimum <= actual_value <= maximum:
        raise RuntimeGateError("sample actual lighting intensity is outside the confirmed brief range")
    if actual_value != planned.lighting_intensity_scale:
        raise RuntimeGateError("sample actual lighting intensity does not match its confirmed plan")
    if recorded_value != actual_value:
        raise RuntimeGateError("sample record does not bind lighting intensity to its Blender recipe")


def validate_task_annotations(
    records: Sequence[Mapping[str, Any]], *, package_dir: Path, brief: GenerationBrief
) -> None:
    """Validate the task-native labels that a receipt is about to bind."""

    for record in records:
        annotation_path = package_dir / str(record["annotation_path"])
        try:
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeGateError(f"task annotation is unreadable: {annotation_path}") from exc
        if brief.task is TaskKind.WINDOW_INSTANCE_COUNT:
            _validate_window_coco_annotation(annotation, record, package_dir)
        elif brief.task is TaskKind.FLOORLINE_HEATMAP:
            _validate_floorline_annotation(annotation, package_dir)
        elif brief.task is TaskKind.VISIBLE_FLOOR_COUNT:
            _validate_visible_floor_count_annotation(annotation, brief)
        elif brief.task is TaskKind.BUILDING_USE:
            if annotation != {"task": brief.task.value, "building_use": record["building_use"]}:
                raise RuntimeGateError("building-use annotation is inconsistent with manifest scene truth")
        elif brief.task is TaskKind.FACADE_COMPONENT_SEGMENTATION:
            _validate_component_annotation(annotation, package_dir)


def _validate_window_coco_annotation(
    annotation: Any, record: Mapping[str, Any], package_dir: Path
) -> None:
    if not isinstance(annotation, Mapping):
        raise RuntimeGateError("window annotation must be a COCO document")
    images = annotation.get("images")
    instances = annotation.get("annotations")
    categories = annotation.get("categories")
    extra = annotation.get("facade_synth")
    if not isinstance(images, list) or len(images) != 1 or not isinstance(instances, list):
        raise RuntimeGateError("window annotation has an invalid COCO image or instance list")
    if categories != [{"id": 1, "name": "window", "supercategory": "facade"}]:
        raise RuntimeGateError("window annotation has an invalid COCO category contract")
    if not isinstance(extra, Mapping) or extra.get("task") != TaskKind.WINDOW_INSTANCE_COUNT.value:
        raise RuntimeGateError("window annotation is missing its task contract")
    if int(extra.get("window_count", -1)) != len(instances) or int(record["window_count"]) != len(instances):
        raise RuntimeGateError("window annotation count does not match manifest")
    for key in ("instance_mask_path", "semantic_mask_path"):
        label_path = package_dir / str(extra.get(key, ""))
        if not label_path.exists():
            raise RuntimeGateError(f"window annotation references a missing {key}")
    for instance in instances:
        if not isinstance(instance, Mapping) or instance.get("category_id") != 1:
            raise RuntimeGateError("window annotation contains an invalid COCO instance")
        if not isinstance(instance.get("segmentation"), Mapping) or not isinstance(instance["segmentation"].get("counts"), list):
            raise RuntimeGateError("window annotation lacks COCO RLE segmentation")


def _validate_floorline_annotation(annotation: Any, package_dir: Path) -> None:
    if not isinstance(annotation, Mapping) or annotation.get("task") != TaskKind.FLOORLINE_HEATMAP.value:
        raise RuntimeGateError("floorline annotation has an invalid task contract")
    heatmap_path = package_dir / str(annotation.get("floorline_heatmap_path", ""))
    if not heatmap_path.exists() or not annotation.get("floorline_polylines_px"):
        raise RuntimeGateError("floorline annotation lacks visible heatmap or polylines")
    heatmap = np.asarray(Image.open(heatmap_path).convert("L"))
    if not np.any(heatmap):
        raise RuntimeGateError("floorline heatmap contains no visible scene-truth evidence")


def _validate_visible_floor_count_annotation(annotation: Any, brief: GenerationBrief) -> None:
    if not isinstance(annotation, Mapping) or annotation.get("task") != TaskKind.VISIBLE_FLOOR_COUNT.value:
        raise RuntimeGateError("floor-count annotation has an invalid task contract")
    if annotation.get("visibility_threshold") != brief.task_visibility_threshold:
        raise RuntimeGateError("floor-count annotation does not use the confirmed visibility threshold")
    if int(annotation.get("visible_floor_count", 0)) < 1:
        raise RuntimeGateError("floor-count annotation has no visible floors")


def _validate_component_annotation(annotation: Any, package_dir: Path) -> None:
    if not isinstance(annotation, Mapping) or annotation.get("task") != TaskKind.FACADE_COMPONENT_SEGMENTATION.value:
        raise RuntimeGateError("component annotation has an invalid task contract")
    mask_path = package_dir / str(annotation.get("semantic_mask_path", ""))
    if not mask_path.exists() or annotation.get("target") != "visible_raster_only":
        raise RuntimeGateError("component annotation lacks a visible-raster semantic mask")


def _load_cached_record(
    cache_path: Path, *, planned: PlannedSample, provenance: Mapping[str, str]
) -> dict[str, Any] | None:
    """Accept a resume cache only when it belongs to this immutable render execution."""

    try:
        value = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, Mapping):
        return None
    if value.get("provenance") != dict(provenance) or value.get("planned_sample") != asdict(planned):
        return None
    record = value.get("record")
    if not isinstance(record, dict):
        return None
    if record.get("sample_id") != planned.sample_id or record.get("recipe_id") != planned.recipe_id:
        return None
    return record


def _quarantine_invalid_resume_sample(sample_root: Path, package_dir: Path) -> None:
    """Preserve a failed resume candidate while making its planned sample rerenderable."""

    if not sample_root.exists():
        return
    quarantine_dir = package_dir / "invalid_samples"
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    destination = quarantine_dir / sample_root.name
    suffix = 1
    while destination.exists():
        destination = quarantine_dir / f"{sample_root.name}-{suffix}"
        suffix += 1
    try:
        sample_root.replace(destination)
    except OSError as exc:
        raise RuntimeGateError("cannot quarantine an invalid cached Facade Sample") from exc


def _load_validated_render_summary(seed_root: Path) -> dict[str, Any]:
    """Read the Worker-written summary instead of trusting manifest literals."""

    summary_path = seed_root / "run_summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeGateError("sample lacks a readable BlenderProc run summary") from exc
    if not isinstance(summary, dict):
        raise RuntimeGateError("sample BlenderProc run summary must be an object")
    validate_render_summary(summary, expected_count=1)
    return summary


def validate_frozen_sample_records(
    records: Sequence[Mapping[str, Any]],
    *,
    brief: GenerationBrief,
    package_dir: Path,
    provenance: Mapping[str, str],
) -> None:
    """Bind mutable package files back to each Worker-written validated record."""

    planned_by_id = {sample.sample_id: sample for sample in plan_samples(brief)}
    if set(str(record.get("sample_id")) for record in records) != set(planned_by_id):
        raise RuntimeGateError("manifest records do not match the confirmed sample plan")
    for record in records:
        sample_id = str(record["sample_id"])
        planned = planned_by_id[sample_id]
        expected_metadata = f"seed_samples/{sample_id}/metadata/facade_000000_metadata.json"
        if record.get("source_metadata_path") != expected_metadata:
            raise RuntimeGateError("manifest record does not reference its planned Blender source")
        cache_path = package_dir / "seed_samples" / sample_id / "validated_record.json"
        frozen_record = _load_cached_record(cache_path, planned=planned, provenance=provenance)
        if frozen_record is None:
            raise RuntimeGateError("sample lacks a validated record for this immutable execution")
        if _canonical_json(frozen_record) != _canonical_json(record):
            raise RuntimeGateError("manifest record changed after sample validation")


def task_artifact_sha256(package_dir: Path, record: Mapping[str, Any]) -> str:
    """Hash source RGB/truth plus the task annotation and its label files."""

    package_root = package_dir.resolve()
    source_metadata = package_root / str(record.get("source_metadata_path", ""))
    if not source_metadata.exists():
        raise RuntimeGateError("sample source metadata is missing")
    source_root = source_metadata.parent.parent
    paths = {package_root / str(record.get("annotation_path", ""))}
    try:
        annotation = json.loads(next(iter(paths)).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeGateError("task annotation is unreadable for artifact validation") from exc
    for relative in _annotation_paths(annotation):
        candidate = package_root / relative
        try:
            candidate.resolve().relative_to(package_root)
        except ValueError:
            raise RuntimeGateError("task annotation references a path outside its package")

        paths.add(candidate)
    digest = hashlib.sha256()
    digest.update(source_artifact_sha256(source_root).encode("ascii"))
    for path in sorted(paths):
        if not path.exists() or not path.is_file():
            raise RuntimeGateError("task annotation references a missing artifact")
        digest.update(path.relative_to(package_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _annotation_paths(value: Any) -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if key.endswith("_path") and isinstance(nested, str):
                paths.append(nested)
            paths.extend(_annotation_paths(nested))
    elif isinstance(value, list):
        for nested in value:
            paths.extend(_annotation_paths(nested))
    return paths


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def write_contact_sheet(package_dir: Path, records: Sequence[Mapping[str, Any]]) -> Path:
    tiles: list[Image.Image] = []
    for record in records[:16]:
        with Image.open(package_dir / str(record["rgb_path"])) as image:
            tile = image.convert("RGB").resize((192, 144))
        canvas = Image.new("RGB", (192, 172), "white")
        canvas.paste(tile, (0, 0))
        ImageDraw.Draw(canvas).text(
            (6, 149),
            f"{record['sample_id']}  {record['split']}",
            fill="black",
        )
        tiles.append(canvas)
    columns = 4
    rows = max(1, (len(tiles) + columns - 1) // columns)
    sheet = Image.new("RGB", (columns * 192, rows * 172), "#eeeeee")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * 192, (index // columns) * 172))
    output = package_dir / "preview" / "contact_sheet.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    return output


def write_qa_summary(package_dir: Path, records: Sequence[Mapping[str, Any]], brief: GenerationBrief) -> None:
    payload = {
        "task": brief.task.value,
        "sample_count": len(records),
        "splits": {split: sum(record["split"] == split for record in records) for split in ("train", "validation", "test")},
        "building_uses": {use: sum(record["building_use"] == use for record in records) for use in BUILDING_USES},
        "view_bands": sorted({str(record["view_band"]) for record in records}),
        "daylight_conditions": sorted({str(record["daylight_condition"]) for record in records}),
        "occlusion_bands": sorted({str(record["occlusion_band"]) for record in records}),
        "target_domain": brief.target_domain,
        "visibility": {
            "minimum": min(float(record["visibility_score"]) for record in records),
            "mean": sum(float(record["visibility_score"]) for record in records) / len(records),
            "threshold": brief.task_visibility_threshold,
        },
        "occlusion_ratio": {
            "maximum": max(float(record["occlusion_ratio"]) for record in records),
            "bands": {
                band: sum(record["occlusion_band"] == band for record in records)
                for band in OCCLUSION_BANDS
            },
        },
        "contact_sheet": "preview/contact_sheet.png",
    }
    (package_dir / "qa_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def fingerprint_local_asset(asset_path: Path) -> str:
    """Return the automatically captured path-plus-content identity for a local asset."""

    path = Path(asset_path).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"{path}:{digest}"


def validate_local_assets(brief: GenerationBrief) -> None:
    """Fail closed if a confirmed internal texture/HDRI moved or changed before render."""

    for path_value, expected in zip(brief.asset_paths, brief.asset_fingerprints, strict=True):
        path = Path(path_value)
        if not path.exists() or not path.is_file():
            raise RuntimeGateError(f"confirmed visual asset is unavailable: {path}")
        actual = fingerprint_local_asset(path)
        if actual != expected:
            raise RuntimeGateError(f"confirmed visual asset changed after brief confirmation: {path}")


def _task_visibility_score(task: TaskKind, metadata: Mapping[str, Any]) -> float:
    visibility = metadata.get("scene_truth", {}).get("visibility", {})
    if not isinstance(visibility, Mapping):
        raise RuntimeGateError("sample is missing scene-truth visibility evidence")
    key = {
        TaskKind.WINDOW_INSTANCE_COUNT: "window_glass",
        TaskKind.FLOORLINE_HEATMAP: "floor_band",
        TaskKind.VISIBLE_FLOOR_COUNT: "floor_band",
        TaskKind.BUILDING_USE: "facade_wall",
        TaskKind.FACADE_COMPONENT_SEGMENTATION: "facade_components",
    }[task]
    value = visibility.get(key)
    if not isinstance(value, (int, float)):
        raise RuntimeGateError(f"sample is missing {key} visibility truth")
    return float(value)


def _validate_task_annotation(annotation: Mapping[str, Any], task: TaskKind) -> None:
    if task is TaskKind.WINDOW_INSTANCE_COUNT:
        extra = annotation.get("facade_synth")
        if not isinstance(extra, Mapping) or int(extra.get("window_count", 0)) < 1:
            raise RuntimeGateError("window-count sample has no visible Blender-derived instances")
    if task is TaskKind.FLOORLINE_HEATMAP and not annotation["floorline_polylines_px"]:
        raise RuntimeGateError("floorline sample has no visible floorline evidence")
    if task is TaskKind.VISIBLE_FLOOR_COUNT and int(annotation["visible_floor_count"]) < 1:
        raise RuntimeGateError("floor-count sample has no visible floors")
    if task is TaskKind.FACADE_COMPONENT_SEGMENTATION:
        counts = annotation["class_pixel_counts"]
        if int(counts["facade_wall"]) < 1 or int(counts["background"]) < 1:
            raise RuntimeGateError("component segmentation sample lacks visible wall/background classes")


def _validate_occlusion_band(band: str, value: Any) -> None:
    if not isinstance(value, (int, float)):
        raise RuntimeGateError("sample lacks scene-truth foreground occlusion ratio")
    ratio = float(value)
    if not 0.0 <= ratio <= 0.30:
        raise RuntimeGateError("foreground occlusion must remain within the 0–30% contract")
    if band == "clear" and ratio > 0.001:
        raise RuntimeGateError("clear sample contains foreground occlusion")
    if band == "light_0_15" and not 0.001 < ratio <= 0.15:
        raise RuntimeGateError("light occlusion sample is outside its 0–15% contract")
    if band == "moderate_15_30" and not 0.15 < ratio <= 0.30:
        raise RuntimeGateError("moderate occlusion sample is outside its 15–30% contract")
    if band not in OCCLUSION_BANDS:
        raise RuntimeGateError("sample has an unknown occlusion band")


def _balanced_labels(distribution: Mapping[str, float], count: int, seed: int) -> list[str]:
    counts = {key: int(value * count) for key, value in distribution.items()}
    remainder = count - sum(counts.values())
    fractions = sorted(
        distribution,
        key=lambda key: (-(distribution[key] * count - counts[key]), key),
    )
    for key in fractions[:remainder]:
        counts[key] += 1
    labels = [key for key, label_count in counts.items() for _ in range(label_count)]
    random.Random(seed).shuffle(labels)
    return labels


def _structure_variant(building_use: str, index: int) -> str:
    if building_use == "residential":
        return ("residential_recessed", "balcony_residential")[index % 2]
    if building_use == "mixed_use":
        return "podium_mixed_use"
    if building_use in {"office", "commercial"}:
        return "commercial_curtain_wall"
    raise ValueError(f"unsupported building use: {building_use}")


def _lighting_variant(daylight_condition: str) -> str:
    return {
        "clear": "morning_side",
        "overcast": "overcast",
        "warm_low_angle": "late_afternoon",
        "backlit": "soft_front",
    }[daylight_condition]


def _material_variant(building_use: str, seed: int) -> str:
    options = {
        "residential": ("stucco_cool", "brick_warm"),
        "office": ("painted_panel", "concrete_light"),
        "commercial": ("concrete_light", "painted_panel"),
        "mixed_use": ("brick_warm", "concrete_light"),
    }[building_use]
    return options[seed % len(options)]


def _code_revision() -> str:
    source_root = next(
        (parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists()),
        None,
    )
    if source_root is None:
        return "unknown"
    try:
        revision = subprocess.run(
            ("git", "-C", str(source_root), "rev-parse", "HEAD"),
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        dirty_status = subprocess.run(
            ("git", "-C", str(source_root), "status", "--porcelain=v1", "-z"),
            capture_output=True,
            check=True,
        ).stdout
        if not dirty_status:
            return revision
        tracked_diff = subprocess.run(
            ("git", "-C", str(source_root), "diff", "--binary", "HEAD"),
            capture_output=True,
            check=True,
        ).stdout
        untracked = subprocess.run(
            ("git", "-C", str(source_root), "ls-files", "--others", "--exclude-standard", "-z"),
            capture_output=True,
            check=True,
        ).stdout.split(b"\0")
        material = bytearray(tracked_diff)
        for relative_path in sorted(path for path in untracked if path):
            file_path = source_root / relative_path.decode("utf-8")
            material.extend(relative_path)
            material.extend(b"\0")
            material.extend(hashlib.sha256(file_path.read_bytes()).digest())
        working_tree_hash = hashlib.sha256(bytes(material)).hexdigest()[:16]
        return f"{revision}+dirty:{working_tree_hash}"
    except (OSError, UnicodeDecodeError, subprocess.CalledProcessError):
        return "unknown"


def source_artifact_sha256(root: Path) -> str:
    """Hash all persisted source artifacts while excluding the mutable resume cache."""

    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        if path.name == "validated_record.json":
            continue
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()
