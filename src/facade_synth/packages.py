"""Task-native Training Package construction on top of BlenderProc seed output."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

from .contracts import BUILDING_USES, COMPONENT_CLASSES, GenerationBrief, GenerationJob, RenderedPackage, TaskKind
from .runtime import BlenderProcRuntime, RuntimeGateError, validate_render_summary
from .seed_v31.validate_dataset import validate_dataset


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
    occlusion_band: str


class BlenderProcRenderer:
    """Real Worker renderer: BlenderProc evidence first, task package second."""

    identity = "blenderproc/v3.1-local-seed"

    def __init__(self, runtime: BlenderProcRuntime | None = None, *, render_samples: int = 16) -> None:
        self.runtime = runtime or BlenderProcRuntime()
        self.render_samples = render_samples

    def render(self, job: GenerationJob, package_dir: Path) -> RenderedPackage:
        validate_local_assets(job.brief)
        evidence = self.runtime.preflight()
        package_dir.mkdir(parents=True, exist_ok=True)
        plan = plan_samples(job.brief)
        records: list[dict[str, Any]] = []
        for sample in plan:
            if job.cancelled_requested:
                raise RenderCancelled("job cancelled at a Facade Sample boundary")
            sample_root = package_dir / "seed_samples" / sample.sample_id
            validated_record = sample_root / "validated_record.json"
            if validated_record.exists():
                validate_dataset(sample_root)
                cached_record = json.loads(validated_record.read_text(encoding="utf-8"))
                validate_task_records((cached_record,), brief=job.brief, package_dir=package_dir, allow_partial_recipe=True)
                records.append(cached_record)
                continue
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
            validated_record.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
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
            code_revision=_code_revision(),
            blender_version=str(evidence["blender_version"]),
            blenderproc_version=str(evidence.get("blenderproc_version", "unknown")),
            sample_records=tuple(records),
        )


def plan_samples(brief: GenerationBrief) -> list[PlannedSample]:
    """Allocate recipes first, then render every requested view in the same split."""

    recipe_count = brief.output_target // len(brief.view_family)
    uses = _balanced_labels(brief.building_use_distribution, recipe_count, brief.seed)
    splits = _balanced_labels(brief.split_ratio, recipe_count, brief.seed + 1)
    daylight_options = (
        ("clear", "overcast")
        if brief.daylight_profile == "controlled_daylight"
        else ("clear", "overcast", "warm_low_angle", "backlit")
    )
    occlusion_options = ("clear", "light_0_15", "moderate_15_30")
    plan: list[PlannedSample] = []
    for recipe_index, (building_use, split) in enumerate(zip(uses, splits, strict=True)):
        recipe_seed = brief.seed + recipe_index
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
                    daylight_condition=daylight_options[recipe_index % len(daylight_options)],
                    occlusion_band=occlusion_options[recipe_index % len(occlusion_options)],
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
    source_prefix = seed_root.relative_to(package_dir).as_posix()
    rgb_path = f"{source_prefix}/{metadata['image']['rgb_path']}"
    labels = metadata["labels"]
    annotation = _task_annotation(package_dir, seed_root, planned, brief, metadata, source_prefix)
    _validate_task_annotation(annotation, brief.task)
    annotation_path = package_dir / "annotations" / f"{planned.sample_id}.json"
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    annotation_path.write_text(json.dumps(annotation, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "sample_id": planned.sample_id,
        "recipe_id": planned.recipe_id,
        "split": planned.split,
        "task": brief.task.value,
        "rgb_path": rgb_path,
        "annotation_path": annotation_path.relative_to(package_dir).as_posix(),
        "source_metadata_path": f"{source_prefix}/metadata/facade_000000_metadata.json",
        "render_backend": "blenderproc_blender",
        "used_projection_fallback": False,
        "building_use": planned.building_use,
        "view_band": planned.view_band,
        "daylight_condition": planned.daylight_condition,
        "occlusion_band": planned.occlusion_band,
        "occlusion_ratio": metadata["building"]["occlusion_ratio"],
        "visible_floor_count": metadata["building"]["floor_count_visible"],
        "window_count": metadata["windows"]["instance_count"],
        "scene_truth": metadata.get("scene_truth", {}),
        "render_parameters": metadata["generation_params"],
        "visibility_score": _task_visibility_score(brief.task, metadata),
        "source_labels": {
            "window_instance_mask_path": f"{source_prefix}/{labels['window_instance_mask_path']}",
            "floorline_heatmap_path": f"{source_prefix}/{labels['floorline_heatmap_path']}",
        },
    }


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
        return {
            "task": brief.task.value,
            "window_count": metadata["windows"]["instance_count"],
            "instances": metadata["windows"]["instances"],
            "window_instance_mask_path": f"{source_prefix}/{labels['window_instance_mask_path']}",
            "window_semantic_mask_path": f"{source_prefix}/{labels['window_semantic_mask_path']}",
        }
    if brief.task is TaskKind.FLOORLINE_HEATMAP:
        return {
            "task": brief.task.value,
            "floorline_heatmap_path": f"{source_prefix}/{labels['floorline_heatmap_path']}",
            "floorline_polylines_px": geometry["floorline_polylines_px"],
        }
    if brief.task is TaskKind.VISIBLE_FLOOR_COUNT:
        return {
            "task": brief.task.value,
            "visible_floor_count": metadata["building"]["floor_count_visible"],
            "visibility_fraction": geometry.get("floor_visibility_fraction", []),
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
        if not isinstance(score, (int, float)) or float(score) < brief.visibility_threshold:
            raise RuntimeGateError("sample does not meet the confirmed task visibility threshold")
        truth = record.get("scene_truth")
        if not isinstance(truth, Mapping) or truth.get("component_mask_origin") != "blender_object_index_pass":
            raise RuntimeGateError("sample lacks Blender object-index scene truth")
        render_parameters = record.get("render_parameters")
        if not isinstance(render_parameters, Mapping) or "lighting_recipe" not in render_parameters:
            raise RuntimeGateError("sample lacks actual Blender lighting recipe evidence")
        _validate_occlusion_band(str(record.get("occlusion_band")), record.get("occlusion_ratio"))
        for key in ("rgb_path", "annotation_path", "source_metadata_path"):
            if not (package_dir / str(record[key])).exists():
                raise RuntimeGateError(f"package record references missing {key}")
    if not allow_partial_recipe:
        expected_views = set(brief.view_family)
        for _recipe_id, (_split, views) in recipes.items():
            if views != expected_views:
                raise RuntimeGateError("each Building Recipe must contain the complete confirmed view family")


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
            "threshold": brief.visibility_threshold,
        },
        "occlusion_ratio": {
            "maximum": max(float(record["occlusion_ratio"]) for record in records),
            "bands": {
                band: sum(record["occlusion_band"] == band for record in records)
                for band in ("clear", "light_0_15", "moderate_15_30")
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
    if task is TaskKind.WINDOW_INSTANCE_COUNT and int(annotation["window_count"]) < 1:
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
    if band not in {"clear", "light_0_15", "moderate_15_30"}:
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
    try:
        revision = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ("git", "status", "--porcelain"),
            text=True,
            capture_output=True,
            check=True,
        ).stdout
        if not dirty:
            return revision
        working_tree_hash = hashlib.sha256(dirty.encode("utf-8")).hexdigest()[:16]
        return f"{revision}+dirty:{working_tree_hash}"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
