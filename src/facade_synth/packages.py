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
from .seed_v31.projection import draw_polygon_mask, project_points_px
from .seed_v31.structure_scene import sample_structure_scene_spec
from .seed_v31.validate_dataset import validate_dataset


class RenderCancelled(RuntimeGateError):
    """The Worker stopped safely before beginning the next Facade Sample."""


@dataclass(frozen=True)
class PlannedSample:
    sample_id: str
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
                records.append(json.loads(validated_record.read_text(encoding="utf-8")))
                continue
            sample_root.parent.mkdir(parents=True, exist_ok=True)
            summary = self.runtime.run_generator(
                (
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
                    "--render-samples",
                    str(self.render_samples),
                )
            )
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
            blenderproc_version="2.8.0",
            sample_records=tuple(records),
        )


def plan_samples(brief: GenerationBrief) -> list[PlannedSample]:
    """Allocate a deterministic, recipe-owned split and curriculum per sample."""

    uses = _balanced_labels(brief.building_use_distribution, brief.output_target, brief.seed)
    splits = _balanced_labels(brief.split_ratio, brief.output_target, brief.seed + 1)
    daylight_options = (
        ("clear", "overcast")
        if brief.daylight_profile == "controlled_daylight"
        else ("clear", "overcast", "warm_low_angle", "backlit")
    )
    occlusion_options = ("clear", "light_0_15", "moderate_15_30")
    plan: list[PlannedSample] = []
    for index, (building_use, split) in enumerate(zip(uses, splits, strict=True)):
        plan.append(
            PlannedSample(
                sample_id=f"facade_{index:06d}",
                seed=brief.seed + index,
                split=split,
                building_use=building_use,
                structure_variant=_structure_variant(building_use, index),
                view_band=brief.view_family[index % len(brief.view_family)],
                daylight_condition=daylight_options[index % len(daylight_options)],
                occlusion_band=occlusion_options[index % len(occlusion_options)],
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
    annotation_path = package_dir / "annotations" / f"{planned.sample_id}.json"
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    annotation_path.write_text(json.dumps(annotation, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "sample_id": planned.sample_id,
        "recipe_id": f"recipe_{planned.seed}",
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
        "visible_floor_count": metadata["building"]["floor_count_visible"],
        "window_count": metadata["windows"]["instance_count"],
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
        mask_path, class_pixel_counts = _build_component_mask(
            package_dir,
            seed_root,
            planned,
            metadata,
        )
        return {
            "task": brief.task.value,
            "semantic_mask_path": mask_path,
            "classes": {name: index for index, name in enumerate(COMPONENT_CLASSES)},
            "class_pixel_counts": class_pixel_counts,
            "target": "visible_raster_only",
        }
    raise ValueError(f"unsupported task: {brief.task}")


def _build_component_mask(
    package_dir: Path,
    seed_root: Path,
    planned: PlannedSample,
    metadata: Mapping[str, Any],
) -> tuple[str, dict[str, int]]:
    labels = metadata["labels"]
    facade = np.asarray(Image.open(seed_root / labels["facade_mask_path"]).convert("L")) > 0
    windows = np.asarray(Image.open(seed_root / labels["window_semantic_mask_path"]).convert("L")) > 0
    floorlines = np.asarray(Image.open(seed_root / labels["floorline_heatmap_path"]).convert("L")) > 127
    roofline = np.asarray(Image.open(seed_root / labels["roofline_heatmap_path"]).convert("L")) > 127
    component = np.full(
        facade.shape,
        COMPONENT_CLASSES.index("background"),
        dtype=np.uint8,
    )
    component[facade] = COMPONENT_CLASSES.index("facade_wall")
    component[windows] = COMPONENT_CLASSES.index("window_glass")
    spec = sample_structure_scene_spec(
        sample_id="facade_000000",
        seed=planned.seed,
        structure_variant=planned.structure_variant,
        width_px=int(metadata["image"]["width"]),
        height_px=int(metadata["image"]["height"]),
        view_band=planned.view_band,
        lighting_variant=_lighting_variant(planned.daylight_condition),
        material_variant=_material_variant(planned.building_use, planned.seed),
    )
    image_width = int(metadata["image"]["width"])
    image_height = int(metadata["image"]["height"])
    intrinsics = metadata["camera"]["intrinsics"]
    world_to_camera = spec.label_scene_spec.camera_world_to_camera
    for window in spec.windows:
        opening = _project_mask(
            window.opening_corners_world,
            world_to_camera=world_to_camera,
            intrinsics=intrinsics,
            width=image_width,
            height=image_height,
        )
        component[(opening > 0) & ~windows & facade] = COMPONENT_CLASSES.index("window_frame")
    for balcony in spec.balconies:
        half_width = balcony.width_m / 2.0
        y = balcony.center_world[1] - balcony.depth_m / 2.0
        z0 = balcony.center_world[2]
        z1 = z0 + balcony.height_m
        balcony_mask = _project_mask(
            (
                (balcony.center_world[0] - half_width, y, z0),
                (balcony.center_world[0] + half_width, y, z0),
                (balcony.center_world[0] + half_width, y, z1),
                (balcony.center_world[0] - half_width, y, z1),
            ),
            world_to_camera=world_to_camera,
            intrinsics=intrinsics,
            width=image_width,
            height=image_height,
        )
        component[balcony_mask > 0] = COMPONENT_CLASSES.index("balcony")
    if spec.podium_floor_count:
        for polygon in metadata["geometry"].get("floor_index_polygons_px", [])[: spec.podium_floor_count]:
            points = tuple((float(point[0]), float(point[1])) for point in polygon)
            podium_mask = draw_polygon_mask(image_width, image_height, points) > 0
            component[podium_mask & facade & ~windows] = COMPONENT_CLASSES.index("podium_storefront")
    component[floorlines] = COMPONENT_CLASSES.index("floor_band")
    component[roofline] = COMPONENT_CLASSES.index("roof_parapet")
    output = package_dir / "annotations" / f"{planned.sample_id}_components.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(component, mode="L").save(output)
    counts = {
        name: int(np.count_nonzero(component == index))
        for index, name in enumerate(COMPONENT_CLASSES)
    }
    return output.relative_to(package_dir).as_posix(), counts


def _project_mask(
    points: tuple[tuple[float, float, float], ...],
    *,
    world_to_camera: Any,
    intrinsics: Mapping[str, Any],
    width: int,
    height: int,
) -> np.ndarray:
    projected = project_points_px(
        np.asarray(points, dtype=np.float64),
        np.asarray(world_to_camera, dtype=np.float64),
        {key: float(value) for key, value in intrinsics.items()},
    )
    polygon = tuple((float(point[0]), float(point[1])) for point in projected)
    return draw_polygon_mask(width, height, polygon)


def validate_task_records(records: Sequence[Mapping[str, Any]], *, brief: GenerationBrief, package_dir: Path) -> None:
    if len(records) != brief.output_target:
        raise RuntimeGateError("package records do not match confirmed output_target")
    recipes: dict[str, str] = {}
    for record in records:
        if record.get("task") != brief.task.value:
            raise RuntimeGateError("package mixes task datasets")
        if record.get("render_backend") != "blenderproc_blender" or record.get("used_projection_fallback") is not False:
            raise RuntimeGateError("package contains non-BlenderProc or fallback output")
        split = record.get("split")
        if split not in {"train", "validation", "test"}:
            raise RuntimeGateError("package record has an invalid split")
        recipe_id = str(record.get("recipe_id"))
        existing = recipes.setdefault(recipe_id, str(split))
        if existing != split:
            raise RuntimeGateError("a Building Recipe crosses dataset splits")
        for key in ("rgb_path", "annotation_path", "source_metadata_path"):
            if not (package_dir / str(record[key])).exists():
                raise RuntimeGateError(f"package record references missing {key}")


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
        "contact_sheet": "preview/contact_sheet.png",
    }
    (package_dir / "qa_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def fingerprint_local_asset(asset_path: Path) -> str:
    """Return the automatically captured path-plus-content identity for a local asset."""

    path = Path(asset_path).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"{path}:{digest}"


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
        return subprocess.run(
            ("git", "rev-parse", "HEAD"),
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
