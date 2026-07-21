from __future__ import annotations

import copy
from typing import Any, Sequence

import numpy as np

from facade_synth.seed_v31.blender_scene import FacadeSceneSpec, Point3D
from facade_synth.seed_v31.blenderproc_facade_mvp import GeneratedSample
from facade_synth.seed_v31.projection import draw_polygon_mask, project_points_px
from facade_synth.seed_v31.render_outputs import build_projected_sample
from facade_synth.seed_v31.schema import validate_metadata
from facade_synth.seed_v31.structure_scene import FacadeStructureSpec, FloorRegion3D, StructuredWindow3D


Point2D = tuple[int, int]


def build_projected_structure_sample(
    spec: FacadeStructureSpec,
    *,
    width: int,
    height: int,
    rgb: np.ndarray | None = None,
    component_semantic_mask: np.ndarray | None = None,
    scene_truth: Any | None = None,
    asset_fingerprints: tuple[str, ...] = (),
) -> GeneratedSample:
    base = build_projected_sample(spec.label_scene_spec, width=width, height=height, rgb=rgb)
    metadata = copy.deepcopy(base.metadata)
    floor_regions = _sorted_contiguous_floor_regions(spec)

    metadata["generation_params"]["structure_variant"] = spec.structure_variant
    metadata["generation_params"]["target_domain"] = "china_post_2000_urban_facades"
    metadata["generation_params"]["asset_fingerprints"] = list(asset_fingerprints)
    if scene_truth is not None:
        metadata["generation_params"]["lighting_recipe"] = dict(scene_truth.lighting_recipe)
        metadata["generation_params"]["occluder_variant"] = str(scene_truth.occluder_variant)
        metadata["building"]["occlusion_ratio"] = round(float(scene_truth.occlusion_ratio), 4)
        metadata["scene_truth"] = {
            "component_mask_origin": "blender_object_index_pass",
            "component_class_ids": {
                name: index
                for index, name in enumerate(
                    (
                        "facade_wall", "window_glass", "window_frame", "door", "balcony",
                        "floor_band", "podium_storefront", "roof_parapet", "background",
                    )
                )
            },
            "visibility": {key: round(float(value), 6) for key, value in scene_truth.visibility.items()},
        }
    if component_semantic_mask is not None:
        expected = (height, width)
        if component_semantic_mask.shape != expected:
            raise ValueError(f"component semantic mask must have shape {expected}")
        metadata["labels"]["component_semantic_mask_path"] = f"masks/{spec.sample_id}_component_semantic_mask.png"
    metadata["geometry"]["floor_index_polygons_px"] = [
        _points_to_json(
            _project_nonempty_polygon(
                floor.corners_world,
                spec=spec.label_scene_spec,
                width=width,
                height=height,
                label=f"floor {floor.floor_index}",
            )
        )
        for floor in floor_regions
    ]
    metadata["geometry"]["floor_visibility_fraction"] = [
        float(floor.visibility_fraction) for floor in floor_regions
    ]

    structured_by_id = {window.instance_id: window for window in spec.windows}
    for instance in metadata["windows"]["instances"]:
        structured_window = structured_by_id.get(int(instance["id"]))
        if structured_window is None:
            raise ValueError(f"missing StructuredWindow3D for metadata instance {instance['id']}")
        _augment_window_instance(
            instance,
            structured_window,
            spec=spec.label_scene_spec,
            width=width,
            height=height,
        )

    facade_mask = base.facade_mask
    window_semantic_mask = base.window_semantic_mask
    window_instance_mask = base.window_instance_mask
    floorline_heatmap = base.floorline_heatmap
    roofline_heatmap = base.roofline_heatmap
    if component_semantic_mask is not None:
        facade_mask, window_semantic_mask, window_instance_mask, floorline_heatmap, roofline_heatmap = (
            _apply_visible_scene_truth_to_task_labels(
                base=base,
                metadata=metadata,
                component_semantic_mask=component_semantic_mask,
            )
        )

    validate_metadata(metadata)
    return GeneratedSample(
        metadata=metadata,
        rgb=base.rgb,
        facade_mask=facade_mask,
        window_semantic_mask=window_semantic_mask,
        window_instance_mask=window_instance_mask,
        floorline_heatmap=floorline_heatmap,
        roofline_heatmap=roofline_heatmap,
        groundline_heatmap=base.groundline_heatmap,
        depth=base.depth,
        normal=base.normal,
    )


def _apply_visible_scene_truth_to_task_labels(
    *,
    base: GeneratedSample,
    metadata: dict[str, Any],
    component_semantic_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Remove foreground-occluded evidence from all published task labels."""

    class_ids = {
        "facade_wall": 0,
        "window_glass": 1,
        "floor_band": 5,
        "roof_parapet": 7,
        "background": 8,
    }
    visible_facade = component_semantic_mask != class_ids["background"]
    window_semantic = np.where(
        component_semantic_mask == class_ids["window_glass"], 255, 0
    ).astype(np.uint8)
    instances = np.where(window_semantic > 0, base.window_instance_mask, 0).astype(np.uint8)
    visible_ids = {int(value) for value in np.unique(instances) if int(value) != 0}
    metadata["windows"]["instances"] = [
        instance for instance in metadata["windows"]["instances"] if int(instance["id"]) in visible_ids
    ]
    metadata["windows"]["instance_count"] = len(metadata["windows"]["instances"])
    floorlines = np.where(component_semantic_mask == class_ids["floor_band"], 255, 0).astype(np.uint8)
    roofline = np.where(component_semantic_mask == class_ids["roof_parapet"], 255, 0).astype(np.uint8)
    return (
        np.where(visible_facade, 255, 0).astype(np.uint8),
        window_semantic,
        instances,
        floorlines,
        roofline,
    )


def _sorted_contiguous_floor_regions(spec: FacadeStructureSpec) -> tuple[FloorRegion3D, ...]:
    regions = tuple(sorted(spec.floor_regions, key=lambda floor: floor.floor_index))
    expected = tuple(range(spec.floor_count))
    actual = tuple(floor.floor_index for floor in regions)
    if actual != expected:
        raise ValueError(
            "floor_regions must have unique contiguous floor_index values matching "
            f"0..{spec.floor_count - 1}: got {actual}"
        )
    return regions


def _augment_window_instance(
    instance: dict[str, Any],
    structured_window: StructuredWindow3D,
    *,
    spec: FacadeSceneSpec,
    width: int,
    height: int,
) -> None:
    opening_polygon = _project_nonempty_polygon(
        structured_window.opening_corners_world,
        spec=spec,
        width=width,
        height=height,
        label=f"window {structured_window.instance_id} opening",
    )
    glass_polygon = _project_nonempty_polygon(
        structured_window.glass_window.corners_world,
        spec=spec,
        width=width,
        height=height,
        label=f"window {structured_window.instance_id} glass",
    )

    instance["opening_bbox_px"] = _polygon_bbox(
        opening_polygon,
        width=width,
        height=height,
        label=f"window {structured_window.instance_id} opening",
    )
    instance["glass_bbox_px"] = _polygon_bbox(
        glass_polygon,
        width=width,
        height=height,
        label=f"window {structured_window.instance_id} glass",
    )
    instance["visibility_fraction"] = float(structured_window.visibility_fraction)
    instance["mullion_count"] = int(structured_window.mullion_count)


def _project_nonempty_polygon(
    points_world: Sequence[Point3D],
    *,
    spec: FacadeSceneSpec,
    width: int,
    height: int,
    label: str,
) -> list[Point2D]:
    polygon = _project_polygon(points_world, spec=spec, width=width, height=height)
    mask = draw_polygon_mask(width, height, polygon)
    if not np.any(mask):
        raise ValueError(f"{label} polygon must contain at least one pixel")
    return polygon


def _project_polygon(
    points_world: Sequence[Point3D],
    *,
    spec: FacadeSceneSpec,
    width: int,
    height: int,
) -> list[Point2D]:
    if len(points_world) < 3:
        raise ValueError("polygon must have at least three points")
    projected = project_points_px(
        np.asarray(points_world, dtype=np.float64),
        np.asarray(spec.camera_world_to_camera, dtype=np.float64),
        spec.camera_intrinsics.as_dict(),
    )
    return [_projected_point_to_pixel(point, width=width, height=height) for point in projected]


def _projected_point_to_pixel(point: np.ndarray, *, width: int, height: int) -> Point2D:
    x = float(point[0])
    y = float(point[1])
    if (
        not np.isfinite(x)
        or not np.isfinite(y)
        or x < 0.0
        or y < 0.0
        or x > width - 1
        or y > height - 1
    ):
        raise ValueError(
            "projected point is outside image bounds: "
            f"point=({x:.6g}, {y:.6g}) bounds=(0..{width - 1}, 0..{height - 1})"
        )
    return int(round(x)), int(round(y))


def _polygon_bbox(polygon: Sequence[Point2D], *, width: int, height: int, label: str) -> list[int]:
    mask = draw_polygon_mask(width, height, polygon)
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        raise ValueError(f"{label} mask must contain at least one pixel")
    x0 = int(np.min(xs))
    y0 = int(np.min(ys))
    x1 = min(int(np.max(xs)) + 1, width)
    y1 = min(int(np.max(ys)) + 1, height)
    if x0 >= x1 or y0 >= y1:
        raise ValueError(f"{label} has an invalid bounding box")
    return [x0, y0, x1, y1]


def _points_to_json(points: Sequence[Point2D]) -> list[list[int]]:
    return [[int(x), int(y)] for x, y in points]
