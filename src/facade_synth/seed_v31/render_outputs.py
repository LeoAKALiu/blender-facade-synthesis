from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from facade_synth.seed_v31.blender_scene import FacadeSceneSpec, Point3D, Window3D
from facade_synth.seed_v31.blenderproc_facade_mvp import GeneratedSample
from facade_synth.seed_v31.projection import draw_line_heatmap, draw_polygon_mask, project_points_px
from facade_synth.seed_v31.schema import SCHEMA_VERSION, validate_metadata


Point2D = tuple[int, int]


def build_projected_sample(
    spec: FacadeSceneSpec,
    *,
    width: int,
    height: int,
    rgb: np.ndarray | None = None,
) -> GeneratedSample:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    _validate_spec_window_instance_ids(spec.windows)

    facade_polygon = _project_polygon(spec.facade_corners_world, spec=spec, width=width, height=height)
    facade_mask = draw_polygon_mask(width, height, facade_polygon)
    facade_bbox = _mask_bbox(facade_mask, width=width, height=height, label="facade_mask")

    floorline_polylines = _project_segments(spec.floorline_segments_world, spec=spec, width=width, height=height)
    roofline_polyline = _project_segment(spec.roofline_segment_world, spec=spec, width=width, height=height)
    groundline_polyline = _project_segment(spec.groundline_segment_world, spec=spec, width=width, height=height)

    floorline_heatmap = draw_line_heatmap(width, height, floorline_polylines, sigma=2.0)
    roofline_heatmap = draw_line_heatmap(width, height, [roofline_polyline], sigma=2.0)
    groundline_heatmap = draw_line_heatmap(width, height, [groundline_polyline], sigma=2.0)

    window_semantic_mask = np.zeros((height, width), dtype=np.uint8)
    window_instance_mask = np.zeros((height, width), dtype=np.uint8)
    window_instances: list[dict[str, Any]] = []
    for window in spec.windows:
        window_polygon = _project_polygon(window.corners_world, spec=spec, width=width, height=height)
        window_mask = draw_polygon_mask(width, height, window_polygon)
        if not np.any(window_mask):
            continue
        window_semantic_mask[window_mask > 0] = 255
        window_instance_mask[window_mask > 0] = np.uint8(window.instance_id)
        window_instances.append(_window_metadata(window, window_mask, width=width, height=height))

    depth = _depth_from_facade_plane(spec, width=width, height=height, facade_mask=facade_mask)
    normal = _normal_from_facade_plane(spec, width=width, height=height, facade_mask=facade_mask)
    rgb_array = _coerce_rgb(rgb, width=width, height=height)
    if rgb_array is None:
        rgb_array = _fallback_rgb(
            spec,
            width=width,
            height=height,
            facade_mask=facade_mask,
            window_semantic_mask=window_semantic_mask,
            depth=depth,
        )

    metadata = _metadata(
        spec,
        width=width,
        height=height,
        facade_bbox=facade_bbox,
        floorline_polylines=floorline_polylines,
        roofline_polyline=roofline_polyline,
        groundline_polyline=groundline_polyline,
        window_instances=window_instances,
    )
    validate_metadata(metadata)
    _validate_artifact_arrays(
        rgb=rgb_array,
        facade_mask=facade_mask,
        window_semantic_mask=window_semantic_mask,
        window_instance_mask=window_instance_mask,
        floorline_heatmap=floorline_heatmap,
        roofline_heatmap=roofline_heatmap,
        groundline_heatmap=groundline_heatmap,
        depth=depth,
        normal=normal,
        width=width,
        height=height,
    )
    _validate_window_instance_ids_match_metadata(window_instance_mask, window_instances)

    return GeneratedSample(
        metadata=metadata,
        rgb=rgb_array,
        facade_mask=facade_mask,
        window_semantic_mask=window_semantic_mask,
        window_instance_mask=window_instance_mask,
        floorline_heatmap=floorline_heatmap,
        roofline_heatmap=roofline_heatmap,
        groundline_heatmap=groundline_heatmap,
        depth=depth,
        normal=normal,
    )


def _project_polygon(
    points_world: Sequence[Point3D],
    *,
    spec: FacadeSceneSpec,
    width: int,
    height: int,
) -> list[Point2D]:
    if len(points_world) < 3:
        raise ValueError("polygon must have at least three points")
    return _project_points(points_world, spec=spec, width=width, height=height)


def _project_segments(
    segments_world: Sequence[tuple[Point3D, Point3D]],
    *,
    spec: FacadeSceneSpec,
    width: int,
    height: int,
) -> list[list[Point2D]]:
    polylines = [_project_segment(segment, spec=spec, width=width, height=height) for segment in segments_world]
    if not polylines:
        raise ValueError("at least one floorline segment is required")
    return polylines


def _project_segment(
    segment_world: tuple[Point3D, Point3D],
    *,
    spec: FacadeSceneSpec,
    width: int,
    height: int,
) -> list[Point2D]:
    return _project_points(segment_world, spec=spec, width=width, height=height)


def _project_points(
    points_world: Sequence[Point3D],
    *,
    spec: FacadeSceneSpec,
    width: int,
    height: int,
) -> list[Point2D]:
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


def _mask_bbox(mask: np.ndarray, *, width: int, height: int, label: str) -> list[int]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        raise ValueError(f"{label} must contain at least one pixel")
    x0 = int(np.min(xs))
    y0 = int(np.min(ys))
    x1 = min(int(np.max(xs)) + 1, width)
    y1 = min(int(np.max(ys)) + 1, height)
    if x0 >= x1 or y0 >= y1:
        raise ValueError(f"{label} has an invalid bounding box")
    return [x0, y0, x1, y1]


def _coerce_rgb(rgb: np.ndarray | None, *, width: int, height: int) -> np.ndarray | None:
    if rgb is None:
        return None
    if not isinstance(rgb, np.ndarray):
        raise ValueError("rgb must be a np.uint8 array")
    if rgb.shape != (height, width, 3):
        raise ValueError(f"rgb must have shape {(height, width, 3)}")
    if rgb.dtype != np.uint8:
        raise ValueError("rgb must be a np.uint8 array")
    return rgb


def _fallback_rgb(
    spec: FacadeSceneSpec,
    *,
    width: int,
    height: int,
    facade_mask: np.ndarray,
    window_semantic_mask: np.ndarray,
    depth: np.ndarray,
) -> np.ndarray:
    sky_top, sky_bottom = _lighting_sky_colors(spec.lighting_variant)
    gradient = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    image = sky_top * (1.0 - gradient) + sky_bottom * gradient
    rgb = np.repeat(image, width, axis=1)

    facade_color = _material_color(spec.material_variant)
    facade = facade_mask > 0
    if np.any(facade):
        facade_depth = depth[facade]
        depth_span = max(float(np.ptp(facade_depth)), 1e-6)
        shade = 1.04 - 0.14 * ((depth - float(np.min(facade_depth))) / depth_span)
        vertical = np.linspace(0.94, 1.08, height, dtype=np.float32)[:, None]
        facade_rgb = facade_color[None, None, :] * shade[:, :, None] * vertical[:, :, None]
        rgb[facade] = facade_rgb[facade]

    window = window_semantic_mask > 0
    if np.any(window):
        row_tint = np.linspace(0.86, 1.12, height, dtype=np.float32)[:, None]
        col_tint = np.linspace(0.94, 1.08, width, dtype=np.float32)[None, :]
        glass = np.array([42.0, 68.0, 92.0], dtype=np.float32)
        glass_rgb = glass[None, None, :] * row_tint[:, :, None] * col_tint[:, :, None]
        rgb[window] = glass_rgb[window]

    return np.clip(rgb, 0, 255).astype(np.uint8)


def _depth_from_facade_plane(
    spec: FacadeSceneSpec,
    *,
    width: int,
    height: int,
    facade_mask: np.ndarray,
) -> np.ndarray:
    world_to_camera = np.asarray(spec.camera_world_to_camera, dtype=np.float64)
    cam_to_world = _invert_matrix4(world_to_camera)
    intrinsics = spec.camera_intrinsics.as_dict()
    plane_point, plane_normal = _facade_plane(spec)

    y_indices, x_indices = np.mgrid[0:height, 0:width]
    x_cam = (x_indices.astype(np.float64) + 0.5 - float(intrinsics["cx"])) / float(intrinsics["fx"])
    y_cam = (float(intrinsics["cy"]) - (y_indices.astype(np.float64) + 0.5)) / float(intrinsics["fy"])
    directions_camera = np.stack([x_cam, y_cam, np.ones_like(x_cam)], axis=-1)
    directions_world = directions_camera @ cam_to_world[:3, :3].T
    origin_world = cam_to_world[:3, 3]

    denominator = np.tensordot(directions_world, plane_normal, axes=([-1], [0]))
    numerator = float(np.dot(plane_point - origin_world, plane_normal))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = numerator / denominator
    intersections_world = origin_world + directions_world * t[:, :, None]
    intersections_h = np.concatenate(
        [intersections_world, np.ones((height, width, 1), dtype=np.float64)],
        axis=2,
    )
    camera_points = intersections_h @ world_to_camera.T
    facade_depth = camera_points[:, :, 2]

    facade = facade_mask > 0
    invalid_facade = facade & (
        ~np.isfinite(facade_depth)
        | (facade_depth <= 0.0)
        | ~np.isfinite(t)
        | (t <= 0.0)
    )
    if np.any(invalid_facade):
        raise ValueError("depth contains invalid facade intersections")
    if not np.any(facade):
        raise ValueError("depth requires a non-empty facade mask")

    facade_values = facade_depth[facade]
    far_depth = float(np.max(facade_values) + max(spec.depth_m, 10.0))
    depth = np.full((height, width), far_depth, dtype=np.float64)
    depth[facade] = facade_values
    return depth.astype(np.float32)


def _normal_from_facade_plane(
    spec: FacadeSceneSpec,
    *,
    width: int,
    height: int,
    facade_mask: np.ndarray,
) -> np.ndarray:
    _, normal_world = _facade_plane(spec)
    world_to_camera = np.asarray(spec.camera_world_to_camera, dtype=np.float64)
    normal_camera = world_to_camera[:3, :3] @ normal_world
    normal_camera = _normalized(normal_camera).astype(np.float32)

    normal = np.zeros((height, width, 3), dtype=np.float32)
    normal[:, :, 2] = 1.0
    normal[facade_mask > 0] = normal_camera
    return normal


def _metadata(
    spec: FacadeSceneSpec,
    *,
    width: int,
    height: int,
    facade_bbox: list[int],
    floorline_polylines: list[list[Point2D]],
    roofline_polyline: list[Point2D],
    groundline_polyline: list[Point2D],
    window_instances: list[dict[str, Any]],
) -> dict[str, Any]:
    sample_id = spec.sample_id
    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": sample_id,
        "image": {"width": width, "height": height, "rgb_path": f"images/{sample_id}_rgb.png"},
        "labels": {
            "facade_mask_path": f"masks/{sample_id}_facade_mask.png",
            "window_semantic_mask_path": f"masks/{sample_id}_window_semantic_mask.png",
            "window_instance_mask_path": f"masks/{sample_id}_window_instance_mask.png",
            "floorline_heatmap_path": f"heatmaps/{sample_id}_floorline_heatmap.png",
            "roofline_heatmap_path": f"heatmaps/{sample_id}_roofline_heatmap.png",
            "groundline_heatmap_path": f"heatmaps/{sample_id}_groundline_heatmap.png",
            "depth_path": f"depth/{sample_id}_depth.npy",
            "normal_path": f"normal/{sample_id}_normal.npy",
        },
        "building": {
            "archetype": spec.archetype,
            "floor_count_true": spec.floor_count,
            "floor_count_visible": spec.floor_count,
            "story_height_m": spec.story_height_m,
            "width_m": spec.width_m,
            "depth_m": spec.depth_m,
            "roof_type": spec.roof_type,
            "facade_bbox_px": facade_bbox,
            "occlusion_ratio": 0.0,
        },
        "windows": {
            "rows": spec.floor_count,
            "columns": spec.columns,
            "instance_count": len(window_instances),
            "instances": window_instances,
        },
        "geometry": {
            "floorline_polylines_px": _points_to_json(floorline_polylines),
            "roofline_polyline_px": _points_to_json(roofline_polyline),
            "groundline_polyline_px": _points_to_json(groundline_polyline),
        },
        "camera": {
            "intrinsics": spec.camera_intrinsics.as_dict(),
            "extrinsics_cam_to_world": _invert_matrix4(
                np.asarray(spec.camera_world_to_camera, dtype=np.float64)
            ).tolist(),
            "view": spec.camera_view.as_dict(),
        },
        "generation_params": {
            "seed": spec.seed,
            "material_variant": spec.material_variant,
            "lighting_variant": spec.lighting_variant,
            "occluder_variant": "none",
        },
    }


def _window_metadata(window: Window3D, mask: np.ndarray, *, width: int, height: int) -> dict[str, Any]:
    return {
        "id": window.instance_id,
        "floor_index": window.floor_index,
        "column_index": window.column_index,
        "bbox_px": _mask_bbox(mask, width=width, height=height, label=f"window {window.instance_id}"),
        "visible_fraction": 1.0,
        "occluded": False,
    }


def _validate_spec_window_instance_ids(windows: Sequence[Window3D]) -> None:
    for window in windows:
        if window.instance_id < 1 or window.instance_id > 255:
            raise ValueError("window instance IDs must be in the 1..255 range for window_instance_mask")


def _validate_window_instance_ids_match_metadata(
    window_instance_mask: np.ndarray,
    window_instances: list[dict[str, Any]],
) -> None:
    mask_ids = {int(value) for value in np.unique(window_instance_mask) if int(value) != 0}
    declared_ids = {int(instance["id"]) for instance in window_instances}
    if mask_ids != declared_ids:
        raise ValueError(
            "window_instance_mask IDs do not match declared metadata window instances: "
            f"mask={sorted(mask_ids)} declared={sorted(declared_ids)}"
        )


def _points_to_json(points: Sequence[Any]) -> list[Any]:
    result: list[Any] = []
    for point in points:
        if len(point) == 2 and isinstance(point[0], int):
            result.append([int(point[0]), int(point[1])])
        else:
            result.append(_points_to_json(point))
    return result


def _facade_plane(spec: FacadeSceneSpec) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(spec.facade_corners_world, dtype=np.float64)
    normal = np.cross(points[1] - points[0], points[2] - points[0])
    return points[0], _normalized(normal)


def _invert_matrix4(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float64)
    if array.shape != (4, 4):
        raise ValueError("matrix must be 4x4")
    inverse = np.linalg.inv(array)
    if not np.isfinite(inverse).all():
        raise ValueError("matrix inverse must be finite")
    return inverse


def _normalized(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("cannot normalize zero-length vector")
    return np.asarray(vector, dtype=np.float64) / norm


def _lighting_sky_colors(lighting_variant: str) -> tuple[np.ndarray, np.ndarray]:
    palettes = {
        "overcast": ([190.0, 202.0, 211.0], [154.0, 166.0, 174.0]),
        "morning_side": ([212.0, 204.0, 186.0], [160.0, 177.0, 190.0]),
        "late_afternoon": ([220.0, 188.0, 156.0], [154.0, 151.0, 164.0]),
        "soft_front": ([196.0, 212.0, 224.0], [146.0, 166.0, 181.0]),
    }
    top, bottom = palettes.get(lighting_variant, palettes["overcast"])
    return np.asarray(top, dtype=np.float32), np.asarray(bottom, dtype=np.float32)


def _material_color(material_variant: str) -> np.ndarray:
    colors = {
        "concrete_light": [174.0, 170.0, 160.0],
        "brick_warm": [158.0, 104.0, 82.0],
        "stucco_cool": [158.0, 171.0, 176.0],
        "painted_panel": [140.0, 154.0, 145.0],
    }
    return np.asarray(colors.get(material_variant, colors["concrete_light"]), dtype=np.float32)


def _validate_artifact_arrays(
    *,
    rgb: np.ndarray,
    facade_mask: np.ndarray,
    window_semantic_mask: np.ndarray,
    window_instance_mask: np.ndarray,
    floorline_heatmap: np.ndarray,
    roofline_heatmap: np.ndarray,
    groundline_heatmap: np.ndarray,
    depth: np.ndarray,
    normal: np.ndarray,
    width: int,
    height: int,
) -> None:
    expected_gray_shape = (height, width)
    if rgb.shape != (height, width, 3) or rgb.dtype != np.uint8:
        raise ValueError("rgb must be a uint8 HxWx3 array")
    for name, array in (
        ("facade_mask", facade_mask),
        ("window_semantic_mask", window_semantic_mask),
        ("window_instance_mask", window_instance_mask),
        ("floorline_heatmap", floorline_heatmap),
        ("roofline_heatmap", roofline_heatmap),
        ("groundline_heatmap", groundline_heatmap),
    ):
        if array.shape != expected_gray_shape or array.dtype != np.uint8:
            raise ValueError(f"{name} must be a uint8 HxW array")
    if depth.shape != expected_gray_shape or depth.dtype != np.float32 or not np.isfinite(depth).all():
        raise ValueError("depth must be a finite float32 HxW array")
    if normal.shape != (height, width, 3) or normal.dtype != np.float32 or not np.isfinite(normal).all():
        raise ValueError("normal must be a finite float32 HxWx3 array")

