from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "synthetic_facade_mvp/v1"

LABEL_PATH_KEYS = (
    "facade_mask_path",
    "window_semantic_mask_path",
    "window_instance_mask_path",
    "floorline_heatmap_path",
    "roofline_heatmap_path",
    "groundline_heatmap_path",
    "depth_path",
    "normal_path",
)
OPTIONAL_LABEL_PATH_KEYS = ("component_semantic_mask_path",)

_TOP_LEVEL_KEYS = frozenset(
    (
        "schema_version",
        "sample_id",
        "image",
        "labels",
        "building",
        "windows",
        "geometry",
        "camera",
        "generation_params",
        "scene_truth",
    )
)
_IMAGE_KEYS = frozenset(("width", "height", "rgb_path"))
_LABEL_KEYS = frozenset((*LABEL_PATH_KEYS, *OPTIONAL_LABEL_PATH_KEYS))
_BUILDING_KEYS = frozenset(
    (
        "archetype",
        "floor_count_true",
        "floor_count_visible",
        "story_height_m",
        "width_m",
        "depth_m",
        "roof_type",
        "facade_bbox_px",
        "occlusion_ratio",
    )
)
_WINDOWS_KEYS = frozenset(("rows", "columns", "instance_count", "instances"))
_WINDOW_INSTANCE_KEYS = frozenset(
    (
        "id",
        "floor_index",
        "column_index",
        "bbox_px",
        "visible_fraction",
        "occluded",
        "opening_bbox_px",
        "glass_bbox_px",
        "visibility_fraction",
        "mullion_count",
    )
)
_GEOMETRY_KEYS = frozenset(
    (
        "floorline_polylines_px",
        "roofline_polyline_px",
        "groundline_polyline_px",
        "floor_index_polygons_px",
        "floor_visibility_fraction",
    )
)
_CAMERA_KEYS = frozenset(("intrinsics", "extrinsics_cam_to_world", "view"))
_CAMERA_INTRINSICS_KEYS = frozenset(("fx", "fy", "cx", "cy"))
_CAMERA_VIEW_KEYS = frozenset(("azimuth_deg", "elevation_deg", "distance_m", "focal_length_mm"))
_GENERATION_PARAMS_KEYS = frozenset(
    (
        "seed",
        "material_variant",
        "lighting_variant",
        "occluder_variant",
        "structure_variant",
        "target_domain",
        "lighting_recipe",
        "asset_fingerprints",
    )
)


class ValidationError(ValueError):
    pass


def load_metadata(path: Path) -> dict[str, Any]:
    try:
        metadata = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except json.JSONDecodeError as exc:
        raise ValidationError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(metadata, dict):
        raise ValidationError("metadata must be an object")
    return metadata


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    """Write strict JSON without implicitly validating the metadata schema."""
    try:
        content = json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    except ValueError as exc:
        raise ValidationError(f"metadata contains non-standard JSON value: {exc}") from exc
    except TypeError as exc:
        raise ValidationError(f"metadata is not JSON serializable: {exc}") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def validate_dataset_path(value: Any, key: str = "path") -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{key} must be a non-empty relative POSIX path")
    if value != value.strip():
        raise ValidationError(f"{key} must not contain leading or trailing whitespace")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValidationError(f"{key} must not contain control characters")
    if value.startswith("/") or "\\" in value or _has_windows_drive_prefix(value):
        raise ValidationError(f"{key} must be a relative POSIX path")

    parts = value.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValidationError(f"{key} must not contain empty, current, or parent path segments")
    return value


def validate_metadata(metadata: dict[str, Any]) -> None:
    if not isinstance(metadata, dict):
        raise ValidationError("metadata must be an object")
    _reject_unknown_keys(metadata, _TOP_LEVEL_KEYS, "metadata")

    _require_value(metadata, "schema_version", SCHEMA_VERSION)
    _require_nonempty_string(metadata, "sample_id")

    image = _require_mapping(metadata, "image")
    _reject_unknown_keys(image, _IMAGE_KEYS, "image")
    image_width = _require_positive_int(image, "width")
    image_height = _require_positive_int(image, "height")
    _require_dataset_path(image, "rgb_path")

    labels = _require_mapping(metadata, "labels")
    _reject_unknown_keys(labels, _LABEL_KEYS, "labels")
    for key in LABEL_PATH_KEYS:
        _require_dataset_path(labels, key)
    if "component_semantic_mask_path" in labels:
        _require_dataset_path(labels, "component_semantic_mask_path")

    building = _require_mapping(metadata, "building")
    _reject_unknown_keys(building, _BUILDING_KEYS, "building")
    _require_nonempty_string(building, "archetype")
    floor_count_true = _require_positive_int(building, "floor_count_true")
    floor_count_visible = _require_nonnegative_int(building, "floor_count_visible")
    if floor_count_visible > floor_count_true:
        raise ValidationError("floor_count_visible must be <= floor_count_true")
    _require_positive_number(building, "story_height_m")
    _require_positive_number(building, "width_m")
    _require_positive_number(building, "depth_m")
    _require_nonempty_string(building, "roof_type")
    _validate_bbox(building.get("facade_bbox_px"), image_width, image_height, "facade_bbox_px")
    _validate_ratio(building.get("occlusion_ratio"), "occlusion_ratio")

    windows = _require_mapping(metadata, "windows")
    _reject_unknown_keys(windows, _WINDOWS_KEYS, "windows")
    rows = _require_nonnegative_int(windows, "rows")
    columns = _require_nonnegative_int(windows, "columns")
    instance_count = _require_nonnegative_int(windows, "instance_count")
    instances = windows.get("instances")
    if not isinstance(instances, list):
        raise ValidationError("windows.instances must be a list")
    if instance_count != len(instances):
        raise ValidationError("windows.instance_count must match windows.instances length")

    seen_ids: set[int] = set()
    for index, instance in enumerate(instances):
        _validate_window_instance(instance, index, image_width, image_height, rows, columns, seen_ids)

    geometry = _require_mapping(metadata, "geometry")
    _reject_unknown_keys(geometry, _GEOMETRY_KEYS, "geometry")
    _validate_polylines(
        geometry.get("floorline_polylines_px"),
        image_width,
        image_height,
        "floorline_polylines_px",
    )
    _validate_polyline(
        geometry.get("roofline_polyline_px"),
        image_width,
        image_height,
        "roofline_polyline_px",
    )
    _validate_polyline(
        geometry.get("groundline_polyline_px"),
        image_width,
        image_height,
        "groundline_polyline_px",
    )
    has_floor_polygons = "floor_index_polygons_px" in geometry
    has_floor_visibility = "floor_visibility_fraction" in geometry
    if has_floor_polygons != has_floor_visibility:
        missing_key = "floor_visibility_fraction" if has_floor_polygons else "floor_index_polygons_px"
        present_key = "floor_index_polygons_px" if has_floor_polygons else "floor_visibility_fraction"
        raise ValidationError(f"{present_key} requires paired {missing_key}")

    floor_polygons = geometry.get("floor_index_polygons_px")
    floor_visibility = geometry.get("floor_visibility_fraction")
    if has_floor_polygons:
        _validate_polygons(floor_polygons, image_width, image_height, "floor_index_polygons_px")
        _validate_ratio_list(floor_visibility, "floor_visibility_fraction")
        if len(floor_polygons) != len(floor_visibility):
            raise ValidationError("floor_visibility_fraction must match floor_index_polygons_px length")

    camera = _require_mapping(metadata, "camera")
    _reject_unknown_keys(camera, _CAMERA_KEYS, "camera")
    intrinsics = _require_mapping(camera, "intrinsics")
    _reject_unknown_keys(intrinsics, _CAMERA_INTRINSICS_KEYS, "camera.intrinsics")
    _require_positive_number(intrinsics, "fx")
    _require_positive_number(intrinsics, "fy")
    _require_number(intrinsics, "cx")
    _require_number(intrinsics, "cy")

    extrinsics = camera.get("extrinsics_cam_to_world")
    if not _is_numeric_matrix4(extrinsics):
        raise ValidationError("extrinsics_cam_to_world must be a 4x4 numeric matrix")

    view = _require_mapping(camera, "view")
    _reject_unknown_keys(view, _CAMERA_VIEW_KEYS, "camera.view")
    _require_number(view, "azimuth_deg")
    _require_number(view, "elevation_deg")
    _require_positive_number(view, "distance_m")
    _require_positive_number(view, "focal_length_mm")

    generation_params = _require_mapping(metadata, "generation_params")
    _reject_unknown_keys(generation_params, _GENERATION_PARAMS_KEYS, "generation_params")
    _require_nonnegative_int(generation_params, "seed")
    _require_nonempty_string(generation_params, "material_variant")
    _require_nonempty_string(generation_params, "lighting_variant")
    _require_nonempty_string(generation_params, "occluder_variant")
    if "structure_variant" in generation_params:
        _require_nonempty_string(generation_params, "structure_variant")
    if "target_domain" in generation_params:
        _require_nonempty_string(generation_params, "target_domain")
    if "lighting_recipe" in generation_params:
        _validate_lighting_recipe(generation_params["lighting_recipe"])
    if "asset_fingerprints" in generation_params:
        value = generation_params["asset_fingerprints"]
        if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
            raise ValidationError("asset_fingerprints must be a list of non-empty strings")
    if "scene_truth" in metadata:
        _validate_scene_truth(metadata["scene_truth"])


def _validate_lighting_recipe(value: Any) -> None:
    if not isinstance(value, dict):
        raise ValidationError("lighting_recipe must be an object")
    required = {"sun_elevation_deg", "relative_azimuth_deg", "energy", "world_strength", "exposure_ev", "colour_temperature_k"}
    if set(value) != required:
        raise ValidationError("lighting_recipe must contain the required actual lighting fields")
    for key in required:
        if not isinstance(value[key], (int, float)) or not math.isfinite(float(value[key])):
            raise ValidationError(f"lighting_recipe.{key} must be finite numeric")


def _validate_scene_truth(value: Any) -> None:
    if not isinstance(value, dict):
        raise ValidationError("scene_truth must be an object")
    if value.get("component_mask_origin") != "blender_object_index_pass":
        raise ValidationError("scene_truth.component_mask_origin must be blender_object_index_pass")
    class_ids = value.get("component_class_ids")
    expected_names = {
        "facade_wall", "window_glass", "window_frame", "door", "balcony",
        "floor_band", "podium_storefront", "roof_parapet", "background",
    }
    if not isinstance(class_ids, dict) or set(class_ids) != expected_names:
        raise ValidationError("scene_truth.component_class_ids must contain the versioned vocabulary")
    if {int(number) for number in class_ids.values()} != set(range(len(expected_names))):
        raise ValidationError("scene_truth.component_class_ids must be contiguous 0..8")
    visibility = value.get("visibility")
    if not isinstance(visibility, dict) or "facade_components" not in visibility:
        raise ValidationError("scene_truth.visibility must include facade_components")
    for key, ratio in visibility.items():
        if not isinstance(key, str) or not isinstance(ratio, (int, float)) or not 0.0 <= float(ratio) <= 1.0:
            raise ValidationError("scene_truth.visibility values must be ratios")


def _require_value(mapping: dict[str, Any], key: str, expected: Any) -> None:
    if mapping.get(key) != expected:
        raise ValidationError(f"{key} must be {expected!r}")


def _require_mapping(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise ValidationError(f"{key} must be an object")
    return value


def _require_nonempty_string(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{key} must be a non-empty string")
    return value


def _require_dataset_path(mapping: dict[str, Any], key: str) -> str:
    return validate_dataset_path(mapping.get(key), key=key)


def _has_windows_drive_prefix(value: str) -> bool:
    return len(value) >= 2 and value[0].isalpha() and value[1] == ":"


def _require_positive_int(mapping: dict[str, Any], key: str) -> int:
    value = mapping.get(key)
    if not _is_int(value) or value <= 0:
        raise ValidationError(f"{key} must be a positive integer")
    return value


def _require_nonnegative_int(mapping: dict[str, Any], key: str) -> int:
    value = mapping.get(key)
    if not _is_int(value) or value < 0:
        raise ValidationError(f"{key} must be a non-negative integer")
    return value


def _require_number(mapping: dict[str, Any], key: str) -> float:
    value = mapping.get(key)
    if not _is_number(value):
        raise ValidationError(f"{key} must be a number")
    return float(value)


def _require_positive_number(mapping: dict[str, Any], key: str) -> float:
    value = mapping.get(key)
    if not _is_number(value) or value <= 0:
        raise ValidationError(f"{key} must be a positive number")
    return float(value)


def _validate_ratio(value: Any, name: str) -> None:
    if not _is_number(value) or not 0.0 <= value <= 1.0:
        raise ValidationError(f"{name} must be a number between 0.0 and 1.0")


def _validate_bbox(value: Any, width: int, height: int, name: str) -> None:
    if not isinstance(value, list) or len(value) != 4 or not all(_is_number(item) for item in value):
        raise ValidationError(f"{name} must be a four-number [x_min, y_min, x_max, y_max] list")

    x_min, y_min, x_max, y_max = (float(item) for item in value)
    if x_min >= x_max or y_min >= y_max:
        raise ValidationError(f"{name} must have increasing x and y bounds")
    if x_min < 0 or y_min < 0 or x_max > width or y_max > height:
        raise ValidationError(f"{name} must lie inside image bounds")


def _validate_window_instance(
    instance: Any,
    index: int,
    width: int,
    height: int,
    rows: int,
    columns: int,
    seen_ids: set[int],
) -> None:
    if not isinstance(instance, dict):
        raise ValidationError(f"windows.instances[{index}] must be an object")
    _reject_unknown_keys(instance, _WINDOW_INSTANCE_KEYS, f"windows.instances[{index}]")

    instance_id = _require_positive_int(instance, "id")
    if instance_id in seen_ids:
        raise ValidationError(f"windows.instances[{index}].id must be unique")
    seen_ids.add(instance_id)

    floor_index = _require_nonnegative_int(instance, "floor_index")
    if floor_index >= rows:
        raise ValidationError(f"windows.instances[{index}].floor_index must be less than rows")

    column_index = _require_nonnegative_int(instance, "column_index")
    if column_index >= columns:
        raise ValidationError(f"windows.instances[{index}].column_index must be less than columns")

    _validate_bbox(instance.get("bbox_px"), width, height, f"windows.instances[{index}].bbox_px")
    _validate_ratio(instance.get("visible_fraction"), f"windows.instances[{index}].visible_fraction")
    if not isinstance(instance.get("occluded"), bool):
        raise ValidationError(f"windows.instances[{index}].occluded must be bool")
    if "opening_bbox_px" in instance:
        _validate_bbox(
            instance.get("opening_bbox_px"),
            width,
            height,
            f"windows.instances[{index}].opening_bbox_px",
        )
    if "glass_bbox_px" in instance:
        _validate_bbox(
            instance.get("glass_bbox_px"),
            width,
            height,
            f"windows.instances[{index}].glass_bbox_px",
        )
    if "visibility_fraction" in instance:
        _validate_ratio(instance.get("visibility_fraction"), f"windows.instances[{index}].visibility_fraction")
    if "mullion_count" in instance:
        _require_nonnegative_int(instance, "mullion_count")


def _validate_polylines(value: Any, width: int, height: int, name: str) -> None:
    if not isinstance(value, list) or not value:
        raise ValidationError(f"{name} must be a non-empty list")
    for index, polyline in enumerate(value):
        _validate_polyline(polyline, width, height, f"{name}[{index}]")


def _validate_polygons(value: Any, width: int, height: int, name: str) -> None:
    if not isinstance(value, list) or not value:
        raise ValidationError(f"{name} must be a non-empty list")
    for index, polygon in enumerate(value):
        _validate_polygon(polygon, width, height, f"{name}[{index}]")


def _validate_polygon(value: Any, width: int, height: int, name: str) -> None:
    if not isinstance(value, list) or len(value) < 3:
        raise ValidationError(f"{name} must contain at least three points")
    for index, point in enumerate(value):
        _validate_point(point, width, height, f"{name}[{index}]")


def _validate_ratio_list(value: Any, name: str) -> None:
    if not isinstance(value, list):
        raise ValidationError(f"{name} must be a list")
    for index, item in enumerate(value):
        _validate_ratio(item, f"{name}[{index}]")


def _validate_polyline(value: Any, width: int, height: int, name: str) -> None:
    if not isinstance(value, list) or len(value) < 2:
        raise ValidationError(f"{name} must contain at least two points")
    for index, point in enumerate(value):
        _validate_point(point, width, height, f"{name}[{index}]")


def _validate_point(value: Any, width: int, height: int, name: str) -> None:
    if not isinstance(value, list) or len(value) != 2 or not all(_is_number(item) for item in value):
        raise ValidationError(f"{name} must be an [x, y] numeric point")

    x, y = (float(item) for item in value)
    if x < 0 or y < 0 or x >= width or y >= height:
        raise ValidationError(f"{name} must lie inside image bounds")


def _is_numeric_matrix4(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 4
        and all(
            isinstance(row, list)
            and len(row) == 4
            and all(_is_number(item) for item in row)
            for row in value
        )
    )


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return False


def _reject_unknown_keys(mapping: dict[str, Any], allowed: frozenset[str], name: str) -> None:
    unknown_keys = [key for key in mapping if key not in allowed]
    if unknown_keys:
        joined = ", ".join(str(key) for key in unknown_keys)
        raise ValidationError(f"{name} contains unknown key(s): {joined}")


def _reject_json_constant(value: str) -> None:
    raise ValidationError(f"unsupported JSON constant: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result
