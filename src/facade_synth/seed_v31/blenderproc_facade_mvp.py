"""BlenderProc launch marker: import blenderproc is intentionally not executed."""

import argparse
import json
import random
import shutil
import struct
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from facade_synth.seed_v31.schema import (  # noqa: E402
    SCHEMA_VERSION,
    load_metadata,
    validate_metadata,
    write_metadata,
)


@dataclass
class GeneratedSample:
    metadata: dict
    rgb: np.ndarray
    facade_mask: np.ndarray
    window_semantic_mask: np.ndarray
    window_instance_mask: np.ndarray
    floorline_heatmap: np.ndarray
    roofline_heatmap: np.ndarray
    groundline_heatmap: np.ndarray
    depth: np.ndarray
    normal: np.ndarray


_DATASET_DIRS = ("images", "masks", "heatmaps", "depth", "normal", "metadata")
_MATERIALS = ("concrete_light", "brick_warm", "stucco_cool", "painted_panel")
_LIGHTING = ("overcast", "morning_side", "late_afternoon", "soft_front")


def generate_sample(
    *,
    sample_id: str,
    width: int,
    height: int,
    seed: int,
    enable_occluders: bool = False,
) -> GeneratedSample:
    rng = random.Random(seed)
    floor_count_true = rng.randrange(3, 19)
    floor_count_visible = floor_count_true
    columns = rng.randrange(4, 13)

    margin_x = max(8, int(width * rng.uniform(0.07, 0.13)))
    top_y = max(4, int(height * rng.uniform(0.08, 0.16)))
    bottom_y = min(height - 2, int(height * rng.uniform(0.84, 0.94)))
    top_inset = int(width * rng.uniform(0.03, 0.12))
    bottom_skew = int(width * rng.uniform(-0.04, 0.04))
    left_top = min(width - 4, margin_x + top_inset + max(0, bottom_skew))
    right_top = max(left_top + 8, width - margin_x - top_inset + min(0, bottom_skew))
    left_bottom = max(1, margin_x + max(0, -bottom_skew))
    right_bottom = min(width - 2, width - margin_x + min(0, -bottom_skew))
    facade_poly = [(left_top, top_y), (right_top, top_y), (right_bottom, bottom_y), (left_bottom, bottom_y)]

    facade_mask = _draw_polygon_mask(width, height, facade_poly)
    floor_lines = _floor_lines(floor_count_visible, left_top, right_top, left_bottom, right_bottom, top_y, bottom_y)
    roofline = [[int(left_top), int(top_y)], [int(right_top), int(top_y)]]
    groundline = [[int(left_bottom), int(bottom_y)], [int(right_bottom), int(bottom_y)]]

    facade_color = _facade_color(rng)
    rgb = _background(width, height, rng)
    rgb[facade_mask > 0] = facade_color
    _add_facade_texture(rgb, facade_mask, rng)

    window_semantic_mask = np.zeros((height, width), dtype=np.uint8)
    window_instance_mask = np.zeros((height, width), dtype=np.uint8)
    window_instances: list[dict[str, Any]] = []
    instance_id = 1

    for floor_index in range(floor_count_visible):
        y_top = floor_lines[floor_index][0][1]
        y_bottom = floor_lines[floor_index + 1][0][1]
        if y_bottom <= y_top + 2:
            continue
        left_floor_top = _interp(left_top, left_bottom, top_y, bottom_y, y_top)
        right_floor_top = _interp(right_top, right_bottom, top_y, bottom_y, y_top)
        left_floor_bottom = _interp(left_top, left_bottom, top_y, bottom_y, y_bottom)
        right_floor_bottom = _interp(right_top, right_bottom, top_y, bottom_y, y_bottom)
        for column_index in range(columns):
            if rng.random() < 0.08:
                continue
            bbox = _window_bbox(
                column_index,
                columns,
                y_top,
                y_bottom,
                left_floor_top,
                right_floor_top,
                left_floor_bottom,
                right_floor_bottom,
            )
            if bbox is None:
                continue
            x0, y0, x1, y1 = bbox
            window_semantic_mask[y0:y1, x0:x1] = 255
            window_instance_mask[y0:y1, x0:x1] = instance_id
            rgb[y0:y1, x0:x1] = _window_color(rng, floor_index, column_index)
            window_instances.append(
                {
                    "id": instance_id,
                    "floor_index": floor_index,
                    "column_index": column_index,
                    "bbox_px": [x0, y0, x1, y1],
                    "visible_fraction": 1.0,
                    "occluded": False,
                }
            )
            instance_id += 1

    occlusion_ratio = 0.0
    occluder_variant = "none"
    if enable_occluders:
        occlusion_ratio = _apply_experimental_occluder(rgb, facade_mask, rng)
        occluder_variant = "experimental_rect"

    floorline_heatmap = _heatmap(width, height, floor_lines[1:-1], sigma=2.0)
    roofline_heatmap = _heatmap(width, height, [[tuple(roofline[0]), tuple(roofline[1])]], sigma=2.0)
    groundline_heatmap = _heatmap(width, height, [[tuple(groundline[0]), tuple(groundline[1])]], sigma=2.0)
    depth = _depth(width, height, facade_mask, rng)
    normal = _normal(width, height, facade_mask)

    bbox = [
        int(max(0, min(point[0] for point in facade_poly))),
        int(max(0, min(point[1] for point in facade_poly))),
        int(min(width, max(point[0] for point in facade_poly))),
        int(min(height, max(point[1] for point in facade_poly))),
    ]
    metadata = _metadata(
        sample_id=sample_id,
        width=width,
        height=height,
        seed=seed,
        floor_count_true=floor_count_true,
        floor_count_visible=floor_count_visible,
        columns=columns,
        facade_bbox=bbox,
        floor_lines=floor_lines,
        roofline=roofline,
        groundline=groundline,
        window_instances=window_instances,
        occlusion_ratio=occlusion_ratio,
        material_variant=rng.choice(_MATERIALS),
        lighting_variant=rng.choice(_LIGHTING),
        occluder_variant=occluder_variant,
    )
    validate_metadata(metadata)

    return GeneratedSample(
        metadata=metadata,
        rgb=rgb,
        facade_mask=facade_mask,
        window_semantic_mask=window_semantic_mask,
        window_instance_mask=window_instance_mask,
        floorline_heatmap=floorline_heatmap,
        roofline_heatmap=roofline_heatmap,
        groundline_heatmap=groundline_heatmap,
        depth=depth,
        normal=normal,
    )


def write_sample(output_dir: Path | str, sample: GeneratedSample) -> None:
    root = Path(output_dir)
    metadata = sample.metadata
    validate_metadata(metadata)

    _save_rgb(root / metadata["image"]["rgb_path"], sample.rgb)
    _save_gray(root / metadata["labels"]["facade_mask_path"], sample.facade_mask)
    _save_gray(root / metadata["labels"]["window_semantic_mask_path"], sample.window_semantic_mask)
    _save_gray(root / metadata["labels"]["window_instance_mask_path"], sample.window_instance_mask)
    _save_gray(root / metadata["labels"]["floorline_heatmap_path"], sample.floorline_heatmap)
    _save_gray(root / metadata["labels"]["roofline_heatmap_path"], sample.roofline_heatmap)
    _save_gray(root / metadata["labels"]["groundline_heatmap_path"], sample.groundline_heatmap)
    _save_npy(root / metadata["labels"]["depth_path"], sample.depth.astype(np.float32, copy=False))
    _save_npy(root / metadata["labels"]["normal_path"], sample.normal.astype(np.float32, copy=False))

    metadata_path = root / f"metadata/{metadata['sample_id']}_metadata.json"
    write_metadata(metadata_path, metadata)
    validate_metadata(load_metadata(metadata_path))


def write_manifest(output_dir: Path | str, samples: list[GeneratedSample]) -> None:
    root = Path(output_dir)
    manifest_path = root / "manifest.jsonl"
    lines = []
    for sample in samples:
        sample_id = sample.metadata["sample_id"]
        entry = {"sample_id": sample_id, "metadata_path": f"metadata/{sample_id}_metadata.json"}
        lines.append(json.dumps(entry, sort_keys=True, allow_nan=False, separators=(",", ":")))
    manifest_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def prepare_output_dir(output_dir: Path | str) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for name in _DATASET_DIRS:
        path = root / name
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
    preview_dir = root / "preview"
    if preview_dir.exists():
        shutil.rmtree(preview_dir)
    manifest_path = root / "manifest.jsonl"
    if manifest_path.exists():
        manifest_path.unlink()
    return root


def generate_dataset(
    *,
    output_dir: Path | str,
    count: int,
    width: int,
    height: int,
    seed: int,
    enable_occluders: bool = False,
) -> list[GeneratedSample]:
    if count < 0:
        raise ValueError("count must be non-negative")
    if width < 32 or height < 32:
        raise ValueError("width and height must be at least 32 pixels")

    root = prepare_output_dir(output_dir)
    samples: list[GeneratedSample] = []
    for index in range(count):
        sample_id = f"facade_{index:06d}"
        sample = generate_sample(
            sample_id=sample_id,
            width=width,
            height=height,
            seed=seed + index,
            enable_occluders=enable_occluders,
        )
        write_sample(root, sample)
        samples.append(sample)
    write_manifest(root, samples)
    return samples


def _metadata(
    *,
    sample_id: str,
    width: int,
    height: int,
    seed: int,
    floor_count_true: int,
    floor_count_visible: int,
    columns: int,
    facade_bbox: list[int],
    floor_lines: list[list[tuple[int, int]]],
    roofline: list[list[int]],
    groundline: list[list[int]],
    window_instances: list[dict[str, Any]],
    occlusion_ratio: float,
    material_variant: str,
    lighting_variant: str,
    occluder_variant: str,
) -> dict[str, Any]:
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
            "archetype": "projected_trapezoid",
            "floor_count_true": floor_count_true,
            "floor_count_visible": floor_count_visible,
            "story_height_m": 3.0,
            "width_m": round(columns * 2.8, 2),
            "depth_m": 10.0,
            "roof_type": "flat",
            "facade_bbox_px": facade_bbox,
            "occlusion_ratio": round(occlusion_ratio, 4),
        },
        "windows": {
            "rows": floor_count_visible,
            "columns": columns,
            "instance_count": len(window_instances),
            "instances": window_instances,
        },
        "geometry": {
            "floorline_polylines_px": [[[x, y] for x, y in line] for line in floor_lines],
            "roofline_polyline_px": roofline,
            "groundline_polyline_px": groundline,
        },
        "camera": {
            "intrinsics": {
                "fx": float(width * 0.92),
                "fy": float(width * 0.92),
                "cx": float(width / 2.0),
                "cy": float(height / 2.0),
            },
            "extrinsics_cam_to_world": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, -22.0],
                [0.0, 0.0, 1.0, 8.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "view": {"azimuth_deg": 12.0, "elevation_deg": 8.0, "distance_m": 28.0, "focal_length_mm": 35.0},
        },
        "generation_params": {
            "seed": seed,
            "material_variant": material_variant,
            "lighting_variant": lighting_variant,
            "occluder_variant": occluder_variant,
        },
    }


def _draw_polygon_mask(width: int, height: int, points: list[tuple[int, int]]) -> np.ndarray:
    polygon = np.asarray(points, dtype=np.float32)
    y_grid, x_grid = np.mgrid[0:height, 0:width]
    x = x_grid.astype(np.float32) + 0.5
    y = y_grid.astype(np.float32) + 0.5
    inside = np.zeros((height, width), dtype=bool)
    previous = len(polygon) - 1
    for current in range(len(polygon)):
        xi, yi = polygon[current]
        xj, yj = polygon[previous]
        crosses = (yi > y) != (yj > y)
        x_at_y = (xj - xi) * (y - yi) / ((yj - yi) + 1e-6) + xi
        inside ^= crosses & (x < x_at_y)
        previous = current
    return np.where(inside, 255, 0).astype(np.uint8)


def _floor_lines(
    floors: int,
    left_top: int,
    right_top: int,
    left_bottom: int,
    right_bottom: int,
    top_y: int,
    bottom_y: int,
) -> list[list[tuple[int, int]]]:
    lines = []
    for index in range(floors + 1):
        t = index / floors
        y = int(round(top_y + (bottom_y - top_y) * t))
        left_x = int(round(left_top + (left_bottom - left_top) * t))
        right_x = int(round(right_top + (right_bottom - right_top) * t))
        lines.append([(left_x, y), (right_x, y)])
    return lines


def _window_bbox(
    column_index: int,
    columns: int,
    y_top: int,
    y_bottom: int,
    left_floor_top: float,
    right_floor_top: float,
    left_floor_bottom: float,
    right_floor_bottom: float,
) -> tuple[int, int, int, int] | None:
    floor_h = y_bottom - y_top
    width_top = right_floor_top - left_floor_top
    width_bottom = right_floor_bottom - left_floor_bottom
    if floor_h < 4 or min(width_top, width_bottom) < columns * 3:
        return None

    cell_left_t = column_index / columns
    cell_right_t = (column_index + 1) / columns
    pad_x_t = 0.18 / columns
    x0_top = left_floor_top + width_top * (cell_left_t + pad_x_t)
    x1_top = left_floor_top + width_top * (cell_right_t - pad_x_t)
    x0_bottom = left_floor_bottom + width_bottom * (cell_left_t + pad_x_t)
    x1_bottom = left_floor_bottom + width_bottom * (cell_right_t - pad_x_t)
    x0 = int(round(max(x0_top, x0_bottom)))
    x1 = int(round(min(x1_top, x1_bottom)))
    y0 = int(round(y_top + floor_h * 0.28))
    y1 = int(round(y_bottom - floor_h * 0.22))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _interp(top_value: int, bottom_value: int, top_y: int, bottom_y: int, y: int) -> float:
    if bottom_y == top_y:
        return float(top_value)
    t = (y - top_y) / (bottom_y - top_y)
    return top_value + (bottom_value - top_value) * t


def _background(width: int, height: int, rng: random.Random) -> np.ndarray:
    sky_top = np.array([rng.randrange(178, 213), rng.randrange(195, 225), rng.randrange(210, 238)], dtype=np.float32)
    sky_bottom = np.array([rng.randrange(135, 170), rng.randrange(150, 180), rng.randrange(158, 190)], dtype=np.float32)
    gradient = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    image = sky_top * (1.0 - gradient) + sky_bottom * gradient
    return np.repeat(image, width, axis=1).astype(np.uint8)


def _facade_color(rng: random.Random) -> np.ndarray:
    base = np.array([rng.randrange(145, 205), rng.randrange(135, 195), rng.randrange(125, 185)], dtype=np.uint8)
    return base


def _window_color(rng: random.Random, floor_index: int, column_index: int) -> np.ndarray:
    shimmer = (floor_index * 7 + column_index * 11 + rng.randrange(0, 24)) % 42
    return np.array([42 + shimmer, 64 + shimmer, 86 + shimmer], dtype=np.uint8)


def _add_facade_texture(rgb: np.ndarray, facade_mask: np.ndarray, rng: random.Random) -> None:
    height, width = facade_mask.shape
    row_gradient = np.linspace(-14, 16, height, dtype=np.int16)[:, None]
    col_wave = (np.sin(np.linspace(0, 10, width, dtype=np.float32)) * 5).astype(np.int16)[None, :]
    noise = np.array([[rng.randrange(-5, 6) for _ in range(width)] for _ in range(height)], dtype=np.int16)
    texture = row_gradient + col_wave + noise
    mask = facade_mask > 0
    for channel in range(3):
        channel_data = rgb[:, :, channel].astype(np.int16)
        channel_data[mask] = np.clip(channel_data[mask] + texture[mask], 0, 255)
        rgb[:, :, channel] = channel_data.astype(np.uint8)


def _heatmap(
    width: int,
    height: int,
    lines: list[list[tuple[int, int]]],
    *,
    sigma: float,
) -> np.ndarray:
    image = np.zeros((height, width), dtype=np.float32)
    for line in lines:
        _draw_line(image, line[0], line[-1], value=255.0)
    return _gaussian_blur(image, sigma=sigma).astype(np.uint8)


def _depth(width: int, height: int, facade_mask: np.ndarray, rng: random.Random) -> np.ndarray:
    y_gradient = np.linspace(1.8, 4.2, height, dtype=np.float32)[:, None]
    depth = np.repeat(y_gradient, width, axis=1)
    depth += np.float32(rng.uniform(7.0, 13.0))
    depth[facade_mask == 0] += np.float32(20.0)
    return depth.astype(np.float32, copy=False)


def _normal(width: int, height: int, facade_mask: np.ndarray) -> np.ndarray:
    normal = np.zeros((height, width, 3), dtype=np.float32)
    normal[:, :, 2] = 1.0
    normal[facade_mask == 0] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return normal


def _apply_experimental_occluder(rgb: np.ndarray, facade_mask: np.ndarray, rng: random.Random) -> float:
    ys, xs = np.where(facade_mask > 0)
    if len(xs) == 0:
        return 0.0
    x0 = int(np.percentile(xs, rng.uniform(8, 25)))
    x1 = int(np.percentile(xs, rng.uniform(38, 62)))
    y0 = int(np.percentile(ys, rng.uniform(45, 65)))
    y1 = int(np.percentile(ys, rng.uniform(82, 96)))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    rgb[y0:y1, x0:x1] = np.array([45, 60, 42], dtype=np.uint8)
    occluded = np.count_nonzero(facade_mask[y0:y1, x0:x1])
    total = np.count_nonzero(facade_mask)
    return float(occluded / total) if total else 0.0


def _save_rgb(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_png(path, array.astype(np.uint8, copy=False), color_type=2)


def _save_gray(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_png(path, array.astype(np.uint8, copy=False), color_type=0)


def _save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def _draw_line(image: np.ndarray, start: tuple[int, int], end: tuple[int, int], *, value: float) -> None:
    x0, y0 = start
    x1, y1 = end
    steps = max(abs(x1 - x0), abs(y1 - y0)) + 1
    xs = np.rint(np.linspace(x0, x1, steps)).astype(np.int32)
    ys = np.rint(np.linspace(y0, y1, steps)).astype(np.int32)
    valid = (xs >= 0) & (xs < image.shape[1]) & (ys >= 0) & (ys < image.shape[0])
    image[ys[valid], xs[valid]] = np.maximum(image[ys[valid], xs[valid]], value)


def _gaussian_blur(image: np.ndarray, *, sigma: float) -> np.ndarray:
    radius = max(1, int(np.ceil(sigma * 3.0)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(offsets * offsets) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    blurred = _convolve_axis(image.astype(np.float32, copy=False), kernel, axis=1)
    blurred = _convolve_axis(blurred, kernel, axis=0)
    return np.clip(blurred, 0, 255)


def _convolve_axis(image: np.ndarray, kernel: np.ndarray, *, axis: int) -> np.ndarray:
    radius = len(kernel) // 2
    pad_width = [(0, 0), (0, 0)]
    pad_width[axis] = (radius, radius)
    padded = np.pad(image, pad_width, mode="edge")
    output = np.zeros_like(image, dtype=np.float32)
    for kernel_index, weight in enumerate(kernel):
        offset = kernel_index
        if axis == 0:
            output += padded[offset : offset + image.shape[0], :] * weight
        else:
            output += padded[:, offset : offset + image.shape[1]] * weight
    return output


def _write_png(path: Path, array: np.ndarray, *, color_type: int) -> None:
    if color_type == 0:
        if array.ndim != 2:
            raise ValueError("grayscale PNG arrays must be 2D")
        height, width = array.shape
        scanlines = b"".join(b"\x00" + array[row].tobytes() for row in range(height))
    elif color_type == 2:
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("RGB PNG arrays must have shape HxWx3")
        height, width, _ = array.shape
        scanlines = b"".join(b"\x00" + array[row].tobytes() for row in range(height))
    else:
        raise ValueError(f"unsupported PNG color type: {color_type}")

    png = [
        b"\x89PNG\r\n\x1a\n",
        _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)),
        _png_chunk(b"IDAT", zlib.compress(scanlines, level=6)),
        _png_chunk(b"IEND", b""),
    ]
    path.write_bytes(b"".join(png))


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(kind)
    checksum = zlib.crc32(data, checksum) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)


def _script_args(argv: list[str] | None) -> list[str] | None:
    if argv is not None:
        return argv
    if "--" not in sys.argv:
        return None

    separator_index = sys.argv.index("--")
    forwarded = sys.argv[separator_index + 1 :]
    if len(forwarded) >= 2:
        return forwarded[2:]
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the synthetic facade MVP dataset.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enable-occluders", action="store_true")
    args = parser.parse_args(_script_args(argv))

    samples = generate_dataset(
        output_dir=args.output_dir,
        count=args.count,
        width=args.width,
        height=args.height,
        seed=args.seed,
        enable_occluders=args.enable_occluders,
    )
    summary = {"output_dir": str(args.output_dir), "sample_count": len(samples), "seed": args.seed}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
