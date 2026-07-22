from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


Point2D = tuple[float, float]


def project_points_px(points_xyz: np.ndarray, world_to_camera: np.ndarray, intrinsics: dict[str, float]) -> np.ndarray:
    """Project world-space XYZ points into pixels.

    World points are transformed by a world-to-camera matrix where camera +Z is
    forward. Image y follows the existing camera convention:
    ``cy - fy * y / z``.
    """
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz must have shape Nx3")

    matrix = np.asarray(world_to_camera, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError("world_to_camera must be a 4x4 matrix")

    homogeneous = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
    camera = (matrix @ homogeneous.T).T[:, :3]
    z = camera[:, 2]
    if np.any(z <= 1e-6):
        raise ValueError("all points must be in front of the camera")

    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    x_px = fx * (camera[:, 0] / z) + cx
    y_px = cy - fy * (camera[:, 1] / z)
    return np.stack([x_px, y_px], axis=1)


def draw_polygon_mask(width: int, height: int, points: Sequence[Point2D]) -> np.ndarray:
    _validate_dimensions(width, height)

    polygon = np.asarray(points, dtype=np.float64)
    if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
        raise ValueError("points must contain at least three 2D points")

    y_grid, x_grid = np.mgrid[0:height, 0:width]
    x = x_grid.astype(np.float64) + 0.5
    y = y_grid.astype(np.float64) + 0.5
    inside = np.zeros((height, width), dtype=bool)
    previous = len(polygon) - 1
    for current in range(len(polygon)):
        xi, yi = polygon[current]
        xj, yj = polygon[previous]
        crosses = (yi > y) != (yj > y)
        x_at_y = (xj - xi) * (y - yi) / ((yj - yi) + 1e-9) + xi
        inside ^= crosses & (x < x_at_y)
        previous = current
    return np.where(inside, 255, 0).astype(np.uint8)


def draw_line_heatmap(width: int, height: int, polylines: Iterable[Sequence[Point2D]], sigma: float = 2.0) -> np.ndarray:
    _validate_dimensions(width, height)
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    image = np.zeros((height, width), dtype=np.float32)
    for polyline in polylines:
        points = list(polyline)
        for start, end in zip(points, points[1:]):
            _draw_line(image, start, end, value=255.0)
    return _gaussian_blur(image, sigma=sigma).astype(np.uint8)


def _validate_dimensions(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")


def _draw_line(image: np.ndarray, start: Point2D, end: Point2D, *, value: float) -> None:
    x0, y0 = start
    x1, y1 = end
    steps = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    if steps <= 0:
        return
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
        if axis == 0:
            output += padded[kernel_index : kernel_index + image.shape[0], :] * weight
        else:
            output += padded[:, kernel_index : kernel_index + image.shape[1]] * weight
    return output
