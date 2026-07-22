from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from facade_synth.seed_v31.blender_scene import (
    Archetype,
    FacadeSceneSpec,
    Point3D,
    Window3D,
    WindowCorners3D,
    sample_scene_spec,
)


StructureVariant = Literal[
    "residential_recessed",
    "balcony_residential",
    "commercial_curtain_wall",
    "podium_mixed_use",
]
STRUCTURE_VARIANTS: tuple[StructureVariant, ...] = (
    "residential_recessed",
    "balcony_residential",
    "commercial_curtain_wall",
    "podium_mixed_use",
)


@dataclass(frozen=True)
class FloorRegion3D:
    floor_index: int
    corners_world: WindowCorners3D
    visibility_fraction: float = 1.0


@dataclass(frozen=True)
class StructuredWindow3D:
    instance_id: int
    floor_index: int
    column_index: int
    opening_corners_world: WindowCorners3D
    glass_window: Window3D
    opening_depth_m: float
    frame_thickness_m: float
    sill_depth_m: float
    mullion_count: int
    visibility_fraction: float = 1.0


@dataclass(frozen=True)
class BalconyModule3D:
    floor_index: int
    column_index: int
    center_world: Point3D
    width_m: float
    depth_m: float
    height_m: float


@dataclass(frozen=True)
class FacadeStructureSpec:
    sample_id: str
    seed: int
    structure_variant: StructureVariant
    label_scene_spec: FacadeSceneSpec
    floor_regions: tuple[FloorRegion3D, ...]
    windows: tuple[StructuredWindow3D, ...]
    balconies: tuple[BalconyModule3D, ...]
    podium_floor_count: int

    @property
    def floor_count(self) -> int:
        return self.label_scene_spec.floor_count

    @property
    def columns(self) -> int:
        return self.label_scene_spec.columns


def sample_structure_scene_spec(
    *,
    sample_id: str,
    seed: int,
    structure_variant: StructureVariant | None = None,
    width_px: int = 768,
    height_px: int = 576,
    view_band: str | None = None,
    lighting_variant: str | None = None,
    material_variant: str | None = None,
) -> FacadeStructureSpec:
    rng = random.Random(seed)
    selected = structure_variant if structure_variant is not None else rng.choice(STRUCTURE_VARIANTS)
    if selected not in STRUCTURE_VARIANTS:
        raise ValueError(f"unsupported structure variant: {selected}")

    base_spec = sample_scene_spec(
        sample_id=sample_id,
        seed=seed,
        archetype=_base_archetype_for_variant(selected),
        width_px=width_px,
        height_px=height_px,
        view_band=view_band,
        lighting_variant=lighting_variant,
        material_variant=material_variant,
    )
    podium_floor_count = _podium_floor_count(selected, base_spec.floor_count, rng)
    structured_windows = _structured_windows(
        base_spec,
        selected,
        rng,
        podium_floor_count=podium_floor_count,
    )
    if not structured_windows:
        raise ValueError("structure scene must contain at least one window")

    label_scene_spec = replace(
        base_spec,
        windows=tuple(window.glass_window for window in structured_windows),
    )

    return FacadeStructureSpec(
        sample_id=sample_id,
        seed=seed,
        structure_variant=selected,
        label_scene_spec=label_scene_spec,
        floor_regions=_floor_regions(label_scene_spec),
        windows=tuple(structured_windows),
        balconies=tuple(_balconies(label_scene_spec, structured_windows, selected, rng)),
        podium_floor_count=podium_floor_count,
    )


def _base_archetype_for_variant(variant: StructureVariant) -> Archetype:
    mapping: dict[StructureVariant, Archetype] = {
        "residential_recessed": "slab_flat_roof",
        "balcony_residential": "slab_flat_roof",
        "commercial_curtain_wall": "commercial_grid",
        "podium_mixed_use": "podium_tower",
    }
    return mapping[variant]


def _structured_windows(
    base_spec: FacadeSceneSpec,
    variant: StructureVariant,
    rng: random.Random,
    *,
    podium_floor_count: int,
) -> list[StructuredWindow3D]:
    windows: list[StructuredWindow3D] = []
    next_id = 1
    missing_probability = {
        "residential_recessed": 0.04,
        "balcony_residential": 0.05,
        "commercial_curtain_wall": 0.01,
        "podium_mixed_use": 0.08,
    }[variant]

    for source in base_spec.windows:
        is_podium_blank = (
            variant == "podium_mixed_use"
            and source.floor_index < podium_floor_count
            and source.column_index % 3 == 1
        )
        if is_podium_blank or rng.random() < missing_probability:
            continue

        opening = source.corners_world
        glass = _inset_window_corners(opening, x_margin=0.08, z_margin=0.10)
        glass_window = Window3D(
            instance_id=next_id,
            floor_index=source.floor_index,
            column_index=source.column_index,
            corners_world=glass,
        )
        windows.append(
            StructuredWindow3D(
                instance_id=next_id,
                floor_index=source.floor_index,
                column_index=source.column_index,
                opening_corners_world=opening,
                glass_window=glass_window,
                opening_depth_m=round(rng.uniform(0.10, 0.28), 3),
                frame_thickness_m=round(rng.uniform(0.05, 0.12), 3),
                sill_depth_m=round(rng.uniform(0.08, 0.22), 3),
                mullion_count=2 if variant == "commercial_curtain_wall" else rng.randrange(0, 2),
            )
        )
        next_id += 1

    return windows


def _inset_window_corners(
    corners: WindowCorners3D,
    *,
    x_margin: float,
    z_margin: float,
) -> WindowCorners3D:
    points = np.asarray(corners, dtype=np.float64)
    x0, y0, z0 = np.min(points, axis=0)
    x1, _y1, z1 = np.max(points, axis=0)
    width = max(float(x1 - x0), 1e-6)
    height = max(float(z1 - z0), 1e-6)
    dx = min(width * x_margin, width * 0.35)
    dz = min(height * z_margin, height * 0.35)
    return (
        (float(x0 + dx), float(y0), float(z0 + dz)),
        (float(x1 - dx), float(y0), float(z0 + dz)),
        (float(x1 - dx), float(y0), float(z1 - dz)),
        (float(x0 + dx), float(y0), float(z1 - dz)),
    )


def _floor_regions(spec: FacadeSceneSpec) -> tuple[FloorRegion3D, ...]:
    left_x = -spec.width_m / 2.0
    right_x = spec.width_m / 2.0
    y = -0.02
    regions: list[FloorRegion3D] = []
    for floor_index in range(spec.floor_count):
        z0 = spec.story_height_m * floor_index
        z1 = spec.story_height_m * (floor_index + 1)
        regions.append(
            FloorRegion3D(
                floor_index=floor_index,
                corners_world=((left_x, y, z0), (right_x, y, z0), (right_x, y, z1), (left_x, y, z1)),
                visibility_fraction=1.0,
            )
        )
    return tuple(regions)


def _balconies(
    spec: FacadeSceneSpec,
    windows: list[StructuredWindow3D],
    variant: StructureVariant,
    rng: random.Random,
) -> list[BalconyModule3D]:
    if variant != "balcony_residential":
        return []

    modules: list[BalconyModule3D] = []
    for window in windows:
        if window.floor_index == 0 or window.column_index % 2 != 0 or rng.random() > 0.45:
            continue
        points = np.asarray(window.opening_corners_world, dtype=np.float64)
        x0, y0, z0 = np.min(points, axis=0)
        x1, _y1, _z1 = np.max(points, axis=0)
        modules.append(
            BalconyModule3D(
                floor_index=window.floor_index,
                column_index=window.column_index,
                center_world=(float((x0 + x1) / 2.0), float(y0 - 0.55), float(z0 - 0.18)),
                width_m=max(float(x1 - x0) * 1.25, 0.6),
                depth_m=0.72,
                height_m=0.12,
            )
        )
    return modules


def _podium_floor_count(variant: StructureVariant, floor_count: int, rng: random.Random) -> int:
    if variant != "podium_mixed_use":
        return 0
    return max(1, min(floor_count - 1, rng.randrange(1, 3)))
