from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Literal

import numpy as np

from facade_synth.seed_v31.projection import project_points_px


Archetype = Literal["slab_flat_roof", "tower_flat_roof", "podium_tower", "commercial_grid"]
ARCHETYPES: tuple[Archetype, ...] = (
    "slab_flat_roof",
    "tower_flat_roof",
    "podium_tower",
    "commercial_grid",
)
SENSOR_WIDTH_MM = 36.0
Point3D = tuple[float, float, float]
WindowCorners3D = tuple[Point3D, Point3D, Point3D, Point3D]
Matrix4x4 = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]


@dataclass(frozen=True)
class Window3D:
    instance_id: int
    floor_index: int
    column_index: int
    corners_world: WindowCorners3D


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    def as_dict(self) -> dict[str, float]:
        return {"fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy}


@dataclass(frozen=True)
class CameraView:
    azimuth_deg: float
    elevation_deg: float
    distance_m: float
    focal_length_mm: float

    def as_dict(self) -> dict[str, float]:
        return {
            "azimuth_deg": self.azimuth_deg,
            "elevation_deg": self.elevation_deg,
            "distance_m": self.distance_m,
            "focal_length_mm": self.focal_length_mm,
        }


@dataclass(frozen=True)
class FacadeSceneSpec:
    sample_id: str
    seed: int
    archetype: Archetype
    floor_count: int
    columns: int
    story_height_m: float
    width_m: float
    depth_m: float
    roof_type: str
    windows: tuple[Window3D, ...]
    facade_corners_world: WindowCorners3D
    floorline_segments_world: tuple[tuple[Point3D, Point3D], ...]
    roofline_segment_world: tuple[Point3D, Point3D]
    groundline_segment_world: tuple[Point3D, Point3D]
    camera_world_to_camera: Matrix4x4
    camera_intrinsics: CameraIntrinsics
    camera_view: CameraView
    material_variant: str
    lighting_variant: str


def sample_scene_spec(
    *,
    sample_id: str,
    seed: int,
    archetype: Archetype | None = None,
    width_px: int = 768,
    height_px: int = 576,
    view_band: str | None = None,
    lighting_variant: str | None = None,
    material_variant: str | None = None,
) -> FacadeSceneSpec:
    rng = random.Random(seed)
    selected = archetype or rng.choice(ARCHETYPES)
    if selected not in ARCHETYPES:
        raise ValueError(f"unsupported archetype: {selected}")

    if selected == "tower_flat_roof":
        floor_count = rng.randrange(9, 19)
        columns = rng.randrange(4, 8)
        width_m = round(rng.uniform(16.0, 24.0), 3)
    elif selected == "podium_tower":
        floor_count = rng.randrange(6, 16)
        columns = rng.randrange(6, 11)
        width_m = round(rng.uniform(24.0, 38.0), 3)
    elif selected == "commercial_grid":
        floor_count = rng.randrange(4, 13)
        columns = rng.randrange(6, 13)
        width_m = round(rng.uniform(28.0, 46.0), 3)
    else:
        floor_count = rng.randrange(3, 15)
        columns = rng.randrange(5, 12)
        width_m = round(rng.uniform(22.0, 42.0), 3)

    story_height_m = round(rng.uniform(2.8, 4.2), 3)
    depth_m = round(rng.uniform(8.0, 16.0), 3)
    height_m = floor_count * story_height_m
    facade_y = 0.0
    left_x = -width_m / 2.0
    right_x = width_m / 2.0

    facade_corners: WindowCorners3D = (
        (left_x, facade_y, 0.0),
        (right_x, facade_y, 0.0),
        (right_x, facade_y, height_m),
        (left_x, facade_y, height_m),
    )
    floorlines = tuple(
        (
            (left_x, facade_y - 0.02, story_height_m * index),
            (right_x, facade_y - 0.02, story_height_m * index),
        )
        for index in range(1, floor_count)
    )
    roofline: tuple[Point3D, Point3D] = (
        (left_x, facade_y - 0.02, height_m),
        (right_x, facade_y - 0.02, height_m),
    )
    groundline: tuple[Point3D, Point3D] = (
        (left_x, facade_y - 0.02, 0.0),
        (right_x, facade_y - 0.02, 0.0),
    )

    windows = _make_windows(rng, floor_count, columns, width_m, story_height_m, facade_y)
    azimuth_deg, elevation_deg = _camera_angles(rng, view_band)
    distance_m = rng.uniform(30.0, 52.0)
    world_to_camera = _look_at_world_to_camera(
        eye=_camera_eye(width_m, height_m, distance_m, azimuth_deg, elevation_deg),
        target=np.array([0.0, 0.0, height_m * 0.48], dtype=np.float64),
    )
    intrinsics = _fit_intrinsics_to_facade(
        facade_corners=facade_corners,
        world_to_camera=world_to_camera,
        width_px=width_px,
        height_px=height_px,
    )
    view = CameraView(
        azimuth_deg=float(azimuth_deg),
        elevation_deg=float(elevation_deg),
        distance_m=float(distance_m),
        focal_length_mm=float(intrinsics.fx * SENSOR_WIDTH_MM / float(width_px)),
    )

    return FacadeSceneSpec(
        sample_id=sample_id,
        seed=seed,
        archetype=selected,
        floor_count=floor_count,
        columns=columns,
        story_height_m=story_height_m,
        width_m=width_m,
        depth_m=depth_m,
        roof_type="flat",
        windows=tuple(windows),
        facade_corners_world=facade_corners,
        floorline_segments_world=floorlines,
        roofline_segment_world=roofline,
        groundline_segment_world=groundline,
        camera_world_to_camera=tuple(tuple(float(value) for value in row) for row in world_to_camera),
        camera_intrinsics=intrinsics,
        camera_view=view,
        material_variant=material_variant
        or rng.choice(("concrete_light", "brick_warm", "stucco_cool", "painted_panel")),
        lighting_variant=lighting_variant
        or rng.choice(("overcast", "morning_side", "late_afternoon", "soft_front")),
    )


def _camera_angles(rng: random.Random, view_band: str | None) -> tuple[float, float]:
    if view_band is None:
        return rng.uniform(-24.0, 24.0), rng.uniform(5.0, 16.0)
    if view_band == "frontal":
        return rng.uniform(-6.0, 6.0), rng.uniform(5.0, 11.0)
    if view_band == "light_medium_oblique":
        return rng.choice((-1.0, 1.0)) * rng.uniform(10.0, 24.0), rng.uniform(6.0, 14.0)
    if view_band == "strong_oblique":
        return rng.choice((-1.0, 1.0)) * rng.uniform(24.0, 32.0), rng.uniform(7.0, 16.0)
    raise ValueError(f"unsupported view_band: {view_band}")


def _make_windows(
    rng: random.Random,
    floors: int,
    columns: int,
    width_m: float,
    story_height_m: float,
    facade_y: float,
) -> list[Window3D]:
    windows: list[Window3D] = []
    cell_w = width_m / columns
    window_w = cell_w * rng.uniform(0.42, 0.62)
    window_h = story_height_m * rng.uniform(0.34, 0.50)
    instance_id = 1
    for floor_index in range(floors):
        z_center = floor_index * story_height_m + story_height_m * 0.55
        for column_index in range(columns):
            x_center = -width_m / 2.0 + cell_w * (column_index + 0.5)
            x0 = x_center - window_w / 2.0
            x1 = x_center + window_w / 2.0
            z0 = z_center - window_h / 2.0
            z1 = z_center + window_h / 2.0
            corners: WindowCorners3D = (
                (x0, facade_y - 0.04, z0),
                (x1, facade_y - 0.04, z0),
                (x1, facade_y - 0.04, z1),
                (x0, facade_y - 0.04, z1),
            )
            windows.append(
                Window3D(
                    instance_id=instance_id,
                    floor_index=floor_index,
                    column_index=column_index,
                    corners_world=corners,
                )
            )
            instance_id += 1
    return windows


def _fit_intrinsics_to_facade(
    *,
    facade_corners: WindowCorners3D,
    world_to_camera: np.ndarray,
    width_px: int,
    height_px: int,
) -> CameraIntrinsics:
    if width_px <= 0 or height_px <= 0:
        raise ValueError("width_px and height_px must be positive")

    projected_unit = project_points_px(
        np.asarray(facade_corners, dtype=np.float64),
        world_to_camera,
        {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
    )
    margin_px = max(8.0, min(float(width_px), float(height_px)) * 0.08)
    available_x = float(width_px) / 2.0 - margin_px
    available_y = float(height_px) / 2.0 - margin_px
    if available_x <= 0.0 or available_y <= 0.0:
        raise ValueError("width_px and height_px leave no room for framing margin")

    max_abs_x = max(float(np.max(np.abs(projected_unit[:, 0]))), 1e-9)
    max_abs_y = max(float(np.max(np.abs(projected_unit[:, 1]))), 1e-9)
    focal_length_px = min(available_x / max_abs_x, available_y / max_abs_y) * 0.98
    if not math.isfinite(focal_length_px) or focal_length_px <= 0.0:
        raise ValueError("could not fit facade corners inside image")

    return CameraIntrinsics(
        fx=float(focal_length_px),
        fy=float(focal_length_px),
        cx=float(width_px) / 2.0,
        cy=float(height_px) / 2.0,
    )


def _camera_eye(
    width_m: float,
    height_m: float,
    distance_m: float,
    azimuth_deg: float,
    elevation_deg: float,
) -> np.ndarray:
    del width_m
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    x = math.sin(azimuth) * distance_m
    y = -math.cos(azimuth) * distance_m
    z = height_m * 0.42 + math.sin(elevation) * distance_m
    return np.array([x, y, z], dtype=np.float64)


def _look_at_world_to_camera(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)

    rotation = np.stack([right, up, forward], axis=0)
    translation = -rotation @ eye
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def build_blender_scene(spec: FacadeSceneSpec, *, width: int, height: int, render_samples: int) -> None:
    import bpy

    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if render_samples <= 0:
        raise ValueError("render_samples must be positive")

    _clear_blender_scene(bpy)

    scene = bpy.context.scene
    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    try:
        scene.render.engine = "CYCLES"
        scene.cycles.samples = int(render_samples)
        scene.cycles.use_denoising = True
    except (AttributeError, TypeError):
        pass

    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.color = tuple(float(value) for value in _sky_color(spec.lighting_variant))

    _create_body(bpy, spec)
    _create_windows(bpy, spec)
    _create_ground(bpy, spec)
    _create_lighting(bpy, spec)
    _create_camera(bpy, spec)
    bpy.context.view_layer.update()


def render_current_scene_rgb_array(*, width: int, height: int) -> np.ndarray:
    import bpy
    import tempfile
    from pathlib import Path

    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    scene = bpy.context.scene
    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"

    with tempfile.TemporaryDirectory(prefix="facade_v2_render_") as temp_dir:
        render_path = Path(temp_dir) / "rgb.png"
        scene.render.filepath = str(render_path)
        bpy.ops.render.render(write_still=True)

        if not render_path.exists():
            candidates = sorted(Path(temp_dir).glob("rgb*.png"))
            if not candidates:
                raise RuntimeError(f"Blender render did not write an RGB image to {render_path}")
            render_path = candidates[0]

        rendered_image = bpy.data.images.load(str(render_path))
        try:
            actual_width, actual_height = (int(value) for value in rendered_image.size)
            if (actual_width, actual_height) != (width, height):
                raise RuntimeError(
                    "Blender render dimensions do not match requested output: "
                    f"got {(actual_width, actual_height)} expected {(width, height)}"
                )

            pixels = np.asarray(rendered_image.pixels[:], dtype=np.float32)
            channel_count = pixels.size // (actual_width * actual_height)
            if channel_count < 3:
                raise RuntimeError("Blender render result must contain at least three color channels")

            rgba = pixels.reshape((actual_height, actual_width, channel_count))
            rgb = np.flipud(rgba[:, :, :3])
            return np.clip(rgb * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
        finally:
            bpy.data.images.remove(rendered_image)


def _clear_blender_scene(bpy) -> None:
    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    for datablock_collection in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        for datablock in list(datablock_collection):
            if datablock.users == 0:
                datablock_collection.remove(datablock)


def _create_body(bpy, spec: FacadeSceneSpec):
    height_m = spec.floor_count * spec.story_height_m
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, spec.depth_m / 2.0, height_m / 2.0))
    body = bpy.context.object
    body.name = "facade_body"
    body.dimensions = (float(spec.width_m), float(spec.depth_m), float(height_m))
    body.data.materials.append(
        _create_material(
            bpy,
            "facade_material",
            _material_color_rgba(spec.material_variant),
            roughness=0.92,
        )
    )
    body["semantic_label"] = "facade"
    return body


def _create_windows(bpy, spec: FacadeSceneSpec) -> list:
    material = _create_material(bpy, "window_material", (0.06, 0.11, 0.16, 1.0), roughness=0.18)
    objects = []
    for window in spec.windows:
        corners = np.asarray(window.corners_world, dtype=np.float64)
        x0, y0, z0 = np.min(corners, axis=0)
        x1, y1, z1 = np.max(corners, axis=0)
        thickness = 0.08
        location = (
            float((x0 + x1) / 2.0),
            float((y0 + y1) / 2.0),
            float((z0 + z1) / 2.0),
        )
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
        window_object = bpy.context.object
        window_object.name = f"window_{window.instance_id:03d}"
        window_object.dimensions = (
            max(float(x1 - x0), 0.01),
            thickness,
            max(float(z1 - z0), 0.01),
        )
        window_object.data.materials.append(material)
        window_object["semantic_label"] = "window"
        window_object["instance_id"] = int(window.instance_id)
        window_object["floor_index"] = int(window.floor_index)
        window_object["column_index"] = int(window.column_index)
        objects.append(window_object)
    return objects


def _create_ground(bpy, spec: FacadeSceneSpec):
    extent = max(spec.width_m * 3.0, spec.depth_m * 4.0, spec.camera_view.distance_m * 1.2)
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, spec.depth_m / 2.0, -0.02))
    ground = bpy.context.object
    ground.name = "ground_plane"
    ground.dimensions = (float(extent), float(extent), 0.0)
    ground.data.materials.append(_create_material(bpy, "ground_material", (0.34, 0.35, 0.32, 1.0), roughness=0.96))
    ground["semantic_label"] = "ground"
    return ground


def _create_lighting(bpy, spec: FacadeSceneSpec):
    height_m = spec.floor_count * spec.story_height_m
    if spec.lighting_variant in {"overcast", "soft_front"}:
        bpy.ops.object.light_add(type="AREA", location=(0.0, -spec.depth_m, height_m * 1.4))
        light = bpy.context.object
        light.name = f"{spec.lighting_variant}_area_light"
        light.data.energy = 500.0 if spec.lighting_variant == "overcast" else 650.0
        light.data.size = max(spec.width_m, height_m) * 1.5
    else:
        bpy.ops.object.light_add(type="SUN", location=(0.0, -spec.depth_m, height_m * 1.8))
        light = bpy.context.object
        light.name = f"{spec.lighting_variant}_sun_light"
        light.data.energy = 2.6
        if spec.lighting_variant == "morning_side":
            light.rotation_euler = (math.radians(52.0), 0.0, math.radians(-35.0))
        else:
            light.rotation_euler = (math.radians(58.0), 0.0, math.radians(38.0))
    return light


def _create_camera(bpy, spec: FacadeSceneSpec):
    from mathutils import Matrix

    world_to_camera = np.asarray(spec.camera_world_to_camera, dtype=np.float64)
    camera_to_world = np.linalg.inv(world_to_camera)
    if not np.isfinite(camera_to_world).all():
        raise ValueError("camera transform must be finite")

    # Spec projection uses camera +Z as forward. Blender cameras look along
    # local -Z, so the camera basis is converted by negating the forward column.
    right_world = camera_to_world[:3, 0]
    up_world = camera_to_world[:3, 1]
    forward_world = camera_to_world[:3, 2]
    location_world = camera_to_world[:3, 3]

    blender_camera_to_world = np.eye(4, dtype=np.float64)
    blender_camera_to_world[:3, 0] = right_world
    blender_camera_to_world[:3, 1] = up_world
    blender_camera_to_world[:3, 2] = -forward_world
    blender_camera_to_world[:3, 3] = location_world

    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.name = "facade_camera"
    camera.matrix_world = Matrix(blender_camera_to_world.tolist())
    camera.data.type = "PERSP"
    camera.data.lens = float(spec.camera_view.focal_length_mm)
    camera.data.sensor_width = float(SENSOR_WIDTH_MM)
    camera.data.sensor_fit = "HORIZONTAL"
    camera.data.clip_start = 0.05
    camera.data.clip_end = max(200.0, spec.camera_view.distance_m + spec.depth_m + 50.0)
    bpy.context.scene.camera = camera
    return camera


def _create_material(bpy, name: str, rgba: tuple[float, float, float, float], *, roughness: float):
    material = bpy.data.materials.new(name)
    material.diffuse_color = rgba
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        texture = nodes.new("ShaderNodeTexNoise")
        texture.inputs["Scale"].default_value = 3.5 if roughness > 0.7 else 8.0
        texture.inputs["Detail"].default_value = 3.0
        ramp = nodes.new("ShaderNodeValToRGB")
        dark = tuple(max(0.0, min(1.0, channel * 0.78)) for channel in rgba[:3]) + (rgba[3],)
        light = tuple(max(0.0, min(1.0, channel * 1.08)) for channel in rgba[:3]) + (rgba[3],)
        ramp.color_ramp.elements[0].color = dark
        ramp.color_ramp.elements[1].color = light
        bump = nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = 0.18 if roughness > 0.7 else 0.05
        bump.inputs["Distance"].default_value = 0.08
        links.new(texture.outputs["Fac"], ramp.inputs["Fac"])
        links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
        links.new(texture.outputs["Fac"], bump.inputs["Height"])
        links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
        if "Base Color" in bsdf.inputs:
            bsdf.inputs["Base Color"].default_value = rgba
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = float(roughness)
        if "Metallic" in bsdf.inputs:
            bsdf.inputs["Metallic"].default_value = 0.0
    return material


def _material_color_rgba(material_variant: str) -> tuple[float, float, float, float]:
    colors = {
        "concrete_light": (0.68, 0.66, 0.60, 1.0),
        "brick_warm": (0.60, 0.36, 0.28, 1.0),
        "stucco_cool": (0.58, 0.65, 0.67, 1.0),
        "painted_panel": (0.50, 0.57, 0.53, 1.0),
    }
    return colors.get(material_variant, colors["concrete_light"])


def _sky_color(lighting_variant: str) -> tuple[float, float, float]:
    colors = {
        "overcast": (0.62, 0.66, 0.69),
        "morning_side": (0.72, 0.69, 0.62),
        "late_afternoon": (0.72, 0.59, 0.50),
        "soft_front": (0.64, 0.70, 0.74),
    }
    return colors.get(lighting_variant, colors["overcast"])
