from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from facade_synth.seed_v31.blender_scene import (
    _clear_blender_scene,
    _create_camera,
    _create_ground,
    _create_lighting,
    _create_material,
    _material_color_rgba,
    _sky_color,
)
from facade_synth.seed_v31.structure_scene import BalconyModule3D, FacadeStructureSpec, StructuredWindow3D


@dataclass(frozen=True)
class WindowPartBox:
    name: str
    location: tuple[float, float, float]
    dimensions: tuple[float, float, float]
    material_key: str
    semantic_label: str


COMPONENT_CLASSES = (
    "facade_wall",
    "window_glass",
    "window_frame",
    "door",
    "balcony",
    "floor_band",
    "podium_storefront",
    "roof_parapet",
    "background",
)
_FOREGROUND_OCCLUDER_ID = len(COMPONENT_CLASSES)
_OBJECT_COMPONENTS = {
    "facade": "facade_wall",
    "window": "window_glass",
    "window_detail": "window_frame",
    "door": "door",
    "balcony": "balcony",
    "floor_band": "floor_band",
    "podium_storefront": "podium_storefront",
    "roof_parapet": "roof_parapet",
    "foreground_occluder": "foreground_occluder",
}


@dataclass(frozen=True)
class SceneTruthRender:
    component_mask: np.ndarray
    occlusion_ratio: float
    occluder_variant: str
    visibility: dict[str, float]
    lighting_recipe: dict[str, float]


def window_part_names(instance_id: int) -> dict[str, str]:
    base_name = f"v3_window_{instance_id:03d}"
    return {
        "opening": f"{base_name}_opening",
        "glass": f"{base_name}_glass",
        "frame_left": f"{base_name}_frame_left",
        "frame_right": f"{base_name}_frame_right",
        "frame_top": f"{base_name}_frame_top",
        "frame_bottom": f"{base_name}_frame_bottom",
        "sill": f"{base_name}_sill",
    }


def build_blender_structure_scene(
    spec: FacadeStructureSpec,
    *,
    width: int,
    height: int,
    render_samples: int,
    occlusion_band: str = "clear",
    lighting_intensity_scale: float = 1.0,
    asset_paths: Sequence[str] = (),
) -> None:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if render_samples <= 0:
        raise ValueError("render_samples must be positive")
    if not 0.1 <= lighting_intensity_scale <= 4.0:
        raise ValueError("lighting_intensity_scale must be within 0.1–4.0")

    import bpy

    label_spec = spec.label_scene_spec
    _clear_blender_scene(bpy)
    _configure_scene(bpy, spec, width=width, height=height, render_samples=render_samples)

    materials = _create_v3_materials(bpy, spec, asset_paths=asset_paths)
    _create_body(bpy, spec, materials["facade"])
    _create_floor_bands(bpy, spec, materials["floor_band"])
    _create_podium_storefront_and_door(bpy, spec, materials)
    _create_roof_parapet(bpy, spec, materials["roof_parapet"])
    _create_structured_windows(bpy, spec.windows, materials)
    _create_balconies(bpy, spec.balconies, materials["balcony"])
    _create_environment_strips(bpy, spec, materials["sidewalk"], materials["road"])
    _create_ground(bpy, label_spec)
    _create_lighting(bpy, label_spec, intensity_scale=lighting_intensity_scale)
    bpy.context.scene["facade_lighting_intensity_scale"] = float(lighting_intensity_scale)
    camera = _create_camera(bpy, label_spec)
    _create_controlled_foreground_occluder(bpy, spec, camera, materials["occluder"], occlusion_band)
    bpy.context.view_layer.update()


def _configure_scene(bpy, spec: FacadeStructureSpec, *, width: int, height: int, render_samples: int) -> None:
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
    scene.world.color = tuple(float(value) for value in _sky_color(spec.label_scene_spec.lighting_variant))


def _create_v3_materials(bpy, spec: FacadeStructureSpec, *, asset_paths: Sequence[str]) -> dict[str, object]:
    facade_rgba = _material_color_rgba(spec.label_scene_spec.material_variant)
    materials = {
        "facade": _create_material(bpy, "v3_facade_material", facade_rgba, roughness=0.92),
        "floor_band": _create_material(bpy, "v3_floor_band_material", _darken(facade_rgba, 0.72), roughness=0.9),
        "opening": _create_material(bpy, "v3_window_opening_material", (0.035, 0.036, 0.038, 1.0), roughness=0.84),
        "glass": _create_material(bpy, "v3_window_glass_material", (0.045, 0.11, 0.16, 1.0), roughness=0.16),
        "frame": _create_material(bpy, "v3_window_frame_material", (0.14, 0.15, 0.14, 1.0), roughness=0.58),
        "sill": _create_material(bpy, "v3_window_sill_material", (0.46, 0.47, 0.43, 1.0), roughness=0.72),
        "balcony": _create_material(bpy, "v3_balcony_material", (0.52, 0.53, 0.50, 1.0), roughness=0.82),
        "sidewalk": _create_material(bpy, "v3_sidewalk_material", (0.43, 0.43, 0.40, 1.0), roughness=0.94),
        "road": _create_material(bpy, "v3_road_material", (0.15, 0.16, 0.16, 1.0), roughness=0.9),
        "podium": _create_material(bpy, "v3_podium_material", (0.21, 0.26, 0.28, 1.0), roughness=0.46),
        "door": _create_material(bpy, "v3_door_material", (0.12, 0.085, 0.06, 1.0), roughness=0.58),
        "roof_parapet": _create_material(bpy, "v3_roof_parapet_material", _darken(facade_rgba, 0.64), roughness=0.85),
        "occluder": _create_material(bpy, "v3_foreground_occluder_material", (0.10, 0.13, 0.11, 1.0), roughness=0.78),
    }
    _apply_optional_visual_assets(bpy, materials, asset_paths)
    return materials


def _create_body(bpy, spec: FacadeStructureSpec, material):
    label_spec = spec.label_scene_spec
    height_m = label_spec.floor_count * label_spec.story_height_m
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, label_spec.depth_m / 2.0, height_m / 2.0))
    body = bpy.context.object
    body.name = "v3_facade_body"
    body.dimensions = (float(label_spec.width_m), float(label_spec.depth_m), float(height_m))
    body.data.materials.append(material)
    body["semantic_label"] = "facade"
    body["structure_variant"] = spec.structure_variant
    return body


def _create_floor_bands(bpy, spec: FacadeStructureSpec, material) -> list:
    bands = []
    for boundary_index, segment in enumerate(spec.label_scene_spec.floorline_segments_world, start=1):
        points = np.asarray(segment, dtype=np.float64)
        x0, y0, z0 = np.min(points, axis=0)
        x1, _y1, _z1 = np.max(points, axis=0)
        band = _add_box(
            bpy,
            f"v3_floor_band_{boundary_index:03d}",
            location=(float((x0 + x1) / 2.0), float(y0 - 0.03), float(z0)),
            dimensions=(max(float(x1 - x0), 0.01), 0.08, 0.055),
            material=material,
            semantic_label="floor_band",
        )
        band["floor_boundary_index"] = int(boundary_index)
        bands.append(band)
    return bands


def _create_podium_storefront_and_door(bpy, spec: FacadeStructureSpec, materials: dict[str, object]) -> list:
    """Add explicit semantic objects instead of inferring podium or doors in 2D."""

    label_spec = spec.label_scene_spec
    if spec.podium_floor_count <= 0:
        return []
    podium_height = label_spec.story_height_m * spec.podium_floor_count
    storefront = _add_box(
        bpy,
        "v3_podium_storefront",
        location=(0.0, -0.045, podium_height / 2.0),
        dimensions=(float(label_spec.width_m) * 0.98, 0.065, podium_height * 0.92),
        material=materials["podium"],
        semantic_label="podium_storefront",
    )
    door_width = min(max(label_spec.width_m * 0.12, 1.2), 3.2)
    door = _add_box(
        bpy,
        "v3_podium_door",
        location=(-float(label_spec.width_m) * 0.22, -0.095, label_spec.story_height_m * 0.44),
        dimensions=(door_width, 0.08, label_spec.story_height_m * 0.72),
        material=materials["door"],
        semantic_label="door",
    )
    return [storefront, door]


def _create_roof_parapet(bpy, spec: FacadeStructureSpec, material):
    label_spec = spec.label_scene_spec
    height_m = label_spec.floor_count * label_spec.story_height_m
    return _add_box(
        bpy,
        "v3_roof_parapet",
        location=(0.0, -0.035, height_m + 0.18),
        dimensions=(float(label_spec.width_m) * 1.02, 0.12, 0.36),
        material=material,
        semantic_label="roof_parapet",
    )


def _create_controlled_foreground_occluder(bpy, spec: FacadeStructureSpec, camera, material, occlusion_band: str):
    """Place a real foreground billboard in the camera frustum for 0–30% occlusion."""

    if occlusion_band == "clear":
        return None
    fractions = {"light_0_15": 0.12, "moderate_15_30": 0.24}
    try:
        fraction = fractions[occlusion_band]
    except KeyError as exc:
        raise ValueError(f"unsupported occlusion band: {occlusion_band}") from exc

    from mathutils import Vector

    label_spec = spec.label_scene_spec
    target_distance = max(float(label_spec.camera_view.distance_m) * 0.78, 2.0)
    forward = camera.matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))
    right = camera.matrix_world.to_quaternion() @ Vector((1.0, 0.0, 0.0))
    # The small fixed offset keeps its stripe inside the facade rather than at an edge.
    center = camera.location + forward * target_distance + right * (float(label_spec.width_m) * 0.11)
    height_m = label_spec.floor_count * label_spec.story_height_m
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=center)
    occluder = bpy.context.object
    occluder.name = f"v3_{occlusion_band}_foreground_occluder"
    occluder.rotation_euler = camera.rotation_euler
    occluder.dimensions = (
        max(float(label_spec.width_m) * fraction * 0.82, 0.2),
        max(height_m * 1.35, 0.2),
        0.08,
    )
    occluder.data.materials.append(material)
    occluder["semantic_label"] = "foreground_occluder"
    occluder["occlusion_band"] = occlusion_band
    return occluder


def _create_structured_windows(bpy, windows: tuple[StructuredWindow3D, ...], materials: dict[str, object]) -> list:
    objects = []
    for window in windows:
        objects.extend(_create_window_parts(bpy, window, materials))
    return objects


def _create_window_parts(bpy, window: StructuredWindow3D, materials: dict[str, object]) -> list:
    return [
        _window_box(
            bpy,
            box.name,
            window,
            location=box.location,
            dimensions=box.dimensions,
            material=materials[box.material_key],
            semantic_label=box.semantic_label,
        )
        for box in _window_part_boxes(window)
    ]


def _window_part_boxes(window: StructuredWindow3D) -> list[WindowPartBox]:
    names = window_part_names(window.instance_id)
    x0, y0, z0, x1, _y1, z1 = _bounds(window.opening_corners_world)
    gx0, gy0, gz0, gx1, _gy1, gz1 = _bounds(window.glass_window.corners_world)

    opening_width = max(x1 - x0, 0.01)
    opening_height = max(z1 - z0, 0.01)
    glass_width = max(gx1 - gx0, 0.01)
    glass_height = max(gz1 - gz0, 0.01)
    opening_depth = max(float(window.opening_depth_m), 0.04)
    frame_depth = max(min(opening_depth * 0.75, 0.12), 0.045)
    frame_thickness = max(
        min(float(window.frame_thickness_m), opening_width * 0.22, opening_height * 0.22),
        0.025,
    )
    sill_depth = max(float(window.sill_depth_m), 0.05)
    front_y = float(y0)
    back_panel_depth = min(max(opening_depth * 0.18, 0.02), 0.05)
    recess_start_y = front_y + 0.004
    recess_depth = max(opening_depth - 0.004, 0.02)
    frame_y = front_y - frame_depth / 2.0 - 0.015
    glass_y = float(gy0) - 0.025
    recess_y = recess_start_y + recess_depth / 2.0

    boxes = [
        WindowPartBox(
            names["opening"],
            location=((x0 + x1) / 2.0, front_y + opening_depth - back_panel_depth / 2.0, (z0 + z1) / 2.0),
            dimensions=(opening_width, back_panel_depth, opening_height),
            material_key="opening",
            semantic_label="window_detail",
        ),
        WindowPartBox(
            f"v3_window_{window.instance_id:03d}_recess_left",
            location=((x0 + gx0) / 2.0, recess_y, (z0 + z1) / 2.0),
            dimensions=(max(gx0 - x0, frame_thickness), recess_depth, opening_height),
            material_key="opening",
            semantic_label="window_detail",
        ),
        WindowPartBox(
            f"v3_window_{window.instance_id:03d}_recess_right",
            location=((gx1 + x1) / 2.0, recess_y, (z0 + z1) / 2.0),
            dimensions=(max(x1 - gx1, frame_thickness), recess_depth, opening_height),
            material_key="opening",
            semantic_label="window_detail",
        ),
        WindowPartBox(
            f"v3_window_{window.instance_id:03d}_recess_top",
            location=((x0 + x1) / 2.0, recess_y, (gz1 + z1) / 2.0),
            dimensions=(opening_width, recess_depth, max(z1 - gz1, frame_thickness)),
            material_key="opening",
            semantic_label="window_detail",
        ),
        WindowPartBox(
            f"v3_window_{window.instance_id:03d}_recess_bottom",
            location=((x0 + x1) / 2.0, recess_y, (z0 + gz0) / 2.0),
            dimensions=(opening_width, recess_depth, max(gz0 - z0, frame_thickness)),
            material_key="opening",
            semantic_label="window_detail",
        ),
        WindowPartBox(
            names["glass"],
            location=((gx0 + gx1) / 2.0, glass_y, (gz0 + gz1) / 2.0),
            dimensions=(glass_width, 0.035, glass_height),
            material_key="glass",
            semantic_label="window",
        ),
        WindowPartBox(
            names["frame_left"],
            location=((x0 + gx0) / 2.0, frame_y, (z0 + z1) / 2.0),
            dimensions=(max(gx0 - x0, frame_thickness), frame_depth, opening_height),
            material_key="frame",
            semantic_label="window_detail",
        ),
        WindowPartBox(
            names["frame_right"],
            location=((gx1 + x1) / 2.0, frame_y, (z0 + z1) / 2.0),
            dimensions=(max(x1 - gx1, frame_thickness), frame_depth, opening_height),
            material_key="frame",
            semantic_label="window_detail",
        ),
        WindowPartBox(
            names["frame_top"],
            location=((x0 + x1) / 2.0, frame_y, (gz1 + z1) / 2.0),
            dimensions=(opening_width, frame_depth, max(z1 - gz1, frame_thickness)),
            material_key="frame",
            semantic_label="window_detail",
        ),
        WindowPartBox(
            names["frame_bottom"],
            location=((x0 + x1) / 2.0, frame_y, (z0 + gz0) / 2.0),
            dimensions=(opening_width, frame_depth, max(gz0 - z0, frame_thickness)),
            material_key="frame",
            semantic_label="window_detail",
        ),
        WindowPartBox(
            names["sill"],
            location=((x0 + x1) / 2.0, front_y - sill_depth / 2.0 - 0.035, z0 - frame_thickness / 2.0),
            dimensions=(opening_width + frame_thickness * 1.4, sill_depth, frame_thickness),
            material_key="sill",
            semantic_label="window_detail",
        ),
    ]

    for mullion_index in range(max(int(window.mullion_count), 0)):
        x = gx0 + glass_width * float(mullion_index + 1) / float(window.mullion_count + 1)
        mullion_width = min(frame_thickness * 0.55, max(glass_width / 8.0, 0.018))
        boxes.append(
            WindowPartBox(
                f"v3_window_{window.instance_id:03d}_mullion_{mullion_index + 1:03d}",
                location=(x, frame_y, (gz0 + gz1) / 2.0),
                dimensions=(mullion_width, frame_depth, glass_height),
                material_key="frame",
                semantic_label="window_detail",
            )
        )

    return boxes


def _window_box(
    bpy,
    name: str,
    window: StructuredWindow3D,
    *,
    location: tuple[float, float, float],
    dimensions: tuple[float, float, float],
    material,
    semantic_label: str,
):
    obj = _add_box(
        bpy,
        name,
        location=location,
        dimensions=dimensions,
        material=material,
        semantic_label=semantic_label,
    )
    obj["instance_id"] = int(window.instance_id)
    obj["floor_index"] = int(window.floor_index)
    obj["column_index"] = int(window.column_index)
    return obj


def _create_balconies(bpy, balconies: tuple[BalconyModule3D, ...], material) -> list:
    objects = []
    for index, balcony in enumerate(balconies, start=1):
        obj = _add_box(
            bpy,
            f"v3_balcony_{index:03d}",
            location=tuple(float(value) for value in balcony.center_world),
            dimensions=(float(balcony.width_m), float(balcony.depth_m), float(balcony.height_m)),
            material=material,
            semantic_label="balcony",
        )
        obj["floor_index"] = int(balcony.floor_index)
        obj["column_index"] = int(balcony.column_index)
        objects.append(obj)
    return objects


def _create_environment_strips(bpy, spec: FacadeStructureSpec, sidewalk_material, road_material) -> list:
    label_spec = spec.label_scene_spec
    facade_width = float(label_spec.width_m)
    sidewalk_depth = 2.2
    road_depth = 5.0
    sidewalk = _add_box(
        bpy,
        "v3_sidewalk_strip",
        location=(0.0, -sidewalk_depth / 2.0 - 0.1, 0.0),
        dimensions=(facade_width * 1.8, sidewalk_depth, 0.04),
        material=sidewalk_material,
        semantic_label="environment",
    )
    road = _add_box(
        bpy,
        "v3_road_strip",
        location=(0.0, -sidewalk_depth - road_depth / 2.0 - 0.35, -0.005),
        dimensions=(facade_width * 2.2, road_depth, 0.035),
        material=road_material,
        semantic_label="environment",
    )
    return [sidewalk, road]


def _add_box(
    bpy,
    name: str,
    *,
    location: tuple[float, float, float],
    dimensions: tuple[float, float, float],
    material,
    semantic_label: str,
):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=tuple(float(value) for value in location))
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = tuple(max(float(value), 0.01) for value in dimensions)
    obj.data.materials.append(material)
    obj["semantic_label"] = semantic_label
    return obj


def _apply_optional_visual_assets(bpy, materials: dict[str, object], asset_paths: Sequence[str]) -> None:
    """Make confirmed internal PBR/HDRI inputs affect RGB while never affecting labels."""

    texture_path: Path | None = None
    hdri_path: Path | None = None
    for value in asset_paths:
        path = Path(value)
        suffix = path.suffix.lower()
        if suffix in {".hdr", ".exr"} and hdri_path is None:
            hdri_path = path
        elif suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"} and texture_path is None:
            texture_path = path
    if texture_path is not None:
        image = bpy.data.images.load(str(texture_path), check_existing=True)
        facade = materials["facade"]
        nodes = facade.node_tree.nodes
        links = facade.node_tree.links
        bsdf = nodes.get("Principled BSDF")
        if bsdf is not None:
            image_node = nodes.new("ShaderNodeTexImage")
            image_node.image = image
            links.new(image_node.outputs["Color"], bsdf.inputs["Base Color"])
            facade["visual_asset_path"] = str(texture_path)
    if hdri_path is not None:
        world = bpy.context.scene.world
        if world is None:
            world = bpy.data.worlds.new("World")
            bpy.context.scene.world = world
        world.use_nodes = True
        nodes = world.node_tree.nodes
        links = world.node_tree.links
        background = nodes.get("Background")
        output = nodes.get("World Output")
        if background is not None and output is not None:
            environment = nodes.new("ShaderNodeTexEnvironment")
            environment.image = bpy.data.images.load(str(hdri_path), check_existing=True)
            links.new(environment.outputs["Color"], background.inputs["Color"])
            links.new(background.outputs["Background"], output.inputs["Surface"])
            world["visual_asset_path"] = str(hdri_path)


def render_visible_component_scene_truth(*, width: int, height: int) -> SceneTruthRender:
    """Render visible class IDs from Blender object/material passes, including occlusion truth."""

    import bpy

    occluders = [
        object_
        for object_ in bpy.context.scene.objects
        if object_.get("semantic_label") == "foreground_occluder"
    ]
    for object_ in occluders:
        object_.hide_render = True
    unoccluded = _render_object_index_ids(bpy, width=width, height=height)
    for object_ in occluders:
        object_.hide_render = False
    occluded = _render_object_index_ids(bpy, width=width, height=height)

    background_id = COMPONENT_CLASSES.index("background")
    facade_present = unoccluded != background_id
    occluder_pixels = occluded == _FOREGROUND_OCCLUDER_ID
    occluded_facade = facade_present & occluder_pixels
    denominator = int(np.count_nonzero(facade_present))
    occlusion_ratio = float(np.count_nonzero(occluded_facade) / denominator) if denominator else 0.0
    component_mask = np.where(occluder_pixels, background_id, occluded).astype(np.uint8)
    component_mask[component_mask > background_id] = background_id

    visibility: dict[str, float] = {}
    for index, name in enumerate(COMPONENT_CLASSES[:-1]):
        total = int(np.count_nonzero(unoccluded == index))
        visible = int(np.count_nonzero(component_mask == index))
        visibility[name] = min(1.0, float(visible / total)) if total else 0.0
    visibility["facade_components"] = min(
        1.0,
        float(np.count_nonzero(component_mask != background_id) / denominator),
    ) if denominator else 0.0
    occluder_variant = str(occluders[0].get("occlusion_band")) if occluders else "clear"
    return SceneTruthRender(
        component_mask=component_mask,
        occlusion_ratio=occlusion_ratio,
        occluder_variant=occluder_variant,
        visibility=visibility,
        lighting_recipe=_actual_lighting_recipe(bpy),
    )


def _render_object_index_ids(bpy, *, width: int, height: int) -> np.ndarray:
    """Use Blender's visible object-index pass, rather than inferred 2D geometry."""

    import tempfile

    scene = bpy.context.scene
    original_cycles_samples = int(scene.cycles.samples) if hasattr(scene, "cycles") else None
    view_layer = scene.view_layers[0]
    original_object_index_pass = view_layer.use_pass_object_index
    original_pass_indices: list[tuple[object, int]] = []
    try:
        if hasattr(scene, "cycles"):
            scene.cycles.samples = 1
        view_layer.use_pass_object_index = True
        for object_ in scene.objects:
            original_pass_indices.append((object_, int(object_.pass_index)))
            semantic = _OBJECT_COMPONENTS.get(str(object_.get("semantic_label")), "background")
            object_.pass_index = (
                _FOREGROUND_OCCLUDER_ID + 1
                if semantic == "foreground_occluder"
                else COMPONENT_CLASSES.index(semantic) + 1
            ) if semantic != "background" else 0
        bpy.context.view_layer.update()
        with tempfile.TemporaryDirectory(prefix="facade_semantic_pass_") as temp_dir:
            scene.use_nodes = True
            nodes = scene.node_tree.nodes
            links = scene.node_tree.links
            nodes.clear()
            render_layers = nodes.new("CompositorNodeRLayers")
            output = nodes.new("CompositorNodeOutputFile")
            output.base_path = temp_dir
            output.format.file_format = "OPEN_EXR"
            output.format.color_mode = "RGB"
            output.format.color_depth = "32"
            output.file_slots[0].path = "object_index"
            links.new(render_layers.outputs["IndexOB"], output.inputs[0])
            bpy.ops.render.render(write_still=False)
            candidates = sorted(Path(temp_dir).glob("object_index*.exr"))
            if not candidates:
                raise RuntimeError("Blender did not write an object-index pass")
            image = bpy.data.images.load(str(candidates[0]))
            try:
                pixels = np.asarray(image.pixels[:], dtype=np.float32)
                channels = pixels.size // (width * height)
                if channels < 3:
                    raise RuntimeError("object-index pass has fewer than three channels")
                values = pixels.reshape((height, width, channels))
                ids = np.rint(np.flipud(values[..., 0])).astype(np.int16) - 1
                background_id = COMPONENT_CLASSES.index("background")
                ids[(ids < 0) | (ids > _FOREGROUND_OCCLUDER_ID)] = background_id
                return ids.astype(np.uint8)
            finally:
                bpy.data.images.remove(image)
    finally:
        for object_, index in original_pass_indices:
            object_.pass_index = index
        bpy.context.view_layer.update()
        view_layer.use_pass_object_index = original_object_index_pass
        if original_cycles_samples is not None:
            scene.cycles.samples = original_cycles_samples


def _actual_lighting_recipe(bpy) -> dict[str, float]:
    light = next((object_ for object_ in bpy.context.scene.objects if object_.type == "LIGHT"), None)
    energy = float(light.data.energy) if light is not None else 0.0
    azimuth = float(light.rotation_euler.z * 180.0 / np.pi) if light is not None else 0.0
    elevation = float(light.rotation_euler.x * 180.0 / np.pi) if light is not None else 0.0
    world = bpy.context.scene.world
    world_strength = 1.0
    if world is not None and world.use_nodes and world.node_tree.nodes.get("Background") is not None:
        world_strength = float(world.node_tree.nodes["Background"].inputs["Strength"].default_value)
    return {
        "sun_elevation_deg": round(elevation, 4),
        "relative_azimuth_deg": round(azimuth, 4),
        "energy": round(energy, 4),
        "world_strength": round(world_strength, 4),
        "exposure_ev": float(bpy.context.scene.view_settings.exposure),
        "colour_temperature_k": 6500.0,
        "intensity_scale": float(bpy.context.scene.get("facade_lighting_intensity_scale", 1.0)),
    }


def _bounds(corners: object) -> tuple[float, float, float, float, float, float]:
    points = np.asarray(corners, dtype=np.float64)
    x0, y0, z0 = np.min(points, axis=0)
    x1, y1, z1 = np.max(points, axis=0)
    return (float(x0), float(y0), float(z0), float(x1), float(y1), float(z1))


def _darken(rgba: tuple[float, float, float, float], factor: float) -> tuple[float, float, float, float]:
    return (rgba[0] * factor, rgba[1] * factor, rgba[2] * factor, rgba[3])
