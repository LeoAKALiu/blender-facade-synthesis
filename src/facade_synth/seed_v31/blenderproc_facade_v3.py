"""Locally owned V3.1 renderer, invoked only through BlenderProc bootstrap."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from facade_synth.seed_v31.blender_scene import render_current_scene_rgb_array  # noqa: E402
from facade_synth.seed_v31.blender_scene_v3 import build_blender_structure_scene  # noqa: E402
from facade_synth.seed_v31.blenderproc_facade_mvp import (  # noqa: E402
    GeneratedSample,
    prepare_output_dir,
    write_manifest,
    write_sample,
)
from facade_synth.seed_v31.render_outputs_v3 import build_projected_structure_sample  # noqa: E402
from facade_synth.seed_v31.structure_scene import (  # noqa: E402
    STRUCTURE_VARIANTS,
    FacadeStructureSpec,
    StructureVariant,
    sample_structure_scene_spec,
)


@dataclass(frozen=True)
class _RenderResult:
    rgb: np.ndarray
    used_projection_fallback: bool


@dataclass(frozen=True)
class _DatasetGenerationResult:
    samples: list[GeneratedSample]
    rendered_with_blender_count: int
    projection_fallback_count: int


def render_rgb_for_structure_spec(
    spec: FacadeStructureSpec,
    *,
    width: int,
    height: int,
    render_samples: int,
    force_fallback: bool = False,
) -> np.ndarray:
    return _render_rgb_for_structure_spec_with_evidence(
        spec,
        width=width,
        height=height,
        render_samples=render_samples,
        force_fallback=force_fallback,
    ).rgb


def _render_rgb_for_structure_spec_with_evidence(
    spec: FacadeStructureSpec,
    *,
    width: int,
    height: int,
    render_samples: int,
    force_fallback: bool = False,
) -> _RenderResult:
    if force_fallback:
        raise ValueError("projection fallback is disabled for trainable generation")
    try:
        build_blender_structure_scene(spec, width=width, height=height, render_samples=render_samples)
        return _RenderResult(
            rgb=render_current_scene_rgb_array(width=width, height=height),
            used_projection_fallback=False,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "bpy":
            raise RuntimeError("BlenderProc Blender runtime is required for trainable generation") from exc
        raise


def generate_dataset_v3(
    *,
    output_dir: Path | str,
    count: int,
    width: int,
    height: int,
    seed: int,
    structure_variants: list[str] | None = None,
    render_samples: int = 64,
    force_fallback: bool = False,
    view_band: str | None = None,
    lighting_variant: str | None = None,
    material_variant: str | None = None,
) -> list[GeneratedSample]:
    return _generate_dataset_v3_with_evidence(
        output_dir=output_dir,
        count=count,
        width=width,
        height=height,
        seed=seed,
        structure_variants=structure_variants,
        render_samples=render_samples,
        force_fallback=force_fallback,
        view_band=view_band,
        lighting_variant=lighting_variant,
        material_variant=material_variant,
    ).samples


def _generate_dataset_v3_with_evidence(
    *,
    output_dir: Path | str,
    count: int,
    width: int,
    height: int,
    seed: int,
    structure_variants: list[str] | None = None,
    render_samples: int = 64,
    force_fallback: bool = False,
    view_band: str | None = None,
    lighting_variant: str | None = None,
    material_variant: str | None = None,
) -> _DatasetGenerationResult:
    if count < 0:
        raise ValueError("count must be non-negative")
    if width < 32 or height < 32:
        raise ValueError("width and height must be at least 32 pixels")
    if render_samples < 1:
        raise ValueError("render_samples must be positive")

    selected_variants = _normalize_structure_variants(structure_variants)
    root = prepare_output_dir(output_dir)
    samples: list[GeneratedSample] = []
    projection_fallback_count = 0
    for index in range(count):
        sample_id = f"facade_{index:06d}"
        structure_variant = (
            selected_variants[index % len(selected_variants)]
            if selected_variants is not None
            else None
        )
        spec = sample_structure_scene_spec(
            sample_id=sample_id,
            seed=seed + index,
            structure_variant=structure_variant,
            width_px=width,
            height_px=height,
            view_band=view_band,
            lighting_variant=lighting_variant,
            material_variant=material_variant,
        )
        render_result = _render_rgb_for_structure_spec_with_evidence(
            spec,
            width=width,
            height=height,
            render_samples=render_samples,
            force_fallback=force_fallback,
        )
        if render_result.used_projection_fallback:
            projection_fallback_count += 1
        sample = build_projected_structure_sample(spec, width=width, height=height, rgb=render_result.rgb)
        write_sample(root, sample)
        samples.append(sample)

    write_manifest(root, samples)
    return _DatasetGenerationResult(
        samples=samples,
        rendered_with_blender_count=len(samples) - projection_fallback_count,
        projection_fallback_count=projection_fallback_count,
    )


def _parse_structure_variants(value: str | None) -> list[str] | None:
    if value is None or not value.strip():
        return None
    variants = [item.strip() for item in value.split(",") if item.strip()]
    normalized = _normalize_structure_variants(variants)
    return list(normalized) if normalized is not None else None


def _script_args(argv: list[str] | None) -> list[str] | None:
    if argv is not None:
        return argv
    if "--" not in sys.argv:
        return None

    separator_index = sys.argv.index("--")
    forwarded = sys.argv[separator_index + 1 :]
    if len(forwarded) >= 2 and not forwarded[0].startswith("-") and not forwarded[1].startswith("-"):
        return forwarded[2:]
    return forwarded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the synthetic facade V3 structure dataset.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--structure-variants")
    parser.add_argument("--render-samples", type=int, default=64)
    parser.add_argument("--force-fallback", action="store_true")
    parser.add_argument("--view-band", choices=("frontal", "light_medium_oblique", "strong_oblique"))
    parser.add_argument("--lighting-variant", choices=("overcast", "morning_side", "late_afternoon", "soft_front"))
    parser.add_argument("--material-variant", choices=("concrete_light", "brick_warm", "stucco_cool", "painted_panel"))
    args = parser.parse_args(_script_args(argv))

    try:
        structure_variants = _parse_structure_variants(args.structure_variants)
        generation_result = _generate_dataset_v3_with_evidence(
            output_dir=args.output_dir,
            count=args.count,
            width=args.width,
            height=args.height,
            seed=args.seed,
            structure_variants=structure_variants,
            render_samples=args.render_samples,
            force_fallback=args.force_fallback,
            view_band=args.view_band,
            lighting_variant=args.lighting_variant,
            material_variant=args.material_variant,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    samples = generation_result.samples
    generated_variants = sorted(
        {sample.metadata["generation_params"]["structure_variant"] for sample in samples}
    )
    if generation_result.projection_fallback_count:
        raise RuntimeError("projection fallback output is not publishable")
    render_backend = "blenderproc_blender"
    summary = {
        "output_dir": str(args.output_dir),
        "sample_count": len(samples),
        "seed": args.seed,
        "structure_variants": generated_variants,
        "force_fallback": args.force_fallback,
        "render_backend": render_backend,
        "rendered_with_blender_count": generation_result.rendered_with_blender_count,
        "projection_fallback_count": generation_result.projection_fallback_count,
        "used_projection_fallback": generation_result.projection_fallback_count > 0,
    }
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


def _normalize_structure_variants(variants: list[str] | None) -> list[StructureVariant] | None:
    if variants is None:
        return None
    if not variants:
        raise ValueError("structure_variants must contain at least one value")

    normalized: list[StructureVariant] = []
    for variant in variants:
        if not isinstance(variant, str) or not variant.strip():
            raise ValueError("structure_variants must be non-empty strings")
        value = variant.strip()
        if value not in STRUCTURE_VARIANTS:
            supported = ", ".join(STRUCTURE_VARIANTS)
            raise ValueError(f"unsupported structure variant: {value}; supported: {supported}")
        normalized.append(cast(StructureVariant, value))
    return normalized


if __name__ == "__main__":
    raise SystemExit(main())
