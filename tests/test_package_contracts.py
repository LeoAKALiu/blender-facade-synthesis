from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from facade_synth.contracts import GenerationBrief, GenerationJob, TaskKind
from facade_synth.packages import (
    BlenderProcRenderer,
    RuntimeGateError,
    _cancel_requested_at_sample_boundary,
    fingerprint_local_asset,
    plan_samples,
    validate_local_assets,
)
from facade_synth.seed_v31.validate_dataset import DatasetValidationError


class PackageContractTests(unittest.TestCase):
    def test_recipe_owns_all_confirmed_views_and_one_split(self) -> None:
        brief = _brief(output_target=6)

        plan = plan_samples(brief)

        self.assertEqual(6, len(plan))
        by_recipe: dict[str, list] = {}
        for sample in plan:
            by_recipe.setdefault(sample.recipe_id, []).append(sample)
        self.assertEqual(2, len(by_recipe))
        for samples in by_recipe.values():
            self.assertEqual(set(brief.view_family), {sample.view_band for sample in samples})
            self.assertEqual(1, len({sample.split for sample in samples}))

    def test_target_must_include_the_complete_view_family(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible"):
            _brief(output_target=4)

    def test_partial_view_family_is_not_a_first_release_brief(self) -> None:
        with self.assertRaisesRegex(ValueError, "complete first-release view_family"):
            GenerationBrief(
                task=TaskKind.FACADE_COMPONENT_SEGMENTATION,
                output_target=1,
                split_ratio={"train": 1.0, "validation": 0.0, "test": 0.0},
                building_use_distribution={"residential": 1.0},
                render_width=64,
                render_height=64,
                view_family=("frontal",),
            )

    def test_changed_confirmed_asset_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            asset = Path(temp_dir) / "facade.png"
            asset.write_bytes(b"original")
            brief = _brief(
                output_target=3,
                asset_paths=(str(asset),),
                asset_fingerprints=(fingerprint_local_asset(asset),),
            )
            validate_local_assets(brief)
            asset.write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeGateError, "changed"):
                validate_local_assets(brief)

    def test_sample_boundary_observes_the_durable_cancellation_callback(self) -> None:
        job = GenerationJob.new(_brief(output_target=3))

        self.assertFalse(_cancel_requested_at_sample_boundary(job, lambda: False))
        self.assertTrue(_cancel_requested_at_sample_boundary(job, lambda: True))

    def test_invalid_resume_cache_is_quarantined_and_rerendered(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            package_dir = Path(temp_dir) / "package"
            job = GenerationJob.new(_brief(output_target=3))
            planned = plan_samples(job.brief)[0]
            sample_root = package_dir / "seed_samples" / planned.sample_id
            sample_root.mkdir(parents=True)
            provenance = {
                "brief_hash": job.brief.brief_hash,
                "renderer_identity": BlenderProcRenderer.identity,
                "code_revision": "0123456789abcdef",
                "blender_version": "4.2.1",
                "blenderproc_version": "2.8.0",
            }
            cached_record = {"sample_id": planned.sample_id, "recipe_id": planned.recipe_id}
            (sample_root / "validated_record.json").write_text(
                json.dumps(
                    {"provenance": provenance, "planned_sample": asdict(planned), "record": cached_record}
                ),
                encoding="utf-8",
            )
            runtime = _RerenderingRuntime()
            replacement_record = {"sample_id": planned.sample_id, "recipe_id": planned.recipe_id}
            renderer = BlenderProcRenderer(runtime=runtime)

            with (
                patch("facade_synth.packages._code_revision", return_value="0123456789abcdef"),
                patch("facade_synth.packages.plan_samples", return_value=[planned]),
                patch(
                    "facade_synth.packages.validate_dataset",
                    side_effect=[DatasetValidationError("stale cached sample"), None],
                ),
                patch("facade_synth.packages.build_task_record", return_value=replacement_record),
                patch("facade_synth.packages.validate_task_records"),
                patch("facade_synth.packages.validate_task_annotations"),
                patch("facade_synth.packages.write_contact_sheet"),
                patch("facade_synth.packages.write_qa_summary"),
            ):
                renderer.render(job, package_dir)

            self.assertEqual(1, runtime.generate_calls)
            self.assertEqual("rerendered", (sample_root / "stale-marker.txt").read_text(encoding="utf-8"))
            self.assertTrue((sample_root / "validated_record.json").exists())
            self.assertTrue(any((package_dir / "invalid_samples").iterdir()))

    def test_confirmed_assets_are_revalidated_before_each_facade_sample(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            package_dir = Path(temp_dir) / "package"
            job = GenerationJob.new(_brief(output_target=3))
            plan = plan_samples(job.brief)[:2]
            runtime = _RerenderingRuntime()
            renderer = BlenderProcRenderer(runtime=runtime)

            with (
                patch("facade_synth.packages._code_revision", return_value="0123456789abcdef"),
                patch("facade_synth.packages.plan_samples", return_value=plan),
                patch("facade_synth.packages.validate_local_assets") as validate_assets,
                patch("facade_synth.packages.validate_dataset"),
                patch(
                    "facade_synth.packages.build_task_record",
                    side_effect=[
                        {"sample_id": sample.sample_id, "recipe_id": sample.recipe_id}
                        for sample in plan
                    ],
                ),
                patch("facade_synth.packages.validate_task_records"),
                patch("facade_synth.packages.validate_task_annotations"),
                patch("facade_synth.packages.write_contact_sheet"),
                patch("facade_synth.packages.write_qa_summary"),
            ):
                renderer.render(job, package_dir)

            self.assertEqual(1 + len(plan), validate_assets.call_count)
            self.assertEqual(len(plan), runtime.generate_calls)


def _brief(
    *,
    output_target: int,
    asset_paths: tuple[str, ...] = (),
    asset_fingerprints: tuple[str, ...] = (),
) -> GenerationBrief:
    return GenerationBrief(
        task=TaskKind.FACADE_COMPONENT_SEGMENTATION,
        output_target=output_target,
        split_ratio={"train": 0.5, "validation": 0.5, "test": 0.0},
        building_use_distribution={"residential": 1.0},
        render_width=64,
        render_height=64,
        asset_paths=asset_paths,
        asset_fingerprints=asset_fingerprints,
    )


class _RerenderingRuntime:
    def __init__(self) -> None:
        self.generate_calls = 0

    def preflight(self) -> dict[str, str]:
        return {"blender_version": "4.2.1", "blenderproc_version": "2.8.0"}

    def run_generator(self, arguments: tuple[str, ...]) -> dict[str, object]:
        self.generate_calls += 1
        output_dir = Path(arguments[arguments.index("--output-dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "stale-marker.txt").write_text("rerendered", encoding="utf-8")
        return {
            "sample_count": 1,
            "rendered_with_blender_count": 1,
            "projection_fallback_count": 0,
            "used_projection_fallback": False,
            "render_backend": "blenderproc_blender",
        }


if __name__ == "__main__":
    unittest.main()
