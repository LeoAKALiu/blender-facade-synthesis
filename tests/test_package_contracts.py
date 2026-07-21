from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from facade_synth.contracts import GenerationBrief, GenerationJob, TaskKind
from facade_synth.packages import (
    RuntimeGateError,
    _cancel_requested_at_sample_boundary,
    fingerprint_local_asset,
    plan_samples,
    validate_local_assets,
)


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


if __name__ == "__main__":
    unittest.main()
