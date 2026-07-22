from __future__ import annotations

import unittest

from facade_synth.seed_v31.blenderproc_facade_v3 import render_rgb_for_structure_spec
from facade_synth.seed_v31.structure_scene import sample_structure_scene_spec


class SeedV31RegressionTests(unittest.TestCase):
    def test_structure_scene_remains_deterministic_after_local_absorption(self) -> None:
        first = sample_structure_scene_spec(
            sample_id="facade_000000",
            seed=301,
            structure_variant="balcony_residential",
            width_px=192,
            height_px=144,
        )
        second = sample_structure_scene_spec(
            sample_id="facade_000000",
            seed=301,
            structure_variant="balcony_residential",
            width_px=192,
            height_px=144,
        )
        self.assertEqual(first, second)
        self.assertTrue(first.windows)
        self.assertEqual(tuple(range(1, len(first.windows) + 1)), tuple(window.instance_id for window in first.windows))

    def test_controlled_view_and_lighting_reach_scene_truth(self) -> None:
        spec = sample_structure_scene_spec(
            sample_id="facade_000000",
            seed=302,
            structure_variant="commercial_curtain_wall",
            width_px=192,
            height_px=144,
            view_band="strong_oblique",
            lighting_variant="overcast",
            material_variant="painted_panel",
        )
        self.assertGreaterEqual(abs(spec.label_scene_spec.camera_view.azimuth_deg), 24.0)
        self.assertEqual("overcast", spec.label_scene_spec.lighting_variant)
        self.assertEqual("painted_panel", spec.label_scene_spec.material_variant)

    def test_projection_fallback_is_disabled_in_locally_owned_runner(self) -> None:
        spec = sample_structure_scene_spec(
            sample_id="facade_000000",
            seed=303,
            structure_variant="residential_recessed",
            width_px=128,
            height_px=96,
        )
        with self.assertRaisesRegex(ValueError, "projection fallback is disabled"):
            render_rgb_for_structure_spec(
                spec,
                width=128,
                height=96,
                render_samples=1,
                force_fallback=True,
            )


if __name__ == "__main__":
    unittest.main()
