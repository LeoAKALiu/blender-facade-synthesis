from __future__ import annotations

import unittest

from facade_synth.runtime import RuntimeGateError, validate_render_summary


class RuntimeGateTests(unittest.TestCase):
    def test_rejects_projection_fallback_even_when_output_files_exist(self) -> None:
        with self.assertRaisesRegex(RuntimeGateError, "projection fallback"):
            validate_render_summary(
                {
                    "sample_count": 1,
                    "rendered_with_blender_count": 0,
                    "projection_fallback_count": 1,
                    "used_projection_fallback": True,
                    "render_backend": "projection_fallback",
                },
                expected_count=1,
            )

    def test_accepts_only_complete_blenderproc_blender_summary(self) -> None:
        validate_render_summary(
            {
                "sample_count": 2,
                "rendered_with_blender_count": 2,
                "projection_fallback_count": 0,
                "used_projection_fallback": False,
                "render_backend": "blenderproc_blender",
            },
            expected_count=2,
        )


if __name__ == "__main__":
    unittest.main()
