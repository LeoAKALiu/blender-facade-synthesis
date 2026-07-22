from __future__ import annotations

import subprocess
import unittest
from unittest.mock import patch

from facade_synth.runtime import BlenderProcRuntime, EnvironmentNotReady, RuntimeGateError, validate_render_summary


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

    def test_generator_process_failure_is_not_an_environment_preflight_failure(self) -> None:
        runtime = BlenderProcRuntime()
        failed_run = subprocess.CompletedProcess((), returncode=2, stdout="scene failure", stderr="")

        with patch("facade_synth.runtime.subprocess.run", return_value=failed_run):
            with self.assertRaises(RuntimeGateError) as caught:
                runtime.run_generator(("--output-dir", "sample"))

        self.assertNotIsInstance(caught.exception, EnvironmentNotReady)

    def test_preflight_process_failure_remains_an_environment_failure(self) -> None:
        runtime = BlenderProcRuntime()
        failed_run = subprocess.CompletedProcess((), returncode=2, stdout="", stderr="broken runtime")

        with patch("facade_synth.runtime.subprocess.run", return_value=failed_run):
            with self.assertRaises(EnvironmentNotReady):
                runtime.preflight()


if __name__ == "__main__":
    unittest.main()
