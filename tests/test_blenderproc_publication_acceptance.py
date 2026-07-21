from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from facade_synth.contracts import GenerationBrief, TaskKind
from facade_synth.packages import BlenderProcRenderer
from facade_synth.runtime import BlenderProcRuntime
from facade_synth.studio import StudioService


@unittest.skipUnless(
    os.environ.get("BLENDERPROC_ACCEPTANCE") == "1" and shutil.which("blenderproc"),
    "set BLENDERPROC_ACCEPTANCE=1 with BlenderProc on PATH to run the real worker acceptance seam",
)
class BlenderProcPublicationAcceptanceTests(unittest.TestCase):
    def test_every_task_publishes_a_real_blenderproc_training_package(self) -> None:
        for task in TaskKind:
            with self.subTest(task=task.value), tempfile.TemporaryDirectory() as temp_dir:
                workspace = Path(temp_dir)
                studio = StudioService(
                    workspace=workspace,
                    renderer=BlenderProcRenderer(
                        runtime=BlenderProcRuntime(executable=shutil.which("blenderproc") or "blenderproc"),
                        render_samples=1,
                    ),
                )
                job = studio.create_job(
                    GenerationBrief(
                        task=task,
                        output_target=3,
                        split_ratio={"train": 1.0, "validation": 0.0, "test": 0.0},
                        building_use_distribution={"residential": 1.0},
                        render_width=192,
                        render_height=144,
                        seed=101,
                        lighting_intensity_range={"min": 1.4, "max": 1.4},
                    )
                )
                studio.confirm_brief(job.id, confirmed_by="acceptance-test")
                completed = studio.run_next()
                self.assertEqual("ready_for_review", completed.state.value)
                self.assertEqual(3, completed.validated_sample_count)

                package_dir = Path(completed.package_dir or "")
                records = [json.loads(line) for line in (package_dir / "manifest.jsonl").read_text().splitlines()]
                self.assertEqual(3, len(records))
                self.assertTrue(all(record["render_backend"] == "blenderproc_blender" for record in records))
                self.assertTrue(
                    all(record["render_parameters"]["lighting_recipe"]["intensity_scale"] == 1.4 for record in records)
                )
                annotation = json.loads((package_dir / records[0]["annotation_path"]).read_text())
                _assert_task_native_annotation(self, task, annotation)

                studio.record_review(job.id, reviewer="acceptance-test", approved=True)
                receipt = studio.publish(job.id, published_by="acceptance-test")
                self.assertEqual("passed", receipt.validation_status)
                self.assertTrue((package_dir / "receipt.json").exists())


def _assert_task_native_annotation(test: unittest.TestCase, task: TaskKind, annotation: dict) -> None:
    if task is TaskKind.WINDOW_INSTANCE_COUNT:
        test.assertEqual([{"id": 1, "name": "window", "supercategory": "facade"}], annotation["categories"])
        test.assertGreater(len(annotation["annotations"]), 0)
    elif task is TaskKind.FLOORLINE_HEATMAP:
        test.assertTrue(annotation["floorline_polylines_px"])
    elif task is TaskKind.VISIBLE_FLOOR_COUNT:
        test.assertGreater(annotation["visible_floor_count"], 0)
    elif task is TaskKind.BUILDING_USE:
        test.assertEqual("residential", annotation["building_use"])
    else:
        test.assertEqual("visible_raster_only", annotation["target"])


if __name__ == "__main__":
    unittest.main()
