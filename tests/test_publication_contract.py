from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from facade_synth.contracts import GenerationBrief, TaskKind
from facade_synth.studio import InMemoryTrainableRenderer, StudioService


class PublicationContractTests(unittest.TestCase):
    def test_confirmed_brief_reaches_manually_published_training_package(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            studio = StudioService(
                workspace=Path(temp_dir),
                renderer=InMemoryTrainableRenderer(),
            )
            brief = GenerationBrief(
                task=TaskKind.WINDOW_INSTANCE_COUNT,
                output_target=3,
                split_ratio={"train": 0.5, "validation": 0.5, "test": 0.0},
                building_use_distribution={"residential": 1.0},
                render_width=64,
                render_height=64,
                seed=7,
            )

            job = studio.create_job(brief)
            with self.assertRaisesRegex(ValueError, "confirmed"):
                studio.run_next()

            studio.confirm_brief(job.id, confirmed_by="leo")
            completed = studio.run_next()
            self.assertEqual("ready_for_review", completed.state.value)
            self.assertEqual(3, completed.validated_sample_count)

            with self.assertRaisesRegex(ValueError, "review"):
                studio.publish(job.id, published_by="leo")

            studio.record_review(job.id, reviewer="leo", approved=True)
            receipt = studio.publish(job.id, published_by="leo")

            self.assertEqual("published", studio.get_job(job.id).state.value)
            self.assertEqual(3, receipt.output_target)
            self.assertEqual(TaskKind.WINDOW_INSTANCE_COUNT.value, receipt.task)
            self.assertTrue((Path(temp_dir) / "packages" / job.id / "receipt.json").exists())

    def test_interrupted_running_job_becomes_resumable_after_worker_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            first = StudioService(workspace=workspace, renderer=InMemoryTrainableRenderer())
            job = first.create_job(
                GenerationBrief(
                    task=TaskKind.VISIBLE_FLOOR_COUNT,
                    output_target=3,
                    split_ratio={"train": 1.0, "validation": 0.0, "test": 0.0},
                    building_use_distribution={"residential": 1.0},
                    render_width=64,
                    render_height=64,
                )
            )
            first.confirm_brief(job.id, confirmed_by="leo")
            payload = job.to_dict()
            payload["state"] = "running"
            (workspace / "jobs" / f"{job.id}.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )

            restarted = StudioService(workspace=workspace, renderer=InMemoryTrainableRenderer())
            recovered = restarted.get_job(job.id)
            self.assertEqual("failed", recovered.state.value)
            self.assertIn("interrupted", recovered.failure_reason)

            resumed = restarted.resume(job.id)
            self.assertEqual("queued", resumed.state.value)


if __name__ == "__main__":
    unittest.main()
