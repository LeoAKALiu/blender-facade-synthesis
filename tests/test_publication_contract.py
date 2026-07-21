from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from facade_synth.contracts import GenerationBrief, JobState, TaskKind
from facade_synth.packages import (
    BlenderProcRenderer,
    plan_samples,
    source_artifact_sha256,
    task_artifact_sha256,
)
from facade_synth.studio import StudioService


class PublicationContractTests(unittest.TestCase):
    def test_service_rejects_non_blenderproc_renderer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "owned BlenderProcRenderer"):
                StudioService(workspace=Path(temp_dir), renderer=object())  # type: ignore[arg-type]

    def test_confirmed_brief_reaches_manually_published_training_package(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            studio, job = _ready_building_use_job(Path(temp_dir))

            with self.assertRaisesRegex(ValueError, "review"):
                studio.publish(job.id, published_by="leo")

            studio.record_review(job.id, reviewer="leo", approved=True)
            receipt = studio.publish(job.id, published_by="leo")

            self.assertEqual("published", studio.get_job(job.id).state.value)
            self.assertEqual(3, receipt.output_target)
            self.assertEqual(TaskKind.BUILDING_USE.value, receipt.task)
            self.assertTrue((Path(temp_dir) / "packages" / job.id / "receipt.json").exists())

    def test_interrupted_running_job_becomes_resumable_after_worker_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            first = StudioService(workspace=workspace, renderer=BlenderProcRenderer())
            job = first.create_job(_brief())
            job = first.confirm_brief(job.id, confirmed_by="leo")
            payload = job.to_dict()
            payload["state"] = "running"
            (workspace / "jobs" / f"{job.id}.json").write_text(json.dumps(payload), encoding="utf-8")

            restarted = StudioService(workspace=workspace, renderer=BlenderProcRenderer())
            self.assertEqual("running", restarted.get_job(job.id).state.value)
            with self.assertRaisesRegex(ValueError, "no confirmed"):
                restarted.run_next()
            recovered = restarted.get_job(job.id)
            self.assertEqual("failed", recovered.state.value)
            self.assertIn("interrupted", recovered.failure_reason or "")
            self.assertEqual("queued", restarted.resume(job.id).state.value)

    def test_second_worker_reloads_job_state_after_lock_acquisition(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            first = StudioService(workspace=workspace, renderer=BlenderProcRenderer())
            job = first.create_job(_brief())
            first.confirm_brief(job.id, confirmed_by="leo")
            second = StudioService(workspace=workspace, renderer=BlenderProcRenderer())
            job.state = JobState.READY_FOR_REVIEW
            first._save(job)

            with self.assertRaisesRegex(ValueError, "no confirmed"):
                second.run_next()

    def test_cross_process_state_transitions_refresh_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            first = StudioService(workspace=workspace, renderer=BlenderProcRenderer())
            first_job = first.create_job(_brief())
            stale_second = StudioService(workspace=workspace, renderer=BlenderProcRenderer())

            second_job = stale_second.create_job(_brief())
            self.assertEqual(1, first_job.queue_sequence)
            self.assertEqual(2, second_job.queue_sequence)

            first.confirm_brief(first_job.id, confirmed_by="first")
            with self.assertRaisesRegex(ValueError, "only a draft"):
                stale_second.confirm_brief(first_job.id, confirmed_by="stale")

    def test_running_job_cancellation_is_visible_to_the_worker_without_reloading_it(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            worker = StudioService(workspace=workspace, renderer=BlenderProcRenderer())
            job = worker.create_job(_brief())
            job = worker.confirm_brief(job.id, confirmed_by="worker")
            job.state = JobState.RUNNING
            worker._save(job)
            cancelling_process = StudioService(workspace=workspace, renderer=BlenderProcRenderer())

            cancelling_process.cancel(job.id)

            self.assertTrue(worker._durable_cancel_requested(job.id))

    def test_publication_rejects_changed_manifest_or_task_label(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            studio, job = _ready_building_use_job(Path(temp_dir))
            package_dir = Path(job.package_dir or "")
            records = _read_records(package_dir)
            records[0]["window_count"] = 99
            _write_records(package_dir, records)
            studio.record_review(job.id, reviewer="leo", approved=True)

            with self.assertRaisesRegex(ValueError, "manifest record changed"):
                studio.publish(job.id, published_by="leo")

        with tempfile.TemporaryDirectory() as temp_dir:
            studio, job = _ready_building_use_job(Path(temp_dir))
            package_dir = Path(job.package_dir or "")
            records = _read_records(package_dir)
            annotation_path = package_dir / records[0]["annotation_path"]
            annotation_path.write_text(json.dumps({"task": "building_use", "building_use": "residential"}), encoding="utf-8")
            studio.record_review(job.id, reviewer="leo", approved=True)

            with self.assertRaisesRegex(ValueError, "task artifacts changed"):
                studio.publish(job.id, published_by="leo")

    def test_publication_rejects_changed_source_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            studio, job = _ready_building_use_job(Path(temp_dir))
            package_dir = Path(job.package_dir or "")
            record = _read_records(package_dir)[0]
            (package_dir / record["source_metadata_path"]).write_text("changed", encoding="utf-8")
            studio.record_review(job.id, reviewer="leo", approved=True)

            with self.assertRaisesRegex(ValueError, "source artifacts changed"):
                studio.publish(job.id, published_by="leo")

    def test_confirmed_brief_distributions_are_immutable(self) -> None:
        brief = _brief()
        with self.assertRaises(TypeError):
            brief.split_ratio["train"] = 0.0  # type: ignore[index]


def _ready_building_use_job(workspace: Path) -> tuple[StudioService, object]:
    studio = StudioService(workspace=workspace, renderer=BlenderProcRenderer())
    job = studio.create_job(_brief())
    job = studio.confirm_brief(job.id, confirmed_by="leo")
    package_dir = workspace / "packages" / job.id
    records = _write_frozen_building_use_package(package_dir, job)
    job.package_dir = str(package_dir)
    job.validated_sample_count = len(records)
    job.renderer_identity = BlenderProcRenderer.identity
    job.code_revision = "0123456789abcdef"
    job.blender_version = "4.2.1"
    job.blenderproc_version = "2.8.0"
    job.state = JobState.READY_FOR_REVIEW
    studio._save(job)
    return studio, job


def _write_frozen_building_use_package(package_dir: Path, job: object) -> list[dict]:
    package_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for planned in plan_samples(job.brief):  # type: ignore[attr-defined]
        sample_root = package_dir / "seed_samples" / planned.sample_id
        metadata = sample_root / "metadata" / "facade_000000_metadata.json"
        rgb = sample_root / "images" / "facade_000000.png"
        metadata.parent.mkdir(parents=True, exist_ok=True)
        rgb.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_text("{}", encoding="utf-8")
        Image.new("RGB", (2, 2), "white").save(rgb)
        annotation = package_dir / "annotations" / f"{planned.sample_id}.json"
        annotation.parent.mkdir(parents=True, exist_ok=True)
        annotation.write_text(
            json.dumps({"task": "building_use", "building_use": planned.building_use}),
            encoding="utf-8",
        )
        record = {
            "sample_id": planned.sample_id,
            "recipe_id": planned.recipe_id,
            "split": planned.split,
            "task": job.brief.task.value,  # type: ignore[attr-defined]
            "rgb_path": rgb.relative_to(package_dir).as_posix(),
            "annotation_path": annotation.relative_to(package_dir).as_posix(),
            "source_metadata_path": metadata.relative_to(package_dir).as_posix(),
            "source_artifact_sha256": source_artifact_sha256(sample_root),
            "render_backend": "blenderproc_blender",
            "used_projection_fallback": False,
            "building_use": planned.building_use,
            "view_band": planned.view_band,
            "daylight_condition": planned.daylight_condition,
            "lighting_intensity_scale": planned.lighting_intensity_scale,
            "occlusion_band": planned.occlusion_band,
            "occlusion_ratio": {"clear": 0.0, "light_0_15": 0.12, "moderate_15_30": 0.24}[planned.occlusion_band],
            "visible_floor_count": 1,
            "window_count": 1,
            "visibility_score": 1.0,
            "scene_truth": {"component_mask_origin": "blender_object_index_pass"},
            "render_parameters": {
                "seed": planned.seed,
                "lighting_recipe": {
                    "sun_elevation_deg": 45.0,
                    "relative_azimuth_deg": 0.0,
                    "energy": 1.0,
                    "world_strength": 1.0,
                    "exposure_ev": 0.0,
                    "colour_temperature_k": 6500.0,
                    "intensity_scale": planned.lighting_intensity_scale,
                },
            },
        }
        record["task_artifact_sha256"] = task_artifact_sha256(package_dir, record)
        (sample_root / "validated_record.json").write_text(
            json.dumps(
                {
                    "provenance": {
                        "brief_hash": job.confirmed_brief_hash,  # type: ignore[attr-defined]
                        "renderer_identity": BlenderProcRenderer.identity,
                        "code_revision": "0123456789abcdef",
                        "blender_version": "4.2.1",
                        "blenderproc_version": "2.8.0",
                    },
                    "planned_sample": asdict(planned),
                    "record": record,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        records.append(record)
    _write_records(package_dir, records)
    (package_dir / "qa_summary.json").write_text(json.dumps({"sample_count": len(records)}), encoding="utf-8")
    preview = package_dir / "preview"
    preview.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2), "white").save(preview / "contact_sheet.png")
    return records


def _read_records(package_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (package_dir / "manifest.jsonl").read_text().splitlines()]


def _write_records(package_dir: Path, records: list[dict]) -> None:
    (package_dir / "manifest.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8"
    )


def _brief() -> GenerationBrief:
    return GenerationBrief(
        task=TaskKind.BUILDING_USE,
        output_target=3,
        split_ratio={"train": 1.0, "validation": 0.0, "test": 0.0},
        building_use_distribution={"office": 1.0},
        render_width=64,
        render_height=64,
        seed=7,
    )


if __name__ == "__main__":
    unittest.main()
