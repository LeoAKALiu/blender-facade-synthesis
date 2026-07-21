"""The public Generation Brief to Training Package publication seam."""

from __future__ import annotations

import json
import hashlib
import tempfile
import threading
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from .contracts import DatasetReceipt, GenerationBrief, GenerationJob, JobState, RenderedPackage


class TrainableRenderer(Protocol):
    """Worker-facing renderer that only returns already validated output."""

    def render(self, job: GenerationJob, package_dir: Path) -> RenderedPackage: ...


class InMemoryTrainableRenderer:
    """Test double for the publication contract; never selected by the Web Studio."""

    identity = "test-double/in-memory-trainable-renderer"

    def render(self, job: GenerationJob, package_dir: Path) -> RenderedPackage:
        package_dir.mkdir(parents=True, exist_ok=True)
        records = []
        split_names = tuple(job.brief.split_ratio)
        for index in range(job.brief.output_target):
            sample_id = f"facade_{index:06d}"
            split = split_names[index % len(split_names)]
            image_path = package_dir / "images" / f"{sample_id}.png"
            annotation_path = package_dir / "annotations" / f"{sample_id}.json"
            source_metadata_path = package_dir / "metadata" / f"{sample_id}.json"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            annotation_path.parent.mkdir(parents=True, exist_ok=True)
            source_metadata_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (2, 2), "white").save(image_path)
            annotation_path.write_text("{}", encoding="utf-8")
            source_metadata_path.write_text("{}", encoding="utf-8")
            record = {
                "sample_id": sample_id,
                "split": split,
                "task": job.brief.task.value,
                "validated": True,
                "render_backend": "blenderproc_blender",
                "used_projection_fallback": False,
                "rgb_path": image_path.relative_to(package_dir).as_posix(),
                "annotation_path": annotation_path.relative_to(package_dir).as_posix(),
                "source_metadata_path": source_metadata_path.relative_to(package_dir).as_posix(),
                "render_parameters": {
                    "seed": job.brief.seed + index,
                    "lighting_recipe": {
                        "sun_elevation_deg": 45.0,
                        "relative_azimuth_deg": 0.0,
                        "energy": 1.0,
                        "world_strength": 1.0,
                        "exposure_ev": 0.0,
                        "colour_temperature_k": 6500.0,
                    },
                },
            }
            records.append(record)
        (package_dir / "manifest.jsonl").write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
        )
        (package_dir / "qa_summary.json").write_text(
            json.dumps({"sample_count": len(records)}), encoding="utf-8"
        )
        preview = package_dir / "preview"
        preview.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (2, 2), "white").save(preview / "contact_sheet.png")
        return RenderedPackage(
            package_dir=str(package_dir),
            validated_sample_count=len(records),
            renderer_identity=self.identity,
            code_revision="test",
            blender_version="test",
            blenderproc_version="test",
            sample_records=tuple(records),
        )


class StudioService:
    """Persistent local studio lifecycle with explicit review and publication."""

    def __init__(self, *, workspace: Path, renderer: TrainableRenderer) -> None:
        self.workspace = Path(workspace)
        self.renderer = renderer
        self._jobs: dict[str, GenerationJob] = {}
        self._run_lock = threading.Lock()
        (self.workspace / "jobs").mkdir(parents=True, exist_ok=True)
        (self.workspace / "packages").mkdir(parents=True, exist_ok=True)
        self._load_jobs()

    def create_job(self, brief: GenerationBrief) -> GenerationJob:
        job = GenerationJob.new(brief)
        self._jobs[job.id] = job
        self._save(job)
        return job

    def get_job(self, job_id: str) -> GenerationJob:
        try:
            return self._jobs[job_id]
        except KeyError as exc:
            raise ValueError(f"unknown job: {job_id}") from exc

    def list_jobs(self) -> list[GenerationJob]:
        return list(self._jobs.values())

    def confirm_brief(self, job_id: str, *, confirmed_by: str) -> GenerationJob:
        job = self.get_job(job_id)
        if job.state is not JobState.DRAFT:
            raise ValueError("only a draft brief can be confirmed")
        if not confirmed_by.strip():
            raise ValueError("confirmed_by is required")
        job.confirmed_by = confirmed_by
        job.state = JobState.QUEUED
        self._save(job)
        return job

    def run_next(self) -> GenerationJob:
        if not self._run_lock.acquire(blocking=False):
            raise ValueError("the local BlenderProc worker is already running a job")
        try:
            return self._run_next_serially()
        finally:
            self._run_lock.release()

    def _run_next_serially(self) -> GenerationJob:
        queued = next((job for job in self._jobs.values() if job.state is JobState.QUEUED), None)
        if queued is None:
            raise ValueError("no confirmed job is queued")
        queued.state = JobState.RUNNING
        self._save(queued)
        package_dir = self.workspace / "packages" / queued.id
        try:
            rendered = self.renderer.render(queued, package_dir)
            self._validate_rendered_package(queued, rendered)
        except Exception as exc:
            queued.state = JobState.CANCELLED if queued.cancelled_requested else JobState.FAILED
            queued.failure_reason = str(exc)
            self._save(queued)
            raise
        queued.package_dir = rendered.package_dir
        queued.validated_sample_count = rendered.validated_sample_count
        queued.renderer_identity = rendered.renderer_identity
        queued.code_revision = rendered.code_revision
        queued.blender_version = rendered.blender_version
        queued.blenderproc_version = rendered.blenderproc_version
        queued.state = JobState.READY_FOR_REVIEW
        self._save(queued)
        return queued

    def record_review(self, job_id: str, *, reviewer: str, approved: bool) -> GenerationJob:
        job = self.get_job(job_id)
        if job.state is not JobState.READY_FOR_REVIEW:
            raise ValueError("only a completed job can be reviewed")
        if not reviewer.strip():
            raise ValueError("reviewer is required")
        job.reviewed_by = reviewer
        job.review_approved = approved
        self._save(job)
        return job

    def cancel(self, job_id: str) -> GenerationJob:
        job = self.get_job(job_id)
        if job.state in {JobState.DRAFT, JobState.QUEUED}:
            job.state = JobState.CANCELLED
        elif job.state is JobState.RUNNING:
            job.cancelled_requested = True
        else:
            raise ValueError("only draft, queued, or running jobs can be cancelled")
        self._save(job)
        return job

    def resume(self, job_id: str) -> GenerationJob:
        job = self.get_job(job_id)
        if job.state not in {JobState.FAILED, JobState.CANCELLED}:
            raise ValueError("only failed or cancelled jobs can be resumed")
        if job.confirmed_by is None:
            raise ValueError("only a confirmed brief can be resumed")
        job.cancelled_requested = False
        job.failure_reason = None
        job.state = JobState.QUEUED
        self._save(job)
        return job

    def publish(self, job_id: str, *, published_by: str) -> DatasetReceipt:
        job = self.get_job(job_id)
        if job.state is not JobState.READY_FOR_REVIEW:
            raise ValueError("only a completed job can be published")
        if job.review_approved is not True:
            raise ValueError("review approval is required before publication")
        if not published_by.strip():
            raise ValueError("published_by is required")
        if (
            job.package_dir is None
            or job.renderer_identity is None
            or job.code_revision is None
            or job.blender_version is None
            or job.blenderproc_version is None
        ):
            raise ValueError("completed package evidence is missing")
        package_evidence = self._validate_package_for_publication(job)
        receipt = DatasetReceipt(
            job_id=job.id,
            task=job.brief.task.value,
            output_target=job.brief.output_target,
            brief_hash=job.brief.brief_hash,
            package_dir=job.package_dir,
            renderer_identity=job.renderer_identity,
            code_revision=job.code_revision,
            blender_version=job.blender_version,
            blenderproc_version=job.blenderproc_version,
            asset_fingerprints=job.brief.asset_fingerprints,
            published_by=published_by,
            sample_seeds=tuple(sorted({int(record["render_parameters"]["seed"]) for record in package_evidence["records"]})),
            actual_render_parameters=tuple(record["render_parameters"] for record in package_evidence["records"]),
            validation_evidence={
                "validated_sample_count": job.validated_sample_count,
                "manifest_sha256": package_evidence["manifest_sha256"],
                "qa_summary_sha256": package_evidence["qa_summary_sha256"],
                "contact_sheet_sha256": package_evidence["contact_sheet_sha256"],
            },
            publication_decision={
                "reviewer": job.reviewed_by,
                "review_approved": job.review_approved,
                "published_by": published_by,
                "manual_publication_required": True,
            },
        )
        receipt_path = Path(job.package_dir) / "receipt.json"
        receipt_path.write_text(json.dumps(receipt.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        job.state = JobState.PUBLISHED
        self._save(job)
        return receipt

    def _validate_rendered_package(self, job: GenerationJob, rendered: RenderedPackage) -> None:
        if rendered.validated_sample_count != job.brief.output_target:
            raise ValueError("renderer did not produce the confirmed output_target")
        if not rendered.renderer_identity.strip():
            raise ValueError("renderer identity is required")
        manifest = Path(rendered.package_dir) / "manifest.jsonl"
        if not manifest.exists():
            raise ValueError("renderer did not write a package manifest")
        required = ("qa_summary.json", "preview/contact_sheet.png")
        if any(not (Path(rendered.package_dir) / relative).exists() for relative in required):
            raise ValueError("renderer did not write complete QA artifacts")

    def _validate_package_for_publication(self, job: GenerationJob) -> dict[str, Any]:
        if job.package_dir is None:
            raise ValueError("completed package evidence is missing")
        package_dir = Path(job.package_dir)
        manifest = package_dir / "manifest.jsonl"
        qa_summary = package_dir / "qa_summary.json"
        contact_sheet = package_dir / "preview" / "contact_sheet.png"
        if not manifest.exists() or not qa_summary.exists() or not contact_sheet.exists():
            raise ValueError("publication requires manifest, QA summary, and contact sheet")
        try:
            records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
            qa = json.loads(qa_summary.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"package evidence is unreadable: {exc}") from exc
        if len(records) != job.brief.output_target or qa.get("sample_count") != job.brief.output_target:
            raise ValueError("package QA evidence does not match the confirmed output target")
        for record in records:
            for key in ("rgb_path", "annotation_path", "source_metadata_path"):
                if not (package_dir / str(record.get(key, ""))).exists():
                    raise ValueError(f"package evidence references a missing {key}")
            if not isinstance(record.get("render_parameters"), dict):
                raise ValueError("package evidence lacks actual render parameters")
        return {
            "records": records,
            "manifest_sha256": _sha256_file(manifest),
            "qa_summary_sha256": _sha256_file(qa_summary),
            "contact_sheet_sha256": _sha256_file(contact_sheet),
        }

    def _load_jobs(self) -> None:
        for file_path in (self.workspace / "jobs").glob("*.json"):
            payload = json.loads(file_path.read_text(encoding="utf-8"))
            job = GenerationJob.from_dict(payload)
            if job.state is JobState.RUNNING:
                job.state = JobState.FAILED
                job.failure_reason = "worker interrupted before a sample completed; resume is available"
                self._save(job)
            self._jobs[job.id] = job

    def _save(self, job: GenerationJob) -> None:
        target = self.workspace / "jobs" / f"{job.id}.json"
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{job.id}.",
            suffix=".tmp",
            delete=False,
        ) as file_handle:
            file_handle.write(
            json.dumps(job.to_dict(), indent=2, sort_keys=True),
            )
            temporary_name = file_handle.name
        Path(temporary_name).replace(target)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
