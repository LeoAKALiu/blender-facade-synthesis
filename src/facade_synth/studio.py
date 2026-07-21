"""The public Generation Brief to Training Package publication seam."""

from __future__ import annotations

import json
import tempfile
import threading
from pathlib import Path
from typing import Protocol

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
            record = {
                "sample_id": sample_id,
                "split": split,
                "task": job.brief.task.value,
                "validated": True,
                "render_backend": "blenderproc_blender",
                "used_projection_fallback": False,
            }
            records.append(record)
        (package_dir / "manifest.jsonl").write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
        )
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
