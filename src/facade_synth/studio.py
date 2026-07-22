"""The public Generation Brief to Training Package publication seam."""

from __future__ import annotations

import json
import hashlib
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import fcntl

from .contracts import DatasetReceipt, GenerationBrief, GenerationJob, JobState, RenderedPackage
from .packages import (
    BlenderProcRenderer,
    RuntimeGateError,
    validate_local_assets,
    validate_frozen_sample_records,
    validate_task_annotations,
    validate_task_records,
)


class StudioService:
    """Persistent local studio lifecycle with explicit review and publication."""

    def __init__(self, *, workspace: Path, renderer: BlenderProcRenderer) -> None:
        self.workspace = Path(workspace).resolve()
        if type(renderer) is not BlenderProcRenderer:
            raise ValueError("StudioService accepts only the owned BlenderProcRenderer")
        self.renderer = renderer
        self._jobs: dict[str, GenerationJob] = {}
        self._run_lock = threading.Lock()
        (self.workspace / "jobs").mkdir(parents=True, exist_ok=True)
        (self.workspace / "packages").mkdir(parents=True, exist_ok=True)
        self._load_jobs()

    def create_job(self, brief: GenerationBrief) -> GenerationJob:
        with self._workspace_state_lock():
            self._reload_jobs()
            job = GenerationJob.new(brief)
            job.queue_sequence = max((existing.queue_sequence for existing in self._jobs.values()), default=0) + 1
            self._jobs[job.id] = job
            self._save(job)
            return job

    def get_job(self, job_id: str) -> GenerationJob:
        with self._workspace_state_lock():
            self._reload_jobs()
            return self._get_loaded_job(job_id)

    def _get_loaded_job(self, job_id: str) -> GenerationJob:
        """Return a job after the caller has refreshed durable workspace state."""

        try:
            return self._jobs[job_id]
        except KeyError as exc:
            raise ValueError(f"unknown job: {job_id}") from exc

    def list_jobs(self) -> list[GenerationJob]:
        with self._workspace_state_lock():
            self._reload_jobs()
            return sorted(self._jobs.values(), key=lambda job: job.queue_sequence)

    def confirm_brief(self, job_id: str, *, confirmed_by: str) -> GenerationJob:
        with self._workspace_state_lock():
            self._reload_jobs()
            job = self._get_loaded_job(job_id)
            if job.state is not JobState.DRAFT:
                raise ValueError("only a draft brief can be confirmed")
            if not confirmed_by.strip():
                raise ValueError("confirmed_by is required")
            job.confirmed_by = confirmed_by
            job.confirmed_brief_hash = job.brief.brief_hash
            job.state = JobState.QUEUED
            self._save(job)
            return job

    def run_next(self) -> GenerationJob:
        if not self._run_lock.acquire(blocking=False):
            raise ValueError("the local BlenderProc worker is already running a job")
        try:
            with self._workspace_worker_lock():
                with self._workspace_state_lock():
                    self._reload_jobs(recover_interrupted=True)
                    queued = self._claim_next_job()
                return self._render_claimed_job(queued)
        finally:
            self._run_lock.release()

    def _claim_next_job(self) -> GenerationJob:
        queued = min(
            (job for job in self._jobs.values() if job.state is JobState.QUEUED),
            key=lambda job: job.queue_sequence,
            default=None,
        )
        if queued is None:
            raise ValueError("no confirmed job is queued")
        self._assert_confirmed_brief_integrity(queued)
        queued.state = JobState.RUNNING
        self._save(queued)
        return queued

    def _render_claimed_job(self, queued: GenerationJob) -> GenerationJob:
        package_dir = self.workspace / "packages" / queued.id
        try:
            rendered = self.renderer.render(
                queued,
                package_dir,
                cancellation_requested=lambda: self._durable_cancel_requested(queued.id),
            )
            self._validate_rendered_package(queued, rendered)
        except Exception as exc:
            with self._workspace_state_lock():
                self._reload_jobs()
                current = self._get_loaded_job(queued.id)
                current.state = JobState.CANCELLED if current.cancelled_requested else JobState.FAILED
                current.failure_reason = str(exc)
                self._save(current)
            raise
        with self._workspace_state_lock():
            self._reload_jobs()
            current = self._get_loaded_job(queued.id)
            current.package_dir = rendered.package_dir
            current.validated_sample_count = rendered.validated_sample_count
            current.renderer_identity = rendered.renderer_identity
            current.code_revision = rendered.code_revision
            current.blender_version = rendered.blender_version
            current.blenderproc_version = rendered.blenderproc_version
            current.state = JobState.CANCELLED if current.cancelled_requested else JobState.READY_FOR_REVIEW
            self._save(current)
            return current

    def record_review(self, job_id: str, *, reviewer: str, approved: bool) -> GenerationJob:
        with self._workspace_state_lock():
            self._reload_jobs()
            job = self._get_loaded_job(job_id)
            if job.state is not JobState.READY_FOR_REVIEW:
                raise ValueError("only a completed job can be reviewed")
            if not reviewer.strip():
                raise ValueError("reviewer is required")
            job.reviewed_by = reviewer
            job.review_approved = approved
            self._save(job)
            return job

    def cancel(self, job_id: str) -> GenerationJob:
        with self._workspace_state_lock():
            self._reload_jobs()
            job = self._get_loaded_job(job_id)
            if job.state in {JobState.DRAFT, JobState.QUEUED}:
                job.state = JobState.CANCELLED
            elif job.state is JobState.RUNNING:
                job.cancelled_requested = True
            else:
                raise ValueError("only draft, queued, or running jobs can be cancelled")
            self._save(job)
            return job

    def resume(self, job_id: str) -> GenerationJob:
        with self._workspace_state_lock():
            self._reload_jobs()
            job = self._get_loaded_job(job_id)
            if job.state not in {JobState.FAILED, JobState.CANCELLED}:
                raise ValueError("only failed or cancelled jobs can be resumed")
            if job.confirmed_by is None:
                raise ValueError("only a confirmed brief can be resumed")
            self._assert_confirmed_brief_integrity(job)
            job.cancelled_requested = False
            job.failure_reason = None
            job.state = JobState.QUEUED
            self._save(job)
            return job

    def publish(self, job_id: str, *, published_by: str) -> DatasetReceipt:
        with self._workspace_state_lock():
            self._reload_jobs()
            job = self._get_loaded_job(job_id)
            if job.state is not JobState.READY_FOR_REVIEW:
                raise ValueError("only a completed job can be published")
            if job.review_approved is not True:
                raise ValueError("review approval is required before publication")
            if not published_by.strip():
                raise ValueError("published_by is required")
            self._assert_confirmed_brief_integrity(job)
            if (
            job.package_dir is None
            or job.renderer_identity is None
            or job.code_revision is None
            or job.blender_version is None
            or job.blenderproc_version is None
            ):
                raise ValueError("completed package evidence is missing")
            if job.renderer_identity != BlenderProcRenderer.identity:
                raise ValueError("completed package lacks owned BlenderProc identity evidence")
            package_evidence = self._validate_package_for_publication(job)
            receipt = DatasetReceipt(
            job_id=job.id,
            task=job.brief.task.value,
            output_target=job.brief.output_target,
            brief_hash=job.confirmed_brief_hash or "",
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
        if rendered.renderer_identity != BlenderProcRenderer.identity:
            raise ValueError("renderer did not provide owned BlenderProc identity evidence")
        if not rendered.code_revision.strip() or rendered.code_revision == "unknown":
            raise ValueError("renderer did not provide a source code revision")
        if not rendered.blender_version.strip() or not rendered.blenderproc_version.strip():
            raise ValueError("renderer did not provide Blender runtime evidence")
        manifest = Path(rendered.package_dir) / "manifest.jsonl"
        if not manifest.exists():
            raise ValueError("renderer did not write a package manifest")
        required = ("qa_summary.json", "preview/contact_sheet.png")
        if any(not (Path(rendered.package_dir) / relative).exists() for relative in required):
            raise ValueError("renderer did not write complete QA artifacts")
        try:
            records = [
                json.loads(line)
                for line in manifest.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            validate_task_records(records, brief=job.brief, package_dir=Path(rendered.package_dir))
            validate_task_annotations(records, brief=job.brief, package_dir=Path(rendered.package_dir))
            validate_frozen_sample_records(
                records,
                brief=job.brief,
                package_dir=Path(rendered.package_dir),
                provenance=self._execution_provenance(job, rendered),
            )
            validate_local_assets(job.brief)
        except (OSError, json.JSONDecodeError, RuntimeGateError) as exc:
            raise ValueError(f"renderer did not produce a validated Trainable Package: {exc}") from exc

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
        try:
            validate_task_records(records, brief=job.brief, package_dir=package_dir)
            validate_task_annotations(records, brief=job.brief, package_dir=package_dir)
            validate_frozen_sample_records(
                records,
                brief=job.brief,
                package_dir=package_dir,
                provenance={
                    "brief_hash": job.confirmed_brief_hash or "",
                    "renderer_identity": job.renderer_identity or "",
                    "code_revision": job.code_revision or "",
                    "blender_version": job.blender_version or "",
                    "blenderproc_version": job.blenderproc_version or "",
                },
            )
            validate_local_assets(job.brief)
        except RuntimeGateError as exc:
            raise ValueError(f"package failed final validation: {exc}") from exc
        return {
            "records": records,
            "manifest_sha256": _sha256_file(manifest),
            "qa_summary_sha256": _sha256_file(qa_summary),
            "contact_sheet_sha256": _sha256_file(contact_sheet),
        }

    def _load_jobs(self, *, recover_interrupted: bool = False) -> None:
        for sequence, file_path in enumerate(sorted((self.workspace / "jobs").glob("*.json")), start=1):
            payload = json.loads(file_path.read_text(encoding="utf-8"))
            job = GenerationJob.from_dict(payload)
            if job.queue_sequence <= 0:
                job.queue_sequence = sequence
                self._save(job)
            if recover_interrupted and job.state is JobState.RUNNING:
                job.state = JobState.FAILED
                job.failure_reason = "worker interrupted before a sample completed; resume is available"
                self._save(job)
            self._jobs[job.id] = job

    def _reload_jobs(self, *, recover_interrupted: bool = False) -> None:
        """Refresh durable state while the workspace state lock is held."""

        self._jobs.clear()
        self._load_jobs(recover_interrupted=recover_interrupted)

    def _durable_cancel_requested(self, job_id: str) -> bool:
        """Read the atomic job snapshot without replacing the render-held job object."""

        path = self.workspace / "jobs" / f"{job_id}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return GenerationJob.from_dict(payload).cancelled_requested
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            return True

    @staticmethod
    def _execution_provenance(job: GenerationJob, rendered: RenderedPackage) -> dict[str, str]:
        return {
            "brief_hash": job.confirmed_brief_hash or "",
            "renderer_identity": rendered.renderer_identity,
            "code_revision": rendered.code_revision,
            "blender_version": rendered.blender_version,
            "blenderproc_version": rendered.blenderproc_version,
        }

    @staticmethod
    def _assert_confirmed_brief_integrity(job: GenerationJob) -> None:
        if job.confirmed_brief_hash is None or job.confirmed_brief_hash != job.brief.brief_hash:
            raise ValueError("the immutable confirmed Generation Brief has changed")

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

    @contextmanager
    def _workspace_worker_lock(self) -> Iterator[None]:
        """Serialize one workspace even if two local Studio processes are launched."""

        lock_path = self.workspace / ".blenderproc-worker.lock"
        with lock_path.open("a+") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError("the local BlenderProc worker is already running a job") from exc
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @contextmanager
    def _workspace_state_lock(self) -> Iterator[None]:
        """Serialize durable job transitions without blocking the running renderer."""

        lock_path = self.workspace / ".studio-state.lock"
        with lock_path.open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
