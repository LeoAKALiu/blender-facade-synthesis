"""Public contracts for confirmed generation and publication."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Mapping
from uuid import uuid4


class TaskKind(StrEnum):
    WINDOW_INSTANCE_COUNT = "window_instance_count"
    FLOORLINE_HEATMAP = "floorline_heatmap"
    VISIBLE_FLOOR_COUNT = "visible_floor_count"
    BUILDING_USE = "building_use"
    FACADE_COMPONENT_SEGMENTATION = "facade_component_segmentation"


class JobState(StrEnum):
    DRAFT = "draft"
    QUEUED = "queued"
    RUNNING = "running"
    READY_FOR_REVIEW = "ready_for_review"
    PUBLISHED = "published"
    FAILED = "failed"
    CANCELLED = "cancelled"


BUILDING_USES = ("residential", "office", "commercial", "mixed_use")
COMPONENT_CLASSES = (
    "facade_wall",
    "window_glass",
    "window_frame",
    "door",
    "balcony",
    "floor_band",
    "podium_storefront",
    "roof_parapet",
    "background",
)
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class GenerationBrief:
    """The user-confirmed input for exactly one Task Dataset job."""

    task: TaskKind
    output_target: int
    split_ratio: Mapping[str, float]
    building_use_distribution: Mapping[str, float]
    render_width: int = 1024
    render_height: int = 768
    seed: int = 0
    view_family: tuple[str, ...] = ("frontal", "light_medium_oblique", "strong_oblique")
    daylight_profile: str = "daylight_diverse"
    occlusion_profile: str = "light_controlled_occlusion"
    asset_fingerprints: tuple[str, ...] = ()
    visibility_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.output_target < 1:
            raise ValueError("output_target must be at least 1")
        if self.render_width < 32 or self.render_height < 32:
            raise ValueError("render dimensions must be at least 32 pixels")
        if not 0.0 <= self.visibility_threshold <= 1.0:
            raise ValueError("visibility_threshold must be between 0 and 1")
        _validate_distribution(self.split_ratio, SPLITS, "split_ratio")
        _validate_distribution(
            self.building_use_distribution,
            BUILDING_USES,
            "building_use_distribution",
            require_all_keys=False,
        )
        if not self.view_family:
            raise ValueError("view_family must not be empty")
        if self.daylight_profile not in {"daylight_diverse", "controlled_daylight"}:
            raise ValueError("daylight_profile must be daylight_diverse or controlled_daylight")
        if self.occlusion_profile != "light_controlled_occlusion":
            raise ValueError("only light_controlled_occlusion is supported")

    @property
    def brief_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["task"] = self.task.value
        payload["split_ratio"] = dict(self.split_ratio)
        payload["building_use_distribution"] = dict(self.building_use_distribution)
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GenerationBrief":
        data = dict(value)
        data["task"] = TaskKind(data["task"])
        data["view_family"] = tuple(data.get("view_family", ()))
        data["asset_fingerprints"] = tuple(data.get("asset_fingerprints", ()))
        return cls(**data)


@dataclass
class GenerationJob:
    id: str
    brief: GenerationBrief
    state: JobState = JobState.DRAFT
    confirmed_by: str | None = None
    reviewed_by: str | None = None
    review_approved: bool | None = None
    validated_sample_count: int = 0
    package_dir: str | None = None
    failure_reason: str | None = None
    renderer_identity: str | None = None
    code_revision: str | None = None
    blender_version: str | None = None
    blenderproc_version: str | None = None
    cancelled_requested: bool = False

    @classmethod
    def new(cls, brief: GenerationBrief) -> "GenerationJob":
        return cls(id=f"job_{uuid4().hex}", brief=brief)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "brief": self.brief.to_dict(),
            "state": self.state.value,
            "confirmed_by": self.confirmed_by,
            "reviewed_by": self.reviewed_by,
            "review_approved": self.review_approved,
            "validated_sample_count": self.validated_sample_count,
            "package_dir": self.package_dir,
            "failure_reason": self.failure_reason,
            "renderer_identity": self.renderer_identity,
            "code_revision": self.code_revision,
            "blender_version": self.blender_version,
            "blenderproc_version": self.blenderproc_version,
            "cancelled_requested": self.cancelled_requested,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GenerationJob":
        data = dict(value)
        data["brief"] = GenerationBrief.from_dict(data["brief"])
        data["state"] = JobState(data["state"])
        return cls(**data)


@dataclass(frozen=True)
class RenderedPackage:
    package_dir: str
    validated_sample_count: int
    renderer_identity: str
    code_revision: str
    blender_version: str
    blenderproc_version: str
    sample_records: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class DatasetReceipt:
    job_id: str
    task: str
    output_target: int
    brief_hash: str
    package_dir: str
    renderer_identity: str
    code_revision: str
    blender_version: str
    blenderproc_version: str
    asset_fingerprints: tuple[str, ...]
    published_by: str
    validation_status: str = "passed"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_distribution(
    value: Mapping[str, float],
    allowed_keys: tuple[str, ...],
    label: str,
    *,
    require_all_keys: bool = True,
) -> None:
    actual = set(value)
    allowed = set(allowed_keys)
    if not actual or not actual.issubset(allowed) or (require_all_keys and actual != allowed):
        qualifier = "exactly" if require_all_keys else "only"
        raise ValueError(f"{label} must contain {qualifier} {', '.join(allowed_keys)}")
    if any(not isinstance(item, (int, float)) or item < 0 for item in value.values()):
        raise ValueError(f"{label} values must be non-negative numbers")
    if abs(sum(float(item) for item in value.values()) - 1.0) > 1e-9:
        raise ValueError(f"{label} must sum to 1")
