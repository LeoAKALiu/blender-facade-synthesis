"""Public contracts for confirmed generation and publication."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from types import MappingProxyType
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
VIEW_BANDS = ("frontal", "light_medium_oblique", "strong_oblique")
DAYLIGHT_CONDITIONS = ("clear", "overcast", "warm_low_angle", "backlit")
CONTROLLED_DAYLIGHT_CONDITIONS = ("clear", "overcast")
OCCLUSION_BANDS = ("clear", "light_0_15", "moderate_15_30")
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
    view_family: tuple[str, ...] = VIEW_BANDS
    view_distribution: Mapping[str, float] = field(default_factory=dict)
    daylight_profile: str = "daylight_diverse"
    daylight_distribution: Mapping[str, float] = field(default_factory=dict)
    lighting_intensity_range: Mapping[str, float] = field(default_factory=lambda: {"min": 0.8, "max": 1.2})
    occlusion_profile: str = "light_controlled_occlusion"
    occlusion_distribution: Mapping[str, float] = field(default_factory=dict)
    target_domain: str = "china_post_2000_urban_facades"
    asset_paths: tuple[str, ...] = ()
    asset_fingerprints: tuple[str, ...] = ()
    task_visibility_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.output_target < 1:
            raise ValueError("output_target must be at least 1")
        if self.render_width < 32 or self.render_height < 32:
            raise ValueError("render dimensions must be at least 32 pixels")
        if not 0.0 <= self.task_visibility_threshold <= 1.0:
            raise ValueError("task_visibility_threshold must be between 0 and 1")
        split_ratio = dict(self.split_ratio)
        _validate_distribution(split_ratio, SPLITS, "split_ratio")
        object.__setattr__(self, "split_ratio", MappingProxyType(split_ratio))
        building_use_distribution = dict(self.building_use_distribution)
        _validate_distribution(
            building_use_distribution,
            BUILDING_USES,
            "building_use_distribution",
            require_all_keys=False,
        )
        object.__setattr__(self, "building_use_distribution", MappingProxyType(building_use_distribution))
        if tuple(self.view_family) != VIEW_BANDS:
            raise ValueError("view_family must be the complete first-release view_family")
        if self.output_target % len(VIEW_BANDS) != 0:
            raise ValueError("output_target must be divisible by the complete view_family")
        view_distribution = dict(self.view_distribution) or _equal_distribution(VIEW_BANDS)
        _validate_distribution(view_distribution, VIEW_BANDS, "view_distribution")
        if view_distribution != _equal_distribution(VIEW_BANDS):
            raise ValueError("view_distribution must represent every view in each Building Recipe")
        object.__setattr__(self, "view_distribution", MappingProxyType(view_distribution))
        if self.daylight_profile not in {"daylight_diverse", "controlled_daylight"}:
            raise ValueError("daylight_profile must be daylight_diverse or controlled_daylight")
        daylight_conditions = (
            DAYLIGHT_CONDITIONS
            if self.daylight_profile == "daylight_diverse"
            else CONTROLLED_DAYLIGHT_CONDITIONS
        )
        daylight_distribution = dict(self.daylight_distribution) or _equal_distribution(daylight_conditions)
        _validate_distribution(daylight_distribution, daylight_conditions, "daylight_distribution")
        object.__setattr__(self, "daylight_distribution", MappingProxyType(daylight_distribution))
        lighting_intensity_range = dict(self.lighting_intensity_range)
        if set(lighting_intensity_range) != {"min", "max"}:
            raise ValueError("lighting_intensity_range must contain min and max")
        minimum = lighting_intensity_range["min"]
        maximum = lighting_intensity_range["max"]
        if (
            not isinstance(minimum, (int, float))
            or not isinstance(maximum, (int, float))
            or not math.isfinite(float(minimum))
            or not math.isfinite(float(maximum))
            or not 0.1 <= float(minimum) <= float(maximum) <= 4.0
        ):
            raise ValueError("lighting_intensity_range must be finite and within 0.1–4.0")
        object.__setattr__(
            self,
            "lighting_intensity_range",
            MappingProxyType({"min": float(minimum), "max": float(maximum)}),
        )
        if self.occlusion_profile != "light_controlled_occlusion":
            raise ValueError("only light_controlled_occlusion is supported")
        occlusion_distribution = dict(self.occlusion_distribution) or _equal_distribution(OCCLUSION_BANDS)
        _validate_distribution(occlusion_distribution, OCCLUSION_BANDS, "occlusion_distribution")
        object.__setattr__(self, "occlusion_distribution", MappingProxyType(occlusion_distribution))
        if self.target_domain != "china_post_2000_urban_facades":
            raise ValueError("only china_post_2000_urban_facades is supported")
        if len(self.asset_paths) != len(self.asset_fingerprints):
            raise ValueError("asset_paths and asset_fingerprints must have matching lengths")
        object.__setattr__(self, "asset_paths", tuple(self.asset_paths))
        object.__setattr__(self, "asset_fingerprints", tuple(self.asset_fingerprints))

    @property
    def brief_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.value,
            "output_target": self.output_target,
            "split_ratio": dict(self.split_ratio),
            "building_use_distribution": dict(self.building_use_distribution),
            "render_width": self.render_width,
            "render_height": self.render_height,
            "seed": self.seed,
            "view_family": list(self.view_family),
            "view_distribution": dict(self.view_distribution),
            "daylight_profile": self.daylight_profile,
            "daylight_distribution": dict(self.daylight_distribution),
            "lighting_intensity_range": dict(self.lighting_intensity_range),
            "occlusion_profile": self.occlusion_profile,
            "occlusion_distribution": dict(self.occlusion_distribution),
            "target_domain": self.target_domain,
            "asset_paths": list(self.asset_paths),
            "asset_fingerprints": list(self.asset_fingerprints),
            "task_visibility_threshold": self.task_visibility_threshold,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GenerationBrief":
        data = dict(value)
        data["task"] = TaskKind(data["task"])
        data["view_family"] = tuple(data.get("view_family", ()))
        data["asset_paths"] = tuple(data.get("asset_paths", ()))
        data["asset_fingerprints"] = tuple(data.get("asset_fingerprints", ()))
        if "task_visibility_threshold" not in data and "visibility_threshold" in data:
            data["task_visibility_threshold"] = data.pop("visibility_threshold")
        return cls(**data)


@dataclass
class GenerationJob:
    id: str
    brief: GenerationBrief
    state: JobState = JobState.DRAFT
    confirmed_by: str | None = None
    confirmed_brief_hash: str | None = None
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
    queue_sequence: int = 0

    @classmethod
    def new(cls, brief: GenerationBrief) -> "GenerationJob":
        return cls(id=f"job_{uuid4().hex}", brief=brief)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "brief": self.brief.to_dict(),
            "state": self.state.value,
            "confirmed_by": self.confirmed_by,
            "confirmed_brief_hash": self.confirmed_brief_hash,
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
            "queue_sequence": self.queue_sequence,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GenerationJob":
        data = dict(value)
        data["brief"] = GenerationBrief.from_dict(data["brief"])
        data["state"] = JobState(data["state"])
        data.setdefault("queue_sequence", 0)
        data.setdefault("confirmed_brief_hash", None)
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
    sample_seeds: tuple[int, ...]
    actual_render_parameters: tuple[Mapping[str, Any], ...]
    validation_evidence: Mapping[str, Any]
    publication_decision: Mapping[str, Any]
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


def _equal_distribution(values: tuple[str, ...]) -> dict[str, float]:
    return {value: 1.0 / len(values) for value in values}
