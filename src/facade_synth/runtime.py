"""Fail-closed BlenderProc runtime evidence and preflight helpers."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


class RuntimeGateError(RuntimeError):
    """Raised when output cannot prove a complete BlenderProc/Blender render."""


class EnvironmentNotReady(RuntimeGateError):
    """Raised when the local BlenderProc runtime cannot initialize."""


def validate_render_summary(summary: Mapping[str, Any], *, expected_count: int) -> None:
    """Accept only complete, real BlenderProc/Blender output."""

    if summary.get("used_projection_fallback") is not False:
        raise RuntimeGateError("projection fallback is not permitted")
    if summary.get("projection_fallback_count") != 0:
        raise RuntimeGateError("projection fallback count must be zero")
    if summary.get("render_backend") != "blenderproc_blender":
        raise RuntimeGateError("render backend is not blenderproc_blender")
    if summary.get("sample_count") != expected_count:
        raise RuntimeGateError("rendered sample count does not match confirmed output_target")
    if summary.get("rendered_with_blender_count") != expected_count:
        raise RuntimeGateError("not every sample was rendered with Blender")


@dataclass(frozen=True)
class BlenderProcRuntime:
    """The only process boundary permitted to create a Trainable Package."""

    executable: str = "blenderproc"
    entrypoint: Path = Path(__file__).with_name("blenderproc_entry.py")

    def preflight(self) -> dict[str, Any]:
        result = self._run(("run", str(self.entrypoint), "--preflight"))
        try:
            evidence = _last_json_object(result.stdout)
        except ValueError as exc:
            raise EnvironmentNotReady("BlenderProc preflight produced no readable evidence") from exc
        if evidence.get("runtime") != "blenderproc_blender":
            raise EnvironmentNotReady("BlenderProc preflight did not confirm Blender runtime")
        return evidence

    def run_generator(self, arguments: Sequence[str]) -> dict[str, Any]:
        result = self._run(("run", str(self.entrypoint), "--", *arguments))
        try:
            return _last_json_object(result.stdout)
        except ValueError as exc:
            raise RuntimeGateError("BlenderProc generator produced no readable summary") from exc

    def _run(self, arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
        try:
            result = subprocess.run(
                (self.executable, *arguments),
                text=True,
                capture_output=True,
                check=False,
            )
        except FileNotFoundError as exc:
            raise EnvironmentNotReady(f"BlenderProc executable not found: {self.executable}") from exc
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
            raise EnvironmentNotReady(f"BlenderProc runtime is not ready: {detail}")
        return result


def _last_json_object(output: str) -> dict[str, Any]:
    for line in reversed(output.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            return value
    raise ValueError("no JSON object found")
