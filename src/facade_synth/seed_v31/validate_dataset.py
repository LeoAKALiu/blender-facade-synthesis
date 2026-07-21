from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from facade_synth.seed_v31.schema import (
    LABEL_PATH_KEYS,
    ValidationError,
    load_metadata,
    validate_dataset_path,
    validate_metadata,
)


class DatasetValidationError(RuntimeError):
    pass


MANIFEST_ENTRY_KEYS = {"sample_id", "metadata_path"}


def validate_dataset(root: Path | str) -> dict[str, Any]:
    root_path = Path(root)
    manifest_path = root_path / "manifest.jsonl"
    if not manifest_path.exists():
        raise DatasetValidationError(f"missing manifest.jsonl: {manifest_path}")

    errors: list[str] = []
    sample_count = 0
    try:
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DatasetValidationError(f"cannot read manifest.jsonl: {exc}") from exc

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            errors.append(f"manifest line {line_number}: blank line")
            continue
        sample_count += 1
        try:
            _validate_manifest_line(root_path, line)
        except DatasetValidationError as exc:
            errors.append(f"manifest line {line_number}: {exc}")

    if errors:
        raise DatasetValidationError("\n".join(errors))

    return {"root": str(root_path), "sample_count": sample_count, "errors": []}


def _validate_manifest_line(root: Path, line: str) -> None:
    entry = parse_manifest_entry(line)

    metadata_relpath = entry["metadata_path"]
    metadata_path = root / metadata_relpath
    try:
        metadata = load_metadata(metadata_path)
        validate_metadata(metadata)
    except ValidationError as exc:
        raise DatasetValidationError(f"metadata_path {metadata_relpath}: {exc}") from exc
    except OSError as exc:
        raise DatasetValidationError(f"metadata_path {metadata_relpath}: cannot read file: {exc}") from exc

    metadata_sample_id = metadata.get("sample_id")
    if entry["sample_id"] != metadata_sample_id:
        raise DatasetValidationError(
            f"sample_id {entry['sample_id']!r} does not match metadata sample_id {metadata_sample_id!r}"
        )

    _validate_sample_files(root, metadata)


def parse_manifest_entry(line: str) -> dict[str, str]:
    try:
        entry = json.loads(
            line,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except json.JSONDecodeError as exc:
        raise DatasetValidationError(f"invalid JSON: {exc.msg}") from exc
    except ValueError as exc:
        raise DatasetValidationError(str(exc)) from exc

    if not isinstance(entry, dict):
        raise DatasetValidationError("manifest entry must be an object")

    extra_keys = sorted(set(entry) - MANIFEST_ENTRY_KEYS)
    if extra_keys:
        raise DatasetValidationError(f"unknown manifest key(s): {', '.join(extra_keys)}")

    sample_id = entry.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise DatasetValidationError("sample_id must be a non-empty string")

    try:
        metadata_relpath = validate_dataset_path(entry.get("metadata_path"), key="metadata_path")
    except ValidationError as exc:
        raise DatasetValidationError(str(exc)) from exc

    return {"sample_id": sample_id, "metadata_path": metadata_relpath}


def _validate_sample_files(root: Path, metadata: dict[str, Any]) -> None:
    width = int(metadata["image"]["width"])
    height = int(metadata["image"]["height"])

    rgb = _validate_image(root, metadata["image"]["rgb_path"], width, height, "rgb_path")
    if rgb.shape != (height, width, 3):
        raise DatasetValidationError(
            f"rgb_path {metadata['image']['rgb_path']}: shape {rgb.shape} must be {(height, width, 3)}"
        )

    labels = metadata["labels"]
    for key in LABEL_PATH_KEYS:
        if key in ("depth_path", "normal_path"):
            continue
        array = _validate_image(root, labels[key], width, height, key)
        if array.ndim != 2:
            raise DatasetValidationError(f"{key} {labels[key]}: shape {array.shape} must be 2D")
        if key == "window_instance_mask_path":
            _validate_window_instance_ids(array, metadata)

    _validate_depth(root, labels["depth_path"], width, height)
    _validate_normal(root, labels["normal_path"], width, height)


def _validate_image(root: Path, relpath: str, width: int, height: int, label: str) -> np.ndarray:
    path = root / relpath
    if not path.exists():
        raise DatasetValidationError(f"{label} {relpath}: missing file")
    try:
        with Image.open(path) as image:
            image.load()
            if image.size != (width, height):
                raise DatasetValidationError(
                    f"{label} {relpath}: shape {image.size[1]}x{image.size[0]} does not match metadata {height}x{width}"
                )
            return np.asarray(image)
    except DatasetValidationError:
        raise
    except (OSError, UnidentifiedImageError) as exc:
        raise DatasetValidationError(f"{label} {relpath}: cannot read image: {exc}") from exc


def _validate_depth(root: Path, relpath: str, width: int, height: int) -> None:
    array = _load_npy(root, relpath, "depth_path")
    if array.shape != (height, width):
        raise DatasetValidationError(f"depth_path {relpath}: shape {array.shape} must be {(height, width)}")
    _validate_finite_numeric(array, relpath, "depth_path")


def _validate_normal(root: Path, relpath: str, width: int, height: int) -> None:
    array = _load_npy(root, relpath, "normal_path")
    if array.shape != (height, width, 3):
        raise DatasetValidationError(f"normal_path {relpath}: shape {array.shape} must be {(height, width, 3)}")
    _validate_finite_numeric(array, relpath, "normal_path")


def _load_npy(root: Path, relpath: str, label: str) -> np.ndarray:
    path = root / relpath
    if not path.exists():
        raise DatasetValidationError(f"{label} {relpath}: missing file")
    try:
        return np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise DatasetValidationError(f"{label} {relpath}: cannot read array: {exc}") from exc


def _validate_finite_numeric(array: np.ndarray, relpath: str, label: str) -> None:
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.complexfloating):
        raise DatasetValidationError(f"{label} {relpath}: array must be real numeric")
    if not np.isfinite(array).all():
        raise DatasetValidationError(f"{label} {relpath}: array must contain only finite values")


def _validate_window_instance_ids(mask: np.ndarray, metadata: dict[str, Any]) -> None:
    if not np.issubdtype(mask.dtype, np.integer):
        raise DatasetValidationError("window_instance_mask_path must contain integer-like label data")
    actual_ids = {int(value) for value in np.unique(mask) if int(value) != 0}
    declared_ids = {int(instance["id"]) for instance in metadata["windows"]["instances"]}
    if actual_ids != declared_ids:
        raise DatasetValidationError(
            "window_instance_mask_path IDs do not match declared windows.instances: "
            f"actual={sorted(actual_ids)} declared={sorted(declared_ids)}"
        )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"unsupported JSON constant: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a synthetic facade MVP dataset.")
    parser.add_argument("root", type=Path)
    args = parser.parse_args(argv)

    try:
        summary = validate_dataset(args.root)
    except DatasetValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
