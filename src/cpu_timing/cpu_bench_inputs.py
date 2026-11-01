"""Resolve registered benchmark inputs without importing or changing native code.

Only fixed metadata paths and (as a last resort for original suites) at most
20 immediate directories with ``seed`` in their name are considered.  This is
path selection, not complete input verification: the native loader must still
hash the selected checkpoint and all native source dependencies before use.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


PUBLIC_PACKAGE_SHA256 = "6ff6f835e9adfcb95550cbdc23706831ab0e40737fe19b2319acdec988364b86"
LEGACY_PACKAGE_SHA256 = "3aee1458499f25f50b9c6d7f15154650622075258539a5f19c8ff3105d36319c"
ACCEPTED_PACKAGE_SHA256 = {PUBLIC_PACKAGE_SHA256, LEGACY_PACKAGE_SHA256}
CANONICAL_CHECKPOINTS = {
    "strict_seed8": "6e2c3756a11ad10f0452b67a267cd8bbea917ccbcf7328f0628be74d0076b8be",
    "strict_seed42": "d4e1e43687dd2bb266116a31a1542ba7ce014dab454ed40d3b24388e58f1f32a",
    "strict_seed77": "c776757b2e8ffa2cd751f4e2d7d4f4e42127919664c24439039038fee5653d31",
    "relaxed_seed8": "2f66f9cdf322595e43ccdb3798ad63d06bdb742c82cdd86faec72641958b1434",
    "relaxed_seed42": "eea33db123316cb34f6be101135c1d0101614660bce6b9ae66a0091bc1cdde6d",
    "relaxed_seed77": "10c9fca33fd653c6bf35d46f3ad433feb530d2a28fe8aca54227701364c34693",
}
# Registered paths from original_input_reference.json; these are the exact
# 40,001,536-step checkpoints, including Strict seed42's intermediate 50M run.
_SEED42_PATHS = {
    "strict": (
        "chapter4_COST11_STRICT_BETAS_50M_v2_seed42/strict_b05/runs/"
        "ppo_progress_COST11_FULL_b0.5_seed42_to50M_v2/checkpoints/step_40001536.zip"
    ),
    "relaxed": (
        "chapter4_COST11_B6_beta05_EXISTING20M_to40M_CP5M_VAL1M_seed42/"
        "B6/continue_existing20M_to40M/runs/"
        "ppo_COST11_FULL_B6_beta05_seed42_existing20M_to40M/checkpoints/step_40001536.zip"
    ),
}
_MANIFEST_PATHS = (
    "chapter4_relaxedB6_C3NT_COST_SWEEP_seed42/cost11/manifest.csv",
    "chapter4_C3NT_COST11_STRICT_BETA_0_05_1_seed42/beta0p5/manifest.csv",
    "chapter4_5m_seed42/manifest.csv",
    "manifest.csv",
)
_SOURCE_PATHS = (
    "src/main_seeds_8_77", "src/MAIN_SEEDS_8_77", "MAIN_SEEDS_8_77",
    "main_seeds_8_77", ".", "src/original/root0/MAIN_SEEDS_8_77",
)
_MAX_METADATA_BYTES = 2 * 1024 * 1024


def _small_bytes(path: Path) -> bytes:
    with path.open("rb") as handle:
        content = handle.read(_MAX_METADATA_BYTES + 1)
    if len(content) > _MAX_METADATA_BYTES:
        raise RuntimeError(f"Metadata exceeds {_MAX_METADATA_BYTES} bytes: {path}")
    return content


def _read_json(path: Path):
    try:
        return json.loads(_small_bytes(path))
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Cannot read JSON metadata {path}: {error}") from error


def _source_package(root: Path, allow_original_fallback: bool):
    inspected = []
    seen = set()

    def probe(package):
        package = package.resolve()
        if package in seen:
            return None
        seen.add(package)
        manifest = package / "PACKAGE_SHA256.json"
        if not manifest.is_file():
            inspected.append(f"{manifest}: missing")
            return None
        try:
            digest = hashlib.sha256(_small_bytes(manifest)).hexdigest()
        except (OSError, RuntimeError) as error:
            inspected.append(f"{manifest}: {error}")
            return None
        if digest not in ACCEPTED_PACKAGE_SHA256:
            inspected.append(f"{manifest}: unregistered SHA-256 {digest}")
            return None
        return package, manifest, digest

    for relative in _SOURCE_PATHS:
        match = probe(root / relative)
        if match:
            return match
    if allow_original_fallback:
        # No recursive search; cap both directory entries examined and probes.
        candidates = []
        try:
            with os.scandir(root) as entries:
                for count, entry in enumerate(entries, 1):
                    if count > 2000:
                        inspected.append("Original-suite directory scan capped at 2,000 entries.")
                        break
                    if "seed" in entry.name.lower() and entry.is_dir(follow_symlinks=False):
                        candidates.append(Path(entry.path))
                        if len(candidates) == 20:
                            inspected.append("Original-suite fallback capped at 20 seed directories.")
                            break
        except OSError as error:
            inspected.append(f"Cannot list immediate suite directories: {error}")
        for candidate in sorted(candidates):
            match = probe(candidate)
            if match:
                return match
    raise RuntimeError(
        "No registered native source package was found under the selected root. "
        "Only the public or original PACKAGE_SHA256.json byte hash is accepted.\n"
        "Inspected candidates:\n  " + "\n  ".join(inspected)
    )


def _archive_checkpoints(root: Path):
    manifests = [root / "provenance/CHECKPOINTS.json", root / "audit/CHECKPOINTS.json"]
    available = [path for path in manifests if path.is_file()]
    if not available:
        raise RuntimeError("Archive checkpoint metadata is missing; checked: " +
                           ", ".join(str(path) for path in manifests))
    mappings = []
    for manifest in available:
        rows = _read_json(manifest)
        if not isinstance(rows, list):
            raise RuntimeError(f"CHECKPOINTS.json must contain a list: {manifest}")
        mapping = {}
        for row in rows:
            if not isinstance(row, dict) or row.get("model") not in CANONICAL_CHECKPOINTS:
                raise RuntimeError(f"Unregistered checkpoint entry in {manifest}: {row!r}")
            key = row["model"]
            if key in mapping:
                raise RuntimeError(f"Duplicate checkpoint entry {key} in {manifest}")
            expected = CANONICAL_CHECKPOINTS[key]
            # Full collector used expected_sha256; slim and public used sha256.
            hashes = [row[name] for name in ("sha256", "expected_sha256") if name in row]
            if not hashes or any(value != expected for value in hashes):
                raise RuntimeError(f"Checkpoint SHA-256 metadata disagrees for {key}: {manifest}")
            method, seed = key.split("_seed")
            relative = row.get("package_path") or f"models/{method}/seed{seed}/final_40001536.zip"
            if not isinstance(relative, str) or Path(relative).is_absolute():
                raise RuntimeError(f"Invalid archive checkpoint path for {key}: {relative!r}")
            selected = (root / relative).resolve()
            if not selected.is_relative_to(root):
                raise RuntimeError(f"Archive checkpoint path escapes its root: {relative}")
            mapping[key] = {"path": str(selected), "sha256": expected}
        missing = sorted(CANONICAL_CHECKPOINTS.keys() - mapping.keys())
        if missing:
            raise RuntimeError(f"Checkpoint entries missing in {manifest}: {', '.join(missing)}")
        mappings.append(mapping)
    if any(mapping != mappings[0] for mapping in mappings[1:]):
        raise RuntimeError("provenance/CHECKPOINTS.json and audit/CHECKPOINTS.json disagree; "
                           "resolve their recorded model/path/hash difference before benchmarking.")
    return available[0], mappings[0]


def _relocate_receipt_path(root: Path, receipt: Path, value: str) -> Path:
    path = Path(value).expanduser()
    parts = path.parts
    if "uav_chapter4_suite_v2" in parts:
        return (root / Path(*parts[parts.index("uav_chapter4_suite_v2") + 1:])).resolve()
    anchor = "chapter4_MAIN_COST11_40M_seed8_77"
    if anchor in parts:
        return (root / Path(*parts[parts.index(anchor):])).resolve()
    if path.is_absolute():
        return path.resolve()
    return (receipt.parent / path).resolve()


def _original_checkpoints(root: Path):
    checkpoints = {}
    for key, expected in CANONICAL_CHECKPOINTS.items():
        method, seed = key.split("_seed")
        if seed == "42":
            selected = root / _SEED42_PATHS[method]
        else:
            job = root / "chapter4_MAIN_COST11_40M_seed8_77" / key
            selected = job / (f"stages/stage40M/runs/ppo_{method}_C3NT_COST11_FULL_H5_"
                              f"b0.5_seed{seed}__continue_to_40000000/last_model.zip")
            receipt = job / "training_result.json"
            if receipt.is_file():
                record = _read_json(receipt)
                if not isinstance(record, dict):
                    raise RuntimeError(f"training_result.json must contain an object: {receipt}")
                recorded_sha = record.get("final_model_sha256")
                if recorded_sha is not None and recorded_sha != expected:
                    raise RuntimeError(f"Registered checkpoint SHA-256 disagrees with {receipt}")
                final_model = record.get("final_model")
                if final_model:
                    if not isinstance(final_model, str):
                        raise RuntimeError(f"final_model must be a path string: {receipt}")
                    selected = _relocate_receipt_path(root, receipt, final_model)
        checkpoints[key] = {"path": str(selected.resolve()), "sha256": expected}
    return checkpoints


def resolve_inputs(root: str | Path, input_mode: str = "auto") -> dict:
    """Return absolute paths and registered hashes; do not read model weights.

    ``auto`` selects archive mode when data/manifest_portable.csv exists.
    ``original`` explicitly selects the original server suite, even if that
    archive marker is present.  Missing model files are returned as expected
    paths so only the models requested by the caller need to be available.
    """
    if input_mode not in {"auto", "archive", "original"}:
        raise ValueError("input_mode must be 'auto', 'archive', or 'original'")
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(f"Input root is not a directory: {root}")
    portable = root / "data/manifest_portable.csv"
    archive = input_mode == "archive" or (input_mode == "auto" and portable.is_file())
    package, source_manifest, source_sha = _source_package(root, not archive)
    if archive:
        if not portable.is_file():
            raise RuntimeError(f"Archive map manifest is missing: {portable}")
        checkpoint_manifest, checkpoints = _archive_checkpoints(root)
        map_manifest = portable
        input_kind = "public_archive" if source_sha == PUBLIC_PACKAGE_SHA256 else "legacy_archive"
        description = (f"{input_kind}: registered native package {package}; portable map manifest; "
                       f"checkpoint registrations from {checkpoint_manifest}")
    else:
        checkpoint_manifest = None
        checkpoints = _original_checkpoints(root)
        map_manifest = next((root / item for item in _MANIFEST_PATHS if (root / item).is_file()), None)
        input_kind = "server_original"
        description = (f"Original server suite: registered native package {package}; "
                       "six registered checkpoints selected from fixed paths or training receipts")
    return {
        "root": str(root), "input_kind": input_kind,
        "source_package": str(package), "source_manifest": str(source_manifest),
        "source_manifest_sha256": source_sha,
        "map_manifest": str(map_manifest.resolve()) if map_manifest else None,
        "map_mode": "archive" if archive else "original",
        "checkpoint_manifest": str(checkpoint_manifest.resolve()) if checkpoint_manifest else None,
        "checkpoints": checkpoints, "description": description,
    }
