"""Load a verified native PPO evaluator for CPU timing without editing it.

Call ``load_native_model`` once in a fresh worker process.  The caller configures
CPU affinity, thread environment variables, and CUDA_VISIBLE_DEVICES before
importing this module.  Imports here are standard-library-only until the loader
is called; importing this module does not parse arguments or load a checkpoint.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
from types import SimpleNamespace


REFERENCE_COMMIT = "8e19d8822983245b811b83e88c99fd2be548a044"
# Both packages were recovered and compared directly.  Their rollout,
# initialization and policy-action functions are byte-identical.  The original
# test8 module and its public copy differ only in five unused path strings.
_SOURCE_MANIFEST_FLAVORS = {
    "6ff6f835e9adfcb95550cbdc23706831ab0e40737fe19b2319acdec988364b86":
        "public_archive_path_relocation",
    "3aee1458499f25f50b9c6d7f15154650622075258539a5f19c8ff3105d36319c":
        "original_main_seeds_8_77",
}
_PACKAGES = ("numpy", "torch", "gymnasium", "stable-baselines3", "sb3-contrib")
_NATIVE_NAMES = (
    "coverage_core", "core_ch4", "core_ch4_bounded_b6", "full_v2_utils",
    "c3nt_cost_sweep_generated_core", "c3nt_cost_sweep_generated_logging",
    "nrg1_c3nt_cost5_features", "trainers_ch4_c3nt_cost_sweep", "resume_ch4",
    "seed_evaluation", "ch5_common", "ch5_vendor",
    "ch5_vendor.test8_original", "ch5_vendor.segment_graph_original",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scoped_file(root: Path, relative: str) -> Path:
    relative_path = Path(relative)
    _require(not relative_path.is_absolute(), f"Expected a path relative to {root}: {relative}")
    path = (root / relative_path).resolve()
    _require(path.is_relative_to(root), f"Path escapes the selected input scope: {relative}")
    _require(path.is_file(), f"Required input file is missing: {path}")
    return path


def _check_hash(path: Path, expected: str) -> str:
    actual = _sha256(path)
    _require(
        actual == expected,
        f"File differs from the reference archive: {path}\n"
        f"Expected SHA-256: {expected}\nActual SHA-256:   {actual}",
    )
    return actual


def load_native_model(
    repo_root: str,
    method: str,
    seed: int,
    threads: int,
    allow_version_difference: bool = False,
    input_mode: str = "auto",
) -> SimpleNamespace:
    """Return ``rt, V, C, model, native, metadata`` for one verified CPU model.

    Native sources match one of two recovered package-manifest anchors;
    checkpoint bytes always match the same six registered final-model hashes.
    ``allow_version_difference`` permits a different *benchmark software*
    environment, records every dependency difference, and leaves the original
    template on disk untouched.  It never disables source/checkpoint/model
    audits.  Setup and audit times are metadata, not route-planning runtimes.
    """
    started = time.perf_counter()
    _require(sys.version_info >= (3, 10), "Python 3.10 or later is required.")
    _require(method in ("strict", "relaxed"), "method must be 'strict' or 'relaxed'.")
    _require(isinstance(seed, int) and not isinstance(seed, bool) and seed in (8, 42, 77),
             "seed must be one of 8, 42, or 77.")
    _require(isinstance(threads, int) and not isinstance(threads, bool) and threads > 0,
             "threads must be a positive integer.")
    _require(isinstance(allow_version_difference, bool),
             "allow_version_difference must be a Boolean.")
    _require(isinstance(input_mode, str), "input_mode must be a string.")
    loaded = [name for name in _NATIVE_NAMES if name in sys.modules]
    _require(not loaded,
             "Use a fresh worker for each model; native modules already loaded: " + ", ".join(loaded))

    from cpu_bench_inputs import CANONICAL_CHECKPOINTS, resolve_inputs
    resolved = resolve_inputs(repo_root, input_mode=input_mode)
    root = Path(resolved["root"]).expanduser().resolve()
    _require(root.is_dir(), f"Input directory is missing: {root}")
    package = Path(resolved["source_package"]).resolve()
    _require(package.is_dir(), f"Native source package is missing: {package}")
    package_manifest_path = Path(resolved["source_manifest"]).resolve()
    _require(package_manifest_path.is_relative_to(package) and package_manifest_path.is_file(),
             "Source manifest must be a file inside the resolved native package.")
    source_manifest_sha256 = _sha256(package_manifest_path)
    _require(source_manifest_sha256 in _SOURCE_MANIFEST_FLAVORS,
             f"Unrecognized native package manifest: {package_manifest_path}\n"
             f"SHA-256: {source_manifest_sha256}")
    _require(resolved["source_manifest_sha256"] == source_manifest_sha256,
             "Source manifest changed after input resolution.")
    source_flavor = _SOURCE_MANIFEST_FLAVORS[source_manifest_sha256]
    package_hashes = json.loads(package_manifest_path.read_text(encoding="utf-8"))
    _require(isinstance(package_hashes, dict), "Native package manifest must be a filename/hash mapping.")
    template_relative = f"templates/{method}.json"
    template_path = _scoped_file(package, template_relative)
    _require(template_relative in package_hashes, "Template is absent from the source manifest.")
    _check_hash(template_path, package_hashes[template_relative])
    template = json.loads(template_path.read_text(encoding="utf-8"))
    _require(template["method"] == method, "Native template method differs.")

    # Audit imported wrappers and all native snapshot files before importing them.
    wrapper_files = (
        "seed_evaluation.py", "ch5_common.py", "ch5_vendor/__init__.py",
        "ch5_vendor/test8_original.py", "ch5_vendor/segment_graph_original.py",
        template_relative,
    )
    source_hashes = {}
    for relative in wrapper_files:
        _require(relative in package_hashes, f"Missing source-manifest entry: {relative}")
        path = _scoped_file(package, relative)
        source_hashes[relative] = _check_hash(path, package_hashes[relative])
    runtime_hashes = template["runtime_hashes"]
    for name, expected in runtime_hashes.items():
        _require(Path(name).name == name and name.endswith(".py"),
                 f"Invalid native snapshot filename: {name}")
        relative = f"runtime/{method}/{name}"
        _require(package_hashes.get(relative) == expected,
                 f"Native template/source-manifest hash disagreement: {relative}")
        path = _scoped_file(package, relative)
        source_hashes[relative] = _check_hash(path, expected)

    model_key = f"{method}_seed{seed}"
    _require(model_key in resolved["checkpoints"], f"No registered final checkpoint for {model_key}.")
    checkpoint_entry = resolved["checkpoints"][model_key]
    # resolve_inputs authenticates this hash against its fixed six-model
    # registry, independently of any local checkpoint manifest.  The chosen
    # model's actual bytes are always rechecked here before loading.
    canonical_checkpoint_hash = checkpoint_entry["sha256"]
    _require(canonical_checkpoint_hash == CANONICAL_CHECKPOINTS[model_key],
             f"Resolved checkpoint hash differs from the fixed registry: {model_key}")
    checkpoint_path = Path(checkpoint_entry["path"]).resolve()
    _require(checkpoint_path.is_file(), f"Registered checkpoint is missing: {checkpoint_path}")
    checkpoint_hash = _check_hash(checkpoint_path, canonical_checkpoint_hash)
    checkpoint_manifest_path = None
    checkpoint_manifest_sha256 = None
    if resolved.get("checkpoint_manifest") is not None:
        checkpoint_manifest_path = Path(resolved["checkpoint_manifest"]).resolve()
        _require(checkpoint_manifest_path.is_relative_to(root) and checkpoint_manifest_path.is_file(),
                 "Checkpoint manifest must be a file inside the resolved input root.")
        checkpoint_manifest_sha256 = _sha256(checkpoint_manifest_path)
        checkpoint_rows = json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
        _require(isinstance(checkpoint_rows, list) and all(isinstance(row, dict) for row in checkpoint_rows),
                 "Checkpoint manifest must contain a list of model records.")
        matching = [row for row in checkpoint_rows if row.get("model") == model_key]
        _require(len(matching) == 1, f"Expected one manifest checkpoint record for {model_key}.")
        manifest_entry = matching[0]
        manifest_hashes = [manifest_entry[name] for name in ("sha256", "expected_sha256")
                           if name in manifest_entry]
        _require(manifest_hashes and all(value == canonical_checkpoint_hash for value in manifest_hashes),
                 f"Checkpoint manifest disagrees with the registered final-model hash: {model_key}")
        # The full collector leaves package_path empty for missing models.
        # Resolve that record only to its registered archive location; actual
        # existence and the canonical file hash were already checked above.
        manifest_relative = (manifest_entry.get("package_path")
                             or f"models/{method}/seed{seed}/final_40001536.zip")
        _require(isinstance(manifest_relative, str),
                 f"Checkpoint manifest has an invalid package-relative model path: {model_key}")
        manifest_checkpoint_path = _scoped_file(root, manifest_relative)
        _require(manifest_checkpoint_path == checkpoint_path,
                 f"Checkpoint manifest and resolved model paths disagree: {model_key}")

    actual_versions = {}
    for name in _PACKAGES:
        try:
            actual_versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"Required package is not installed: {name}") from exc
    original_software = copy.deepcopy(template["software"])
    differences = {
        name: {"reference": original_software.get(name), "benchmark": actual_versions[name]}
        for name in _PACKAGES
        if original_software.get(name) != actual_versions[name]
    }
    if differences and not allow_version_difference:
        detail = "; ".join(
            f"{name}: expected {value['reference']}, found {value['benchmark']}"
            for name, value in differences.items()
        )
        raise RuntimeError(
            "Benchmark dependency versions differ from the original environment. " + detail
            + ". Use the original environment or explicitly enable --allow-version-difference "
              "to record a benchmark under the installed software."
        )

    spec = copy.deepcopy(template["spec"])
    spec["seed"] = seed
    settings = copy.deepcopy(template["settings"])
    settings["torch_threads"] = threads
    settings["device"] = "cpu"
    expected_mask = "progress" if method == "strict" else "relaxed"
    _require(spec["environment"]["mask_mode"] == expected_mask,
             "Wrong native progress-mask setting.")
    _require(spec["environment"]["detour_budget_max"] == 6,
             "Keep the archived budget-observation capacity at 6, including Strict.")
    benchmark_software = copy.deepcopy(original_software)
    if allow_version_difference:
        benchmark_software.update(actual_versions)
    native = dict(
        key=model_key, label=f"{method.title()} seed {seed} 40M",
        kind="ppo", family=method, runtime_kind=method, beta=0.5,
        B=0 if method == "strict" else 6, nominal_m=40, seed=seed,
        native_version=template["native_version"],
        snapshot=str((package / "runtime" / method).resolve()),
        sources={name: {"sha256": digest} for name, digest in runtime_hashes.items()},
        spec=spec, settings=settings,
        checkpoint=dict(path=str(checkpoint_path), sha256=checkpoint_hash, num_timesteps=40_001_536),
        software=benchmark_software,
    )

    import torch
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError as exc:
        raise RuntimeError(
            "Cannot set PyTorch inter-op threads before native loading. "
            "Start a fresh benchmark worker before performing any Torch work."
        ) from exc
    _require(torch.get_num_threads() == threads and torch.get_num_interop_threads() == 1,
             "PyTorch thread settings were not applied.")

    sys.path.insert(0, str(package))
    previous_bytecode_setting = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        E = importlib.import_module("seed_evaluation")
        C = importlib.import_module("ch5_common")
        V = importlib.import_module("ch5_vendor.test8_original")
        A = importlib.import_module("ch5_vendor.segment_graph_original")
        for module, relative in (
            (E, "seed_evaluation.py"), (C, "ch5_common.py"),
            (V, "ch5_vendor/test8_original.py"),
            (A, "ch5_vendor/segment_graph_original.py"),
        ):
            _require(Path(module.__file__).resolve() == (package / relative).resolve(),
                     f"Imported module came from another checkout: {module.__name__}")
        rt = E._load_runtime(native)
        model_started = time.perf_counter()
        model = E._load_model(rt, native)
        model_load_seconds = time.perf_counter() - model_started
    finally:
        sys.dont_write_bytecode = previous_bytecode_setting
        sys.path.remove(str(package))

    _require(str(model.device) == "cpu", f"Expected a CPU model, found {model.device}.")
    non_cpu = [name for name, tensor in list(model.policy.named_parameters())
               + list(model.policy.named_buffers()) if tensor.device.type != "cpu"]
    _require(not non_cpu, "Non-CPU model tensors found: " + ", ".join(non_cpu))
    _require(int(model.action_space.n) == 4, "Expected four absolute-direction actions.")

    metadata = {
        "reference_commit": REFERENCE_COMMIT,
        "repo_root": str(root),
        "input_mode_requested": input_mode,
        "input_kind": resolved["input_kind"],
        "input_description": resolved["description"],
        "source_package": str(package),
        "source_manifest": str(package_manifest_path),
        "source_flavor": source_flavor,
        "method": method,
        "training_seed": seed,
        "checkpoint": copy.deepcopy(native["checkpoint"]),
        "checkpoint_sha256": checkpoint_hash,
        "native_version": native["native_version"],
        "source_hashes": source_hashes,
        "source_hash_scope": "paths relative to source_package",
        "source_manifest_sha256": source_manifest_sha256,
        "checkpoint_manifest": str(checkpoint_manifest_path) if checkpoint_manifest_path else None,
        "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
        "checkpoint_identity_basis": "fixed six-model registry plus selected checkpoint byte SHA-256",
        "checkpoint_manifest_agreement": True if checkpoint_manifest_path is not None else None,
        "loader_sha256": _sha256(Path(__file__).resolve()),
        "original_template_software": original_software,
        "benchmark_software": dict(actual_versions, python=sys.version,
                                   platform=platform.platform(), torch_runtime_version=str(torch.__version__)),
        "software_version_differences": differences,
        "allow_version_difference": allow_version_difference,
        "software_matches_original": not differences,
        "version_comparison": "exact installed versions for numpy/torch/gymnasium/stable-baselines3/sb3-contrib",
        "device": "cpu",
        "all_model_tensors_on_cpu": True,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "model_load_and_audit_wall_sec": model_load_seconds,
        "model_setup_wall_sec": time.perf_counter() - started,
        "setup_timing_scope": "source/checkpoint hashes, dependency imports, native loading and model audits; excluded from route timings",
        "template_spec_seed": template["spec"]["seed"],
        "template_use": "shared method environment/architecture; model identity audited against the requested seed and checkpoint",
    }
    return SimpleNamespace(rt=rt, V=V, C=C, model=model, native=native, metadata=metadata)
