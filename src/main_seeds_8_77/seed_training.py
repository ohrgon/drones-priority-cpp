"""Fresh paired-seed training with the authenticated native COST11 engine.

Each job starts from independent weights at 0 -> 5M.  The two later stages
load only that job's preceding checkpoint, using the original reviewed
continuation implementation for policy/Adam and balanced-sampler recovery.
No seed-42 checkpoint is ever passed to a model loader by this module.
"""
from __future__ import annotations

import ast
from collections import Counter
from copy import deepcopy
import importlib.metadata
import importlib.util
import math
from pathlib import Path
import zipfile
import json

from vendor import continuation_reference as C

VERSION = "paired_seed8_77_native_staged40m_v1"
TARGETS = (5_000_000, 20_000_000, 40_000_000)


def _need(ok, message):
    if not ok:
        raise RuntimeError(message)


def _own(path, root):
    p, r = Path(path).resolve(), Path(root).resolve()
    _need(r in p.parents and not Path(path).is_symlink(),
          "Expected a regular file inside this seed's job: " + str(path))
    return p


def _version_check(job):
    expected = job.get("software", {})
    got = {}
    for name in ("numpy", "torch", "gymnasium", "stable-baselines3", "sb3-contrib"):
        try:
            got[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"Original training dependency is missing: {name}. Use the original training environment.") from exc
        _need(name in expected and got[name] == expected[name],
              f"Original {name}={expected.get(name)!r}; current={got[name]!r}. Use the original environment.")
    import torch
    device = str(job["device"])
    if device.startswith("cuda"):
        _need(torch.cuda.is_available(), "Requested CUDA training device is unavailable.")
        index = torch.device(device).index
        index = 0 if index is None else index
        _need(index < torch.cuda.device_count(), "Requested CUDA index is unavailable.")
        if expected.get("cuda_version") is not None:
            _need(torch.version.cuda == expected["cuda_version"], "Original CUDA runtime changed.")
    if job["settings"].get("tensorboard"):
        _need(importlib.util.find_spec("tensorboard") is not None, "TensorBoard dependency is missing.")
    return got


def _verified_runtime(job):
    import numpy as np

    seed, method = job["seed"], job["method"]
    _need(type(seed) is int and seed in (8, 77), "Only the agreed seeds 8 and 77 are allowed.")
    _need(method in ("strict", "relaxed"), "Only main Strict and Relaxed jobs are allowed.")
    spec, settings = job["spec"], job["settings"]
    mask = "progress" if method == "strict" else "relaxed"
    _need(spec["seed"] == seed and spec["algorithm"] == "ppo", "Job/model seed or algorithm mismatch.")
    _need(spec["environment"]["mask_mode"] == mask and spec["beta"] == .5,
          "Job is not the requested main mask/beta condition.")
    _need(settings["total_timesteps"] == TARGETS[0] and settings["num_envs"] == 8,
          "The independent first stage must retain the original 5M/eight-environment setup.")
    p = settings["ppo"]
    _need(p["n_steps"] == 1024 and p["learning_rate_initial"] == 2.5e-4
          and p["learning_rate_floor"] == 5e-5, "Original first-stage learning-rate/rollout settings changed.")
    versions = _version_check(job)
    snapshot = Path(job["snapshot"]).resolve()
    hashes = job["runtime_hashes"]
    _need(hashes.get(C.TRAINER + ".py") == C.TRAINER_SHA256,
          "Unreviewed native trainer version.")
    for name, expected in hashes.items():
        source = _own(snapshot / name, snapshot)
        _need(source.is_file() and C.sha(source) == expected, "Frozen runtime changed: " + name)
    rt = C.import_runtime(snapshot, hashes)
    _need(rt.core.VERSION == C.CORE_VERSIONS[mask], "Wrong native Strict/Relaxed reward core.")
    records = rt.utils.load_manifest(job["manifest"], 50, verify_files=False)
    counts = Counter((r.source, r.split) for r in records)
    _need(counts == {("synthetic", "train"): 900, ("synthetic", "validation"): 180,
                     ("synthetic", "test"): 720, ("osm", "test"): 500},
          "The frozen Train900/Validation180/SyntheticTest720/OSM500 split changed.")
    train = [r for r in records if r.source == "synthetic" and r.split == "train"]
    from coverage_core import validate_priority_map
    for rec in train:
        a = validate_priority_map(np.load(rec.path, allow_pickle=False), 50, min_active=2)
        _need(rt.utils.array_hash(a) == rec.sha256, "Training map changed: " + rec.map_id)
    report = dict(state="verified", seed=seed, method=method, versions=versions,
                  counts={f"{s}/{p}": n for (s, p), n in counts.items()},
                  verified_training_arrays=len(train), optimizer_updates=0,
                  native_trainer_sha256=C.TRAINER_SHA256, native_core_version=rt.core.VERSION,
                  manifest_sha256=C.sha(job["manifest"]))
    return rt, train, report


def verify_job(job: dict) -> dict:
    """Check the original runtime and all 900 Train arrays; perform no learning."""
    _, _, report = _verified_runtime(job)
    return report


def _restore_native_train_function(rt):
    """Remove an in-memory continuation patch without changing snapshot bytes."""
    path = Path(rt.trainer.__file__)
    source = path.read_text(encoding="utf-8-sig")
    _need(C.sha(path) == C.TRAINER_SHA256, "Native trainer source changed.")
    functions = [n for n in ast.parse(source).body
                 if isinstance(n, ast.FunctionDef) and n.name == "train_sb3"]
    _need(len(functions) == 1, "Native train_sb3 definition missing.")
    original = ast.get_source_segment(source, functions[0])
    exec(compile(original + "\n", str(path), "exec"), rt.trainer.__dict__)


def _completed_stage(stage, rt, records, seed):
    run = Path(stage["run_dir"])
    status = C.read_json(run / "status.json")
    spec = C.read_json(run / "run_spec.json")
    settings = C.read_json(run / "training_settings.json")
    _need(status.get("state") == "completed", "Previous stage is not complete: " + str(run))
    _need(spec == stage["spec"] and settings == stage["settings"] and spec["seed"] == seed,
          "Stage seed/spec/settings mismatch.")
    _need(status["identity"] == rt.trainer.run_identity(spec, settings, records),
          "Stage training identity mismatch.")
    path = run / "last_model.zip"
    final = [x for x in C.read_json(run / "checkpoints.json")
             if x.get("final") and Path(x["path"]).resolve() == path.resolve()]
    _need(len(final) == 1 and C.sha(path) == final[0]["sha256"], "Final checkpoint registry/hash mismatch.")
    with zipfile.ZipFile(path) as archive:
        _need(archive.getinfo("data").file_size < 32_000_000, "Oversized model metadata.")
        data = json.loads(archive.read("data"))
        _need(archive.read("_stable_baselines3_version").decode().strip() == "2.7.0",
              "Checkpoint SB3 version changed.")
    target = int(stage["nominal_target"])
    rollout = settings["num_envs"] * settings["ppo"]["n_steps"]
    actual = math.ceil(target / rollout) * rollout
    _need(status["actual_timesteps"] == data.get("num_timesteps") == final[0]["num_timesteps"] == actual,
          "Stage target/actual timestep mismatch.")
    _need(data.get("seed") == seed and data.get("n_envs") == settings["num_envs"]
          and data.get("_total_timesteps") == target
          and data.get("_n_updates") == status["ppo_epoch_counter"],
          "Checkpoint seed, rollout denominator, or optimizer counter mismatch.")
    for key in ("n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", "ent_coef",
                "vf_coef", "max_grad_norm", "target_kl"):
        _need(data.get(key) == settings["ppo"][key], "Checkpoint PPO setting mismatch: " + key)
    return dict(**stage, status=status, model_path=str(path.resolve()),
                model_sha256=final[0]["sha256"])


def _merged_sampler_logs(rt, records, previous, destination, settings, seed):
    """Join completed stage suffixes, then use the ORIGINAL sampler replay."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    provenance = []
    for rank in range(settings["num_envs"]):
        merged = []
        for stage in previous:
            path = Path(stage["run_dir"]) / f"map_order_rank{rank}.csv"
            suffix = C.rows(path)
            _need(bool(suffix), "Empty completed-stage sampler log: " + str(path))
            for row in suffix:
                expected = len(merged) + 1
                _need(int(row["rank"]) == rank and int(row["episode"]) == expected
                      and int(row["sequence_position"]) == expected,
                      "Stage sampler logs overlap or contain a gap: " + str(path))
                merged.append(row)
            provenance.append(dict(rank=rank, stage_target=stage["nominal_target"],
                                   path=str(path), sha256=C.sha(path), draws=len(suffix)))
        target = destination / f"map_order_rank{rank}.csv"
        if target.exists():
            _need(C.rows(target) == merged, "Frozen cumulative sampler history changed.")
        else:
            C.write_csv(target, merged)
    orders, evidence = C.rebuild_orders(rt, records, destination, settings, seed)
    # Every stage discards its predecessor's eight unfinished episodes and starts
    # eight new ones. These discarded prefixes are NOT completed/failure episodes.
    expected_draws = previous[-1]["status"]["episodes"] + len(previous) * settings["num_envs"]
    _need(sum(x["episode"] for x in orders) == expected_draws,
          "Completed-episode counts do not match cumulative sampler draws/stage boundaries.")
    C.freeze(destination / "audit.json", dict(seed=seed, completed_stages=len(previous),
             expected_total_draws=expected_draws, ranks=evidence, original_stage_logs=provenance))
    return orders


def _freeze_stage(directory, spec, settings, job, target):
    path = Path(directory) / "stage_protocol.json"
    C.freeze(path, dict(version=VERSION, method=job["method"], seed=job["seed"],
                        nominal_target=target, spec=spec, settings=settings,
                        manifest_sha256=C.sha(job["manifest"]), runtime_hashes=job["runtime_hashes"]))
    sp = deepcopy(spec)
    sp["nrg1_protocol_sha256"] = C.sha(path)
    if "continuation" in sp:
        sp["continuation_protocol_sha256"] = C.sha(path)
    return sp


def train_job(job: dict) -> dict:
    """Train 0->5M->20M->40M; native completed-skip/full-state resume applies."""
    job_dir = Path(job["job_dir"]).resolve()
    job_dir.mkdir(parents=True, exist_ok=True)
    with C.lock(job_dir / ".training.lock"):
        rt, records, report = _verified_runtime(job)
        seed, method = job["seed"], job["method"]
        fresh_spec, original_settings = deepcopy(job["spec"]), deepcopy(job["settings"])
        for key in ("nrg1_protocol_sha256", "continuation_protocol_sha256", "continuation"):
            fresh_spec.pop(key, None)
        original_settings.pop("continuation", None)
        original_settings["device"] = job["device"]
        fresh_spec.update(fresh_weights=True, source_weights_used=False)
        _need("configuration_id" in fresh_spec and str(seed) in fresh_spec["run_id"],
              "Fresh run ID/configuration ID is missing the assigned seed.")
        C.freeze(job_dir / "training_plan.json", dict(version=VERSION, seed=seed, method=method,
                 stage_targets=list(TARGETS), initial_spec=fresh_spec, initial_settings=original_settings,
                 snapshot=str(Path(job["snapshot"]).resolve()), runtime_hashes=job["runtime_hashes"],
                 manifest_sha256=report["manifest_sha256"], device=job["device"],
                 seed_training_module_sha256=C.sha(__file__),
                 continuation_reference_sha256=C.sha(C.__file__),
                 boundary_policy="original policy/Adam + sampler continuation; new episodes/global RNG",
                 source_seed42_weights_used=False))
        C.write_json(job_dir / "training_preflight.json", report)
        completed = []
        for index, target in enumerate(TARGETS):
            stage_dir = job_dir / "stages" / f"stage{target // 1_000_000:02d}M"
            stage_dir.mkdir(parents=True, exist_ok=True)
            _restore_native_train_function(rt)
            if index == 0:
                spec, settings = deepcopy(fresh_spec), deepcopy(original_settings)
                _need(settings["total_timesteps"] == target, "Fresh stage target changed.")
                orders = None
            else:
                previous = completed[-1]
                source = _own(previous["model_path"], job_dir)
                _need(source == (Path(previous["run_dir"]) / "last_model.zip").resolve(),
                      "Only the immediately preceding checkpoint of this seed may be loaded.")
                orders = _merged_sampler_logs(rt, records, completed, stage_dir / "source_sampler",
                                               original_settings, seed)
                source_job = dict(case_dir=str(stage_dir), status=previous["status"],
                                  spec=previous["spec"], settings=previous["settings"],
                                  source_model_sha256=previous["model_sha256"])
                interval = int(job.get("checkpoint_interval", 1_000_000))
                _need(interval > 0, "checkpoint_interval must be positive.")
                spec, settings = C.extension_config(source_job, target, interval)
                settings["device"] = job["device"]
                spec["run_id"] = fresh_spec["run_id"] + f"__continue_to_{target}"
                spec["configuration_id"] = fresh_spec["configuration_id"] + f"__continue_to_{target}"
                e = settings["continuation"]
                e.update(source_model=str(source), source_seed=seed, source_method=method,
                         source_nominal_target=previous["nominal_target"],
                         test_access="No test feedback enters training; source is this seed's previous stage.",
                         sampler_rule="original balanced-order replay of all preceding stage log suffixes")
                spec["continuation"] = e
                # Keep the exact reviewed patch for both continuation stages.
                patched, diff = C.patch_train_sb3(Path(rt.trainer.__file__).read_text(encoding="utf-8"))
                C.atomic_text(stage_dir / "patches" / "train_sb3_continuation.py", patched)
                C.atomic_text(stage_dir / "patches" / "train_sb3_continuation.diff", diff)
            spec = _freeze_stage(stage_dir, spec, settings, job, target)
            run_dir = stage_dir / "runs" / spec["run_id"]
            if index:
                context = dict(key=f"{method}_seed{seed}", case_dir=str(stage_dir))
                C.install_continuation(rt, context, spec, settings, records, orders)
            print(f"[STAGE] {method} seed={seed} cumulative target={target:,}; "
                  + ("fresh independent weights" if index == 0 else "own preceding policy/Adam checkpoint"),
                  flush=True)
            rt.trainer.train_run(spec, settings, records, run_dir)
            stage = dict(nominal_target=target, run_dir=str(run_dir), spec=spec, settings=settings)
            completed.append(_completed_stage(stage, rt, records, seed))
        last = completed[-1]
        result = dict(version=VERSION, state="completed", seed=seed, method=method,
                      final_model=last["model_path"], model_path=last["model_path"],
                      final_model_sha256=last["model_sha256"], final_run_dir=last["run_dir"],
                      actual_timesteps=last["status"]["actual_timesteps"], status=last["status"],
                      final_spec=last["spec"], final_settings=last["settings"], stages=completed,
                      source_seed42_weights_used=False)
        C.write_json(job_dir / "training_result.json", result)
        return result
