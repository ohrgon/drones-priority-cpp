#!/usr/bin/env python3
"""Reproducible CPU wall-clock benchmark for drones-priority-cpp.

The published native rollout and refinement functions are called unchanged.
Run --help without NumPy/PyTorch. Heavy imports occur only in fresh workers.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import traceback

VERSION = "drones_cpu_timing_v3_original_inputs_20261031"
REFERENCE_COMMIT = "8e19d8822983245b811b83e88c99fd2be548a044"
REFERENCE_MAPS_SHA256 = "44f34ff149559fd03dd9543d5befc6e38c116d8a6b141bdece706c62fe5dcd12"
PACKAGES = ("numpy", "torch", "gymnasium", "stable-baselines3", "sb3-contrib")
TIMINGS = ("raw_wall_sec", "refine_validate_wall_sec", "total_plan_wall_sec", "attempt_wall_sec")
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path, rows, fields=None):
    rows = list(rows)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    temporary = Path(path).with_name(Path(path).name + ".writing")
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def software_versions():
    out = {}
    for name in PACKAGES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def cpu_details():
    models = set()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                models.add(line.split(":", 1)[1].strip())
    except OSError:
        pass
    available = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    topology = {}
    for cpu in available:
        values = {"topology_verified": True}
        for name in ("physical_package_id", "core_id"):
            try:
                values[name] = int(Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/{name}").read_text())
            except (OSError, ValueError):
                values["topology_verified"] = False
                values[name] = cpu if name == "core_id" else 0
        nodes = sorted(Path(f"/sys/devices/system/cpu/cpu{cpu}").glob("node[0-9]*"))
        values["numa_node"] = int(nodes[0].name[4:]) if len(nodes) == 1 else None
        topology[str(cpu)] = values
    return dict(cpu_models=sorted(models), logical_cpu_count=os.cpu_count(),
                available_cpu_ids=available, topology=topology,
                platform=sys.platform, python=sys.version)


def configure_cpu(threads, cpu_ids):
    for name in THREAD_ENV:
        os.environ[name] = str(threads)
    os.environ["OMP_DYNAMIC"] = "FALSE"
    os.environ["MKL_DYNAMIC"] = "FALSE"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTHONHASHSEED"] = "0"
    os.sched_setaffinity(0, set(cpu_ids))
    require(sorted(os.sched_getaffinity(0)) == sorted(cpu_ids), "CPU affinity was not applied.")


def reference_maps():
    path = Path(__file__).resolve().parent / "reference_test_maps.json"
    require(file_sha(path) == REFERENCE_MAPS_SHA256, "Bundled Test-map identity catalog changed.")
    return read_json(path)


def select_records(inputs, sources, per_source, selection_seed):
    repo = Path(inputs["root"])
    reference = {r["map_id"]: r for r in reference_maps()}
    manifest = inputs.get("map_manifest")
    if manifest:
        with Path(manifest).open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        required = {"map_id", "source", "split", "pattern", "tier", "path", "sha256"}
        require(rows and required <= rows[0].keys(), "Unsupported input manifest columns.")
        rows = [r for r in rows if r["split"] == "test"]
    else:
        require(inputs["map_mode"] == "original", "Archive portable manifest is missing.")
        rows = [dict(r, path="") for r in reference.values()]
    counts = {source: sum(r["source"] == source for r in rows) for source in ("synthetic", "osm")}
    require(counts == {"synthetic": 720, "osm": 500} and len(rows) == 1220,
            f"Expected fixed Test720+500, found {counts} ({len(rows)} total).")
    require(len({r["map_id"] for r in rows}) == len(rows), "Duplicate map IDs.")
    require({r["map_id"] for r in rows} == set(reference), "Test-map IDs differ from the paper's fixed Test set.")
    for row in rows:
        expected = reference[row["map_id"]]
        require(all(row[k] == expected[k] for k in ("source", "split", "pattern", "tier", "sha256")),
                "Test-map identity differs from the paper: " + row["map_id"])
    selected = []
    for source in sources:
        part = [r for r in rows if r["source"] == source]
        part.sort(key=lambda r: hashlib.sha256((str(selection_seed) + "\0" + r["map_id"]).encode()).hexdigest())
        if per_source:
            part = part[:per_source]
        selected.extend(part)
    out = []
    for record in selected:
        record = dict(record)
        if inputs["map_mode"] == "archive":
            path = (repo / record["path"]).resolve()
            require(path.is_relative_to(repo), "Map path escaped repository: " + str(path))
            require(path.is_file(), "Missing selected map: " + str(path))
            require(record.get("file_sha256"), "Archive manifest is missing file_sha256: " + record["map_id"])
            require(file_sha(path) == record["file_sha256"], "Map file SHA mismatch: " + str(path))
        else:
            candidates = []
            registered = record.get("path", "")
            if registered and "<" not in registered:
                registered_path = Path(registered).expanduser()
                candidates.append(registered_path if registered_path.is_absolute() else repo / registered_path)
            candidates.extend([repo.parent / record["map_id"], repo / record["map_id"]])
            candidates = list(dict.fromkeys(p.resolve() for p in candidates))
            allowed = [p for p in candidates if p.is_relative_to(repo.parent)]
            path = next((p for p in allowed if p.is_file()), None)
            require(path is not None, "Missing original Test map: " + record["map_id"] +
                    "\nChecked: " + ", ".join(map(str, allowed)))
            # Original .npy header bytes can differ; the canonical array hash is
            # still checked against the pinned paper catalog in every worker.
            record["file_sha256"] = file_sha(path)
        record["path"] = str(path)
        out.append({key: record[key] for key in ("map_id", "source", "split", "pattern", "tier", "path", "sha256", "file_sha256")})
    require(out, "No maps selected.")
    return out


def read_trials(path, repair_tail=False):
    path = Path(path)
    if not path.exists():
        return []
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        require(repair_tail, "Incomplete final JSONL line: " + str(path))
        stop = data.rfind(b"\n") + 1
        with path.open("r+b") as stream:
            stream.truncate(stop)
        print("[resume] Removed interrupted final result line:", path, flush=True)
        data = data[:stop]
    rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    keys = [(r["model_key"], r["map_id"], r["repeat"]) for r in rows]
    require(len(keys) == len(set(keys)), "Duplicate timing trials: " + str(path))
    return rows


def path_signature(row):
    fields = ("model_key", "map_id", "map_sha256", "initial_pair_id", "raw_success",
              "refined_success", "raw_steps", "refined_steps", "raw_actions_sha256",
              "refined_actions_sha256", "raw_model_T_sec", "raw_model_Lw_sec",
              "refined_model_T_sec", "refined_model_Lw_sec")
    return digest({field: row.get(field) for field in fields})


def validate_trial(row, expected_model, config, records_by_id):
    """Validate exact membership before reusing or aggregating a measured trial."""
    require(row.get("run_identity") == config["identity"], "Trial configuration identity differs.")
    require(row.get("model_key") == expected_model and expected_model in config["model_keys"],
            "Trial model membership/file differs.")
    method, seed = expected_model.split("_seed")
    require(row.get("method") == method and row.get("train_seed") == int(seed),
            "Trial method/training seed differs.")
    if "model_cpu_ids" in config:
        require(row.get("cpu_ids") == config["model_cpu_ids"][expected_model] and
                row.get("worker_slot") == config["model_slots"][expected_model],
                "Trial CPU allocation/worker slot differs.")
    repeat = row.get("repeat")
    require(isinstance(repeat, int) and not isinstance(repeat, bool) and 1 <= repeat <= config["repeats"],
            "Trial repeat is outside configuration membership.")
    rec = records_by_id.get(row.get("map_id"))
    require(rec is not None, "Trial map is outside configuration membership.")
    require(row.get("map_sha256") == rec["sha256"] and row.get("source") == rec["source"],
            "Trial map identity differs.")
    if "file_sha256" in rec:
        require(row.get("file_sha256") == rec["file_sha256"], "Trial input file identity differs.")
    require(isinstance(row.get("raw_success"), bool), "Trial success must be Boolean.")
    for name in ("raw_wall_sec", "attempt_wall_sec"):
        value = row.get(name)
        require(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0,
                "Invalid trial timing: " + name)
    if row["raw_success"]:
        refine = row.get("refine_validate_wall_sec")
        total = row.get("total_plan_wall_sec")
        require(row.get("refined_success") is True and isinstance(refine, (int, float)) and
                math.isfinite(refine) and refine >= 0 and isinstance(total, (int, float)) and math.isfinite(total),
                "Successful trial lacks refinement/complete-plan timing.")
        require(math.isclose(total, row["raw_wall_sec"] + refine, rel_tol=1e-12, abs_tol=1e-12) and
                math.isclose(row["attempt_wall_sec"], total, rel_tol=1e-12, abs_tol=1e-12),
                "Trial total does not equal Raw plus refinement.")
    else:
        require(row.get("total_plan_wall_sec") is None and row.get("refine_validate_wall_sec") is None
                and row.get("refined_success") is None,
                "Failed Raw trial must not report a complete-plan/refinement time.")
        require(row["attempt_wall_sec"] == row["raw_wall_sec"], "Failed attempt timing differs from Raw.")


def measure_case(bundle, rec, array, ic):
    """Two native function calls; no cache, disk I/O, or extra replay in timing."""
    start = time.perf_counter_ns()
    raw, trace = bundle.V.evaluate_map(bundle.rt, bundle.native, bundle.model, rec, array, ic, old=None)
    raw_end = time.perf_counter_ns()
    raw_sec = (raw_end - start) / 1e9
    success = bool(bundle.V.truth(raw["is_success"]))
    refined, refined_trace, audit = None, None, None
    refine_sec = None
    if success:
        refine_start = time.perf_counter_ns()
        refined, refined_trace, audit, _segments = bundle.C.common_refine(bundle.rt, raw, trace)
        refine_end = time.perf_counter_ns()
        refine_sec = (refine_end - refine_start) / 1e9
        require(bundle.V.truth(refined["is_success"]), "Refinement lost completion.")
    # Incomplete Raw episodes have no complete plan, and do not run A*.
    total = raw_sec + refine_sec if success else None
    row = dict(
        raw_success=success, refined_success=success if success else None,
        raw_steps=int(raw["steps"]), refined_steps=int(refined["steps"]) if success else None,
        active_cells=int(raw["active_cells"]), raw_coverage_pct=float(raw["coverage_pct"]),
        raw_wall_sec=raw_sec, refine_validate_wall_sec=refine_sec,
        total_plan_wall_sec=total, attempt_wall_sec=total if success else raw_sec,
        policy_action_only_wall_sec=raw.get("policy_compute_time_sec"),
        raw_model_T_sec=raw[bundle.V.T], raw_model_Lw_sec=raw[bundle.V.LW],
        refined_model_T_sec=refined[bundle.V.T] if success else None,
        refined_model_Lw_sec=refined[bundle.V.LW] if success else None,
        raw_actions_sha256=hashlib.sha256(trace["actions"].tobytes()).hexdigest(),
        refined_actions_sha256=hashlib.sha256(refined_trace["actions"].tobytes()).hexdigest() if success else None,
        refinement_status="completed_with_native_validation" if success else "skipped_raw_incomplete",
        initial_pair_id=ic["pair_id"], start_r=int(ic["start"][0]),
        start_c=int(ic["start"][1]), start_heading=int(ic["heading"]),
    )
    return row


def worker(config_path, model_key):
    config = read_json(config_path)
    out = Path(config_path).parent
    method, seed_text = model_key.split("_seed")
    seed = int(seed_text)
    for name, expected in config["benchmark_code_sha256"].items():
        require(file_sha(Path(__file__).resolve().parent / name) == expected,
                "Benchmark source changed during execution: " + name)
    assigned_cpus = config.get("model_cpu_ids", {}).get(model_key, config["cpu_ids"])
    slot = config.get("model_slots", {}).get(model_key, 0)
    configure_cpu(config["threads"], assigned_cpus)
    from cpu_bench_inputs import resolve_inputs
    require(resolve_inputs(config["repo_root"], config["input_mode"]) == config["inputs"],
            "Input layout or registered model paths changed since this run was configured.")
    from cpu_bench_native import load_native_model
    bundle = load_native_model(config["repo_root"], method, seed, config["threads"],
                               config["allow_version_difference"], input_mode=config["input_mode"])
    import numpy as np
    meta_path = out / "models" / (model_key + "_metadata.json")
    metadata = dict(bundle.metadata)
    metadata.update(benchmark_version=VERSION, model_key=model_key,
                    cpu_affinity=sorted(os.sched_getaffinity(0)),
                    worker_slot=slot, maximum_concurrent_workers=config.get("workers", 1),
                    thread_environment={name: os.environ.get(name) for name in THREAD_ENV},
                    warmup_maps=config["warmup_maps"], loaded_at=utc_now())
    if meta_path.exists():
        old = read_json(meta_path)
        # Configuration identity also guards software, CPU, source and input metadata.
        require(old.get("cpu_affinity") == metadata["cpu_affinity"], "Resume CPU affinity changed.")
    else:
        write_json(meta_path, metadata)
    records = config["records"]
    arrays, initial = {}, {}
    for rec in records:
        require(file_sha(rec["path"]) == rec["file_sha256"], "Map file changed: " + rec["map_id"])
        array = np.load(rec["path"], allow_pickle=False)
        require(array.shape == (50, 50) and array.dtype == np.uint8 and array.max() <= 5 and array.any(),
                "Expected nonempty uint8 50x50 map with priorities 0..5: " + rec["map_id"])
        require(bundle.rt.utils.array_hash(array) == rec["sha256"], "Canonical map SHA mismatch.")
        arrays[rec["map_id"]] = array
        initial[rec["map_id"]] = bundle.V.initial_condition(bundle.rt, rec, array)
    if config["check_only"]:
        print(f"[check OK] {model_key}: checkpoint, model, {len(records)} maps, CPU affinity", flush=True)
        return
    trials_path = out / "trials" / (model_key + ".jsonl")
    existing = read_trials(trials_path, repair_tail=True)
    records_by_id = {r["map_id"]: r for r in records}
    for row in existing:
        validate_trial(row, model_key, config, records_by_id)
    done = {(r["map_id"], r["repeat"]) for r in existing}
    expected_keys = {(r["map_id"], repeat) for r in records for repeat in range(1, config["repeats"] + 1)}
    require(done <= expected_keys, "Stored benchmark trials are outside this configuration.")
    signatures = {}
    for row in existing:
        require(row["run_identity"] == config["identity"], "Stored trial belongs to another run.")
        value = path_signature(row)
        require(row["map_id"] not in signatures or signatures[row["map_id"]] == value,
                "Existing deterministic paths disagree.")
        signatures[row["map_id"]] = value
    if done == expected_keys:
        print(f"[complete] {model_key}: all measured trials already present; no new timing", flush=True)
        return
    warmup_successes = 0
    for rec in records[:config["warmup_maps"]]:
        warmup = measure_case(bundle, rec, arrays[rec["map_id"]], initial[rec["map_id"]])
        warmup_successes += int(warmup["raw_success"])
    metadata.update(actual_raw_warmup_calls=config["warmup_maps"],
                    actual_refinement_warmup_calls=warmup_successes)
    write_json(meta_path, metadata)
    if config["warmup_maps"] and not warmup_successes:
        print("[warmup note] No successful Raw warm-up; refinement has not been warmed.", flush=True)
    print(f"[ready] {model_key}: slot={slot} CPU={assigned_cpus} threads={config['threads']} "
          f"maps={len(records)} repeats={config['repeats']} prior_trials={len(done)}", flush=True)
    with trials_path.open("a", encoding="utf-8") as stream:
        for repeat in range(1, config["repeats"] + 1):
            ordered = sorted(records, key=lambda r: digest([config["selection_seed"], repeat, r["map_id"]]))
            for rec in ordered:
                if (rec["map_id"], repeat) in done:
                    continue
                stamp = utc_now()
                row = measure_case(bundle, rec, arrays[rec["map_id"]], initial[rec["map_id"]])
                row.update(model_key=model_key, method=method, train_seed=seed, repeat=repeat,
                           worker_slot=slot, cpu_ids=assigned_cpus,
                           run_identity=config["identity"], map_id=rec["map_id"], source=rec["source"],
                           pattern=rec["pattern"], tier=rec["tier"], map_sha256=rec["sha256"],
                           file_sha256=rec["file_sha256"], started_at=stamp, ended_at=utc_now())
                signature = path_signature(row)
                require(rec["map_id"] not in signatures or signatures[rec["map_id"]] == signature,
                        "Deterministic Raw/Refined path changed between repeats: " + rec["map_id"])
                signatures[rec["map_id"]] = signature
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                done.add((rec["map_id"], repeat))
                total = f"{row['total_plan_wall_sec']:.3f}s" if row["raw_success"] else "RAW_FAILED"
                print(f"[{model_key} {len(done)}/{len(expected_keys)}] repeat={repeat} "
                      f"raw={row['raw_wall_sec']:.3f}s total={total} {rec['map_id']}", flush=True)


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def summarize(out):
    out = Path(out)
    config = read_json(out / "run_config.json")
    rows = []
    records_by_id = {r["map_id"]: r for r in config["records"]}
    for model in config["model_keys"]:
        trials = read_trials(out / "trials" / (model + ".jsonl"), repair_tail=False)
        for row in trials:
            validate_trial(row, model, config, records_by_id)
        rows.extend(trials)
    actual_keys = {(r["model_key"], r["map_id"], r["repeat"]) for r in rows}
    require(len(actual_keys) == len(rows), "Duplicate trial membership across result files.")
    expected_keys = {(model, rec["map_id"], repeat) for model in config["model_keys"]
                     for rec in config["records"] for repeat in range(1, config["repeats"] + 1)}
    require(actual_keys <= expected_keys, "Unexpected trial membership in results.")
    write_csv(out / "per_trial.csv", rows)
    groups = {}
    pairs = {}
    for row in rows:
        require(row["run_identity"] == config["identity"], "Mixed benchmark identities in output.")
        key = (row["model_key"], row["map_id"])
        groups.setdefault(key, []).append(row)
        previous = pairs.setdefault(row["map_id"], row["initial_pair_id"])
        require(previous == row["initial_pair_id"], "Initial conditions differ between models.")
    per_map = []
    for key, trials in sorted(groups.items()):
        require(len({path_signature(r) for r in trials}) == 1, "Repeated paths differ: " + str(key))
        representative = trials[0]
        item = {k: representative[k] for k in (
            "model_key", "method", "train_seed", "map_id", "source", "pattern", "tier",
            "map_sha256", "initial_pair_id", "raw_success", "raw_steps", "refined_steps")}
        item.update(n_repeats=len(trials), expected_repeats=config["repeats"],
                    complete_repeats={r["repeat"] for r in trials} == set(range(1, config["repeats"] + 1)))
        for metric in TIMINGS:
            values = [r[metric] for r in trials if r[metric] is not None]
            item[metric + "_median"] = statistics.median(values) if values else None
            item[metric + "_min"] = min(values) if values else None
            item[metric + "_max"] = max(values) if values else None
        per_map.append(item)
    write_csv(out / "per_map.csv", per_map)
    completed = [r for r in per_map if r["complete_repeats"]]
    success_models = {}
    for row in completed:
        if row["raw_success"]:
            success_models.setdefault(row["map_id"], set()).add(row["model_key"])
    common = {key for key, models in success_models.items() if models == set(config["model_keys"])}
    summary, paper = [], []
    for model in config["model_keys"]:
        for source in ["all"] + config["sources"]:
            selected = [r for r in completed if r["model_key"] == model and
                        (source == "all" or r["source"] == source)]
            successes = [r for r in selected if r["raw_success"]]
            failures = [r for r in selected if not r["raw_success"]]
            populations = {"all_attempts": selected, "successful": successes,
                           "failed": failures, "common_success": [r for r in selected if r["map_id"] in common]}
            for population, members in populations.items():
                metrics = TIMINGS if population in ("successful", "common_success") else ("raw_wall_sec", "attempt_wall_sec")
                for metric in metrics:
                    values = [r[metric + "_median"] for r in members if r[metric + "_median"] is not None]
                    if not values:
                        continue
                    summary.append(dict(model_key=model, source=source, population=population,
                        metric=metric, n_maps=len(members), repeats_per_map=config["repeats"],
                        mean=statistics.mean(values), sample_sd=statistics.stdev(values) if len(values) > 1 else None,
                        median=statistics.median(values), p95=percentile(values, .95),
                        max_of_map_medians=max(values),
                        max_single_trial=max(r[metric + "_max"] for r in members)))
            for population in ("successful", "common_success"):
                members = populations[population]
                item = dict(model_key=model, source=source, population=population,
                            n_evaluated_maps=len(selected), n_success=len(successes), n_failed=len(failures),
                            n_timing_maps=len(members),
                            success_pct=100 * len(successes) / len(selected) if selected else None)
                for metric in TIMINGS[:3]:
                    values = [r[metric + "_median"] for r in members]
                    for suffix, value in (
                        ("mean", statistics.mean(values) if values else None),
                        ("median", statistics.median(values) if values else None),
                        ("p95", percentile(values, .95)),
                        ("max_map_median", max(values) if values else None),
                        ("max_single_trial", max((r[metric + "_max"] for r in members), default=None))):
                        item[metric + "_" + suffix] = value
                paper.append(item)
    write_csv(out / "summary.csv", summary)
    write_csv(out / "paper_table.csv", paper)
    expected = len(expected_keys)
    complete = actual_keys == expected_keys
    lines = [VERSION, f"Status: {'COMPLETE' if complete else 'PARTIAL - do not report as a complete benchmark'}",
             f"Measured trials: {len(rows)}/{expected}",
             f"CPU pool: {config['cpu_ids']}; maximum concurrent workers: {config.get('workers', 1)}; threads per worker: {config['threads']}",
             "Timing unit: seconds. Each map is summarized by its median over measured repetitions.",
             "P95 uses linear interpolation over map medians. max_single_trial is the largest observed repetition.",
             "Complete-plan statistics exclude failed Raw episodes; failure attempt times remain in per_trial.csv/summary.csv.",
             "Raw includes native diagnostics/trace construction. Refinement includes replay and invariant validation.",
             "Excluded: model loading, map I/O/hash checks, initial-condition selection, warm-up, output I/O and KMZ export.",
             "These are observed offline planning times, not real-time or worst-case guarantees.", ""]
    for row in paper:
        if row["source"] == "all" and row["population"] == "successful":
            value = row["total_plan_wall_sec_mean"]
            lines.append(f"{row['model_key']}: completed {row['n_success']}/{row['n_evaluated_maps']}; "
                         f"mean total plan = {value:.6f} s" if value is not None else f"{row['model_key']}: no completed timing maps")
    (out / "SUMMARY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dict(complete=complete, n_trials=len(rows), expected_trials=expected)


@contextlib.contextmanager
def output_lock(out):
    import fcntl
    with (out / ".benchmark.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another benchmark is using this output directory.") from exc
        yield


def run_model_workers(config, out):
    """Run fresh model processes in fixed slots; never share a slot concurrently."""
    pending = {slot: [model for model in config["model_keys"] if config["model_slots"][model] == slot]
               for slot in range(config["workers"])}
    active, errors, events = {}, [], []
    try:
        while any(pending.values()) or active:
            for slot, queue in pending.items():
                if slot in active or not queue:
                    continue
                model = queue.pop(0)
                log_path = out / "logs" / (model + ".log")
                log = log_path.open("a", encoding="utf-8")
                command = [sys.executable, "-u", str(Path(__file__).resolve()),
                           "--worker", str(out / "run_config.json"), model]
                try:
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                               env=os.environ.copy())
                except BaseException:
                    log.close()
                    raise
                event = dict(model_key=model, worker_slot=slot, cpu_ids=config["cpu_groups"][slot],
                             pid=process.pid, started_at=utc_now())
                events.append(event)
                active[slot] = (model, process, log, log_path, event)
                print(f"[start] slot={slot} {model}; CPU={config['cpu_groups'][slot]}; log: {log_path}", flush=True)
            for slot, item in list(active.items()):
                model, process, log, log_path, event = item
                code = process.poll()
                if code is None:
                    continue
                process.wait()
                log.close()
                del active[slot]
                event.update(ended_at=utc_now(), return_code=code)
                if code:
                    errors.append(dict(model_key=model, return_code=code, log=str(log_path)))
                    tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
                    print("\n".join(tail), file=sys.stderr)
                    print(f"[failed] {model}; independent workers continue", file=sys.stderr, flush=True)
                else:
                    print(f"[done] slot={slot} {model}", flush=True)
            if active:
                time.sleep(0.2)
    finally:
        # Reap every writer before any result repair or cross-model aggregation.
        for _, process, _, _, _ in active.values():
            if process.poll() is None:
                process.terminate()
        deadline = time.monotonic() + 8
        for model, process, log, _, event in active.values():
            try:
                process.wait(timeout=max(.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            finally:
                log.close()
            event.update(ended_at=utc_now(), return_code=process.returncode, interrupted=True)
        write_json(out / "worker_events.json", events)
    return errors


def stop_requested(_signal_number, _frame):
    raise KeyboardInterrupt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--repo-root", help="Existing public/1gb/collected archive root; no download")
    input_group.add_argument("--suite-root", help="Original server uav_chapter4_suite_v2 root; read existing models/code/maps directly")
    parser.add_argument("--out", help="New output directory; required with --resume")
    parser.add_argument("--methods", nargs="+", choices=("strict", "relaxed"), default=["strict", "relaxed"])
    parser.add_argument("--seeds", nargs="+", type=int, choices=(8, 42, 77), default=[8, 42, 77])
    parser.add_argument("--sources", nargs="+", choices=("synthetic", "osm"), default=["synthetic", "osm"])
    parser.add_argument("--threads", type=int, default=4, help="Same CPU thread budget for every model")
    parser.add_argument("--workers", type=int, default=1, help="Maximum simultaneous model processes; cores allocated automatically")
    parser.add_argument("--cpu-ids", default="auto", help="Eligible CPU pool; auto detects disjoint physical-core groups")
    parser.add_argument("--repeats", type=int, default=3, help="Measured repetitions per model/map")
    parser.add_argument("--warmup-maps", type=int, default=1, help="Full unmeasured pipeline warm-ups per worker start")
    parser.add_argument("--maps-per-source", type=int, default=0, help="0=all; positive value selects a fixed hash-ranked subset")
    parser.add_argument("--selection-seed", type=int, default=20261031)
    parser.add_argument("--allow-version-difference", action="store_true", help="Explicitly permit/log software differences from original training environment")
    parser.add_argument("--check-only", action="store_true", help="Verify CPU/model/input setup without timed rollouts")
    parser.add_argument("--resume", action="store_true", help="Retain measured trials from an identical interrupted run")
    parser.add_argument("--summarize-only", metavar="OUTPUT_DIR", help="Regenerate summaries from saved measured trials")
    parser.add_argument("--worker", nargs=2, metavar=("CONFIG", "MODEL"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        worker(*args.worker)
        return 0
    if args.summarize_only:
        summary_out = Path(args.summarize_only).resolve()
        with output_lock(summary_out):
            print(json.dumps(summarize(summary_out), indent=2))
        return 0
    require(sys.version_info >= (3, 10), "Python 3.10 or later is required.")
    require(args.threads >= 1 and args.workers >= 1 and args.repeats >= 1 and args.warmup_maps >= 0 and args.maps_per_source >= 0,
            "Threads/workers/repeats must be positive; warmup/maps-per-source must be nonnegative.")
    require(not args.resume or args.out, "--resume requires the original --out path.")
    require(not (args.resume and args.check_only), "Use a new directory for --check-only.")
    from cpu_bench_inputs import resolve_inputs
    input_mode = "original" if args.suite_root else "auto"
    input_root = args.suite_root or args.repo_root or str(Path(__file__).resolve().parent.parent)
    inputs = resolve_inputs(input_root, input_mode)
    repo = Path(inputs["root"])
    details = cpu_details()
    versions = software_versions()
    require(all(versions.values()), "Missing packages: " + ", ".join(k for k, v in versions.items() if not v))
    methods, seeds, sources = list(dict.fromkeys(args.methods)), list(dict.fromkeys(args.seeds)), list(dict.fromkeys(args.sources))
    records = select_records(inputs, sources, args.maps_per_source, args.selection_seed)
    model_keys = [f"{method}_seed{seed}" for seed in seeds for method in methods]
    for model_key in model_keys:
        checkpoint = inputs["checkpoints"][model_key]
        checkpoint_path = Path(checkpoint["path"])
        require(checkpoint_path.is_file(), "Missing registered checkpoint for " + model_key + ": " + str(checkpoint_path) +
                "\nRun diagnose_cpu_inputs.py --suite-root <original suite> to report existing locations.")
        require(file_sha(checkpoint_path) == checkpoint["sha256"],
                "Registered checkpoint differs from the paper: " + str(checkpoint_path))
    from cpu_bench_parallel import allocate_cpu_groups, assign_models_to_slots
    workers = min(args.workers, len(model_keys))
    cpu_groups = allocate_cpu_groups(details, args.threads, workers, args.cpu_ids)
    model_slots = assign_models_to_slots(model_keys, workers)
    model_cpu_ids = {model: cpu_groups[model_slots[model]] for model in model_keys}
    cpu_ids = sorted(cpu for group in cpu_groups for cpu in group)
    own_dir = Path(__file__).resolve().parent
    config = dict(version=VERSION, reference_commit=REFERENCE_COMMIT, repo_root=str(repo),
                  input_mode=input_mode, inputs=inputs,
                  model_keys=model_keys, sources=sources, records=records,
                  threads=args.threads, cpu_ids=cpu_ids, repeats=args.repeats,
                  workers_requested=args.workers, workers=workers, cpu_groups=cpu_groups,
                  model_slots=model_slots, model_cpu_ids=model_cpu_ids,
                  warmup_maps=min(args.warmup_maps, len(records)), maps_per_source=args.maps_per_source,
                  selection_seed=args.selection_seed, allow_version_difference=args.allow_version_difference,
                  check_only=args.check_only, software=versions, python=sys.version,
                  cpu_identity={k: details[k] for k in ("cpu_models", "logical_cpu_count", "platform")},
                  selected_cpu_topology={str(k): details["topology"][str(k)] for k in cpu_ids},
                  benchmark_code_sha256={name: file_sha(own_dir / name) for name in ("benchmark_cpu.py", "cpu_bench_native.py", "cpu_bench_parallel.py", "cpu_bench_inputs.py", "reference_test_maps.json")},
                  input_manifest_sha256=file_sha(inputs["map_manifest"]) if inputs.get("map_manifest") else None,
                  fixed_test_catalog_sha256=REFERENCE_MAPS_SHA256,
                  checkpoint_manifest_sha256=file_sha(inputs["checkpoint_manifest"]) if inputs.get("checkpoint_manifest") else None)
    config["identity"] = digest(config)
    out = Path(args.out or ("cpu_timing_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S"))).expanduser().resolve()
    if args.resume:
        require((out / "run_config.json").is_file(), "No original run_config.json to resume.")
    else:
        require(not out.exists() or not any(out.iterdir()), "Output is not empty; use a new --out or identical --resume.")
    out.mkdir(parents=True, exist_ok=True)
    with output_lock(out):
        if args.resume:
            old = read_json(out / "run_config.json")
            require(old["identity"] == config["identity"],
                    "Resume configuration/CPU/software/code/maps changed. Reuse identical arguments, or start a new output.")
        else:
            write_json(out / "run_config.json", config)
            write_json(out / "hardware.json", details)
            write_csv(out / "selected_maps.csv", records)
        (out / "trials").mkdir(exist_ok=True)
        (out / "models").mkdir(exist_ok=True)
        (out / "logs").mkdir(exist_ok=True)
        if args.resume:
            # Repair all interrupted tails before any cross-model summary is attempted.
            for model in model_keys:
                read_trials(out / "trials" / (model + ".jsonl"), repair_tail=True)
        configure_cpu(args.threads, cpu_ids)
        session = dict(started_at=utc_now(), resume=args.resume, state="running",
                       cpu_ids=cpu_ids, threads_per_worker=args.threads, workers=workers,
                       cpu_groups=cpu_groups, load_average_start=os.getloadavg())
        write_json(out / "status.json", session)
        print(f"Output: {out}\nInputs: {inputs['input_kind']}\nSource package: {inputs['source_package']}\nModels: {model_keys}\nMaps: {len(records)}; repeats: {args.repeats}; workers: {workers}; threads/worker: {args.threads}\nCPU groups: {cpu_groups}", flush=True)
        exit_code = 0
        previous_sigterm_handler = signal.signal(signal.SIGTERM, stop_requested)
        try:
            errors = run_model_workers(config, out)
            if errors:
                session.update(state="failed", failed_models=errors)
                exit_code = 1
            else:
                session["state"] = "check_passed" if args.check_only else "completed"
        except KeyboardInterrupt:
            session["state"] = "interrupted"
            exit_code = 130
            print("Interrupted. Completed measured trials are retained; resume with identical arguments.", file=sys.stderr)
        except Exception as exc:
            session["state"] = "failed"
            session["error"] = str(exc)
            exit_code = 1
            print(str(exc), file=sys.stderr)
        finally:
            if not args.check_only:
                try:
                    for model in model_keys:
                        read_trials(out / "trials" / (model + ".jsonl"), repair_tail=True)
                    session["summary"] = summarize(out)
                    if not session["summary"]["complete"] and session["state"] == "completed":
                        session["state"] = "incomplete"
                        exit_code = exit_code or 1
                except Exception as exc:
                    session["summary_error"] = str(exc)
                    exit_code = exit_code or 1
            session.update(ended_at=utc_now(), load_average_end=os.getloadavg())
            write_json(out / "status.json", session)
            signal.signal(signal.SIGTERM, previous_sigterm_handler)
        if exit_code == 0:
            print("Setup checks passed." if args.check_only else (out / "SUMMARY.txt").read_text(), flush=True)
        return exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
