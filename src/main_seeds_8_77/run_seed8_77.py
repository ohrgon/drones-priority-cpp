#!/usr/bin/env python3
"""Train the main Strict/Relaxed models with the shared, fixed seeds 8 and 77.

Run from the original data project with its original Python environment:
  python /path/to/main_seeds_8_77/run_seed8_77.py --root . --phase check
  python /path/to/main_seeds_8_77/run_seed8_77.py --root . --gpus 0 --phase all
No historical checkpoint is included or used to initialize a new seed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from vendor import continuation_reference as ref

VERSION = "main_cost11_shared_seed8_77_v1_20261020"
SEEDS = (8, 77)
METHODS = ("strict", "relaxed")
TARGETS = (5_000_000, 20_000_000, 40_000_000)
PACKAGE = Path(__file__).resolve().parent
COUNTS = {("synthetic", "train"): 900, ("synthetic", "validation"): 180,
          ("synthetic", "test"): 720, ("osm", "test"): 500}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def absolute(path, root):
    p = Path(path).expanduser()
    return (p if p.is_absolute() else root / p).resolve()


def templates():
    out = {}
    for method in METHODS:
        item = ref.read_json(PACKAGE / "templates" / (method + ".json"))
        require(item["method"] == method and item["spec"]["seed"] == 42,
                "The original method template changed.")
        require(item["source_weights_in_package"] is False,
                "Historical weights are not part of this experiment.")
        require(item["runtime_hashes"][ref.TRAINER + ".py"] == ref.TRAINER_SHA256,
                "The authenticated native trainer changed.")
        for name, checksum in item["runtime_hashes"].items():
            path = PACKAGE / "runtime" / method / name
            require(path.is_file() and ref.sha(path) == checksum,
                    "Native source changed: " + str(path))
        require(item["evaluation"]["seed"] == 42 and
                item["evaluation"]["initial_seed"] == 20260907,
                "The original evaluation initial conditions changed.")
        out[method] = item
    require(out["strict"]["manifest_sha256"] == out["relaxed"]["manifest_sha256"],
            "Strict and Relaxed must use the same original manifest.")
    return out


def package_hashes():
    hashes = {}
    for p in sorted(PACKAGE.rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts:
            continue
        relative = p.relative_to(PACKAGE)
        if p.suffix == ".py" and relative.parts[0] != "verification":
            hashes[str(relative)] = ref.sha(p)
        elif relative.parts[0] == "templates" and p.suffix == ".json":
            hashes[str(relative)] = ref.sha(p)
    return hashes


def description():
    return dict(version=VERSION, historical_seed=42, added_seeds=list(SEEDS),
                same_seed_pair_for_both_methods=True,
                new_independent_training_runs=4,
                planned_seeds_per_method=[42, 8, 77],
                nominal_steps_per_new_run=40_000_000,
                cumulative_stage_targets=list(TARGETS),
                actual_stage_steps=[5_005_312, 20_004_864, 40_001_536],
                jobs=[dict(method=m, seed=s, effective_B=0 if m == "strict" else 6)
                      for s in SEEDS for m in METHODS],
                dataset_counts={"/".join(k): v for k, v in COUNTS.items()},
                reward="FULL COST11", beta=0.5,
                initial_weights="fresh for every (method, seed)",
                later_weights="only the same job's preceding stage",
                evaluation_seed=42, initial_seed=20260907,
                random_seed_draw=False,
                execution_state="configuration_only")


def locate_manifest(args, ts):
    if args.manifest:
        chosen = absolute(args.manifest, args.root)
    else:
        candidates = [args.root / ts[m]["original_relative_root"] / "manifest.csv"
                      for m in ("relaxed", "strict")]
        candidates.append(args.root / "protocols_5m_seed42" / "manifest.csv")
        candidates.append(args.root / "manifest.csv")
        matching = [p for p in candidates if p.is_file() and
                    ref.sha(p) == ts["strict"]["manifest_sha256"]]
        require(bool(matching), "The original 2,300-row manifest was not found. "
                "Run in the original project or pass --manifest /actual/path/manifest.csv. "
                "This command does not regenerate maps or splits. Checked: " +
                ", ".join(map(str, candidates)))
        chosen = matching[0].resolve()
    require(chosen.is_file(), "Missing manifest: " + str(chosen))
    require(ref.sha(chosen) == ts["strict"]["manifest_sha256"],
            "Manifest bytes differ from the original fixed Train900/Val180/Test1220 split.")
    rr = ref.rows(chosen)
    require(Counter((r["source"], r["split"]) for r in rr) == COUNTS,
            "The fixed split counts changed.")
    for field in ("map_id", "path", "sha256"):
        require(len({r[field] for r in rr}) == len(rr),
                "Duplicate map or cross-split overlap: " + field)
    require(all(Path(r["path"]).is_absolute() for r in rr),
            "The original manifest must retain its absolute map paths.")
    missing = [r["path"] for r in rr if not Path(r["path"]).is_file()]
    require(not missing, "Some original maps are missing before training starts: " +
            ", ".join(missing[:5]) + (f" (total {len(missing)})" if missing else ""))
    return chosen


def build_job_definition(method, seed, template, out, gpu):
    require(seed in SEEDS and method in METHODS, "Only the four agreed runs are allowed.")
    job_dir = out / (method + "_seed" + str(seed))
    spec = deepcopy(template["spec"])
    spec.pop("nrg1_protocol_sha256", None)
    for key in ("cost_sweep", "cost_scale_ablation", "strict_beta_sweep"):
        spec.pop(key, None)
    rid = f"ppo_{method}_C3NT_COST11_FULL_H5_b0.5_seed{seed}"
    spec.update(seed=seed, run_id=rid, configuration_id=rid.rsplit("_seed", 1)[0],
                stage="main_shared_seed_repetition", fresh_weights=True,
                source_weights_used=False,
                seed_repetition=dict(version=VERSION, shared_added_seeds=list(SEEDS),
                                     method=method, training_seed=seed,
                                     historical_seed=42, fresh_initialization=True))
    settings = deepcopy(template["settings"])
    settings["device"] = "cuda:0"
    # Retention changes disk bookkeeping only; stage 5M's actual training schedule
    # and its original 200k checkpoint interval are preserved.
    settings["retain_completed_resume"] = True
    require(settings["total_timesteps"] == TARGETS[0] and
            settings["checkpoint_every_timesteps"] == 200_000,
            "Unexpected original first-stage cadence.")
    return dict(version=VERSION, job_id=job_dir.name, job_dir=str(job_dir),
                method=method, seed=seed, effective_B=template["effective_B"],
                device="cuda:0", physical_gpu=gpu,
                snapshot=str(job_dir / "source_snapshot"),
                manifest=str(job_dir / "manifest.csv"),
                manifest_sha256=template["manifest_sha256"],
                spec=spec, settings=settings,
                runtime_hashes=template["runtime_hashes"],
                native_version=template["native_version"],
                software=template["software"],
                evaluation=dict(seed=42, initial_seed=20260907, deterministic=True,
                                checkpoint="final40M_only_no_selection"),
                source_weights_used=False,
                source_provenance=dict(original_run_id=template["spec"]["run_id"],
                    original_protocol_sha256=template["original_protocol_sha256"],
                    template_scope="code, hyperparameters and data split; no model weights"))


def prepare(args):
    ts = templates()
    manifest = locate_manifest(args, ts)
    out = args.out
    require(out != PACKAGE and PACKAGE not in out.parents,
            "Choose an output directory outside the delivered program folder.")
    require(out != args.root and out not in manifest.parents,
            "Output must be a new experiment subdirectory, separate from source data.")
    hashes = package_hashes()
    definitions = []
    for seed in SEEDS:
        for method in METHODS:
            gpu = args.gpus[len(definitions) % len(args.gpus)]
            definitions.append(build_job_definition(method, seed, ts[method], out, gpu))
    request = dict(version=VERSION, seeds=list(SEEDS), methods=list(METHODS),
                   source_manifest=str(manifest), manifest_sha256=ref.sha(manifest),
                   output=str(out), gpus=args.gpus, package_hashes=hashes,
                   job_definitions=definitions)
    out.mkdir(parents=True, exist_ok=True)
    pp = out / "plan.json"
    if pp.exists():
        plan = ref.read_json(pp)
        require(plan["request"] == request,
                "The saved seeds/settings/code/GPU assignment changed. Resume with the same command.")
        verify_plan(plan)
        return plan
    known = {".launcher.lock", "initializing.json"}
    known.update(j["job_id"] for j in definitions)
    require(not any(p.name not in known for p in out.iterdir()),
            "The output already contains unrelated files. Use a new --out directory.")
    ref.freeze(out / "initializing.json", request)
    jobs = []
    for definition in definitions:
        folder = Path(definition["job_dir"])
        folder.mkdir(parents=True, exist_ok=True)
        definition_path = folder / "job_definition.json"
        ref.freeze(definition_path, definition)
        snap = Path(definition["snapshot"])
        snap.mkdir(exist_ok=True)
        for name, checksum in definition["runtime_hashes"].items():
            dst = snap / name
            if not dst.exists():
                shutil.copy2(PACKAGE / "runtime" / definition["method"] / name, dst)
            require(ref.sha(dst) == checksum, "A job's native snapshot changed: " + str(dst))
        mf = Path(definition["manifest"])
        if not mf.exists():
            shutil.copy2(manifest, mf)
        require(ref.sha(mf) == definition["manifest_sha256"], "Copied manifest changed.")
        job = deepcopy(definition)
        job["spec"]["nrg1_protocol_sha256"] = ref.sha(definition_path)
        job["job_definition_sha256"] = ref.sha(definition_path)
        jobs.append(job)
    plan = dict(version=VERSION, request=request, jobs=jobs, experiment=description())
    ref.freeze(pp, plan)
    return plan


def verify_plan(plan):
    require(plan["version"] == VERSION and plan["request"]["package_hashes"] == package_hashes(),
            "Program or native runtime changed after the plan was saved.")
    require([(j["method"], j["seed"]) for j in plan["jobs"]] ==
            [(m, s) for s in SEEDS for m in METHODS], "The four paired jobs changed.")
    for job in plan["jobs"]:
        require(ref.sha(job["manifest"]) == job["manifest_sha256"], "Manifest changed.")
        definition_path = Path(job["job_dir"]) / "job_definition.json"
        require(ref.sha(definition_path) ==
                job["job_definition_sha256"], "Frozen job definition changed.")
        expected = ref.read_json(definition_path)
        expected["spec"]["nrg1_protocol_sha256"] = ref.sha(definition_path)
        expected["job_definition_sha256"] = ref.sha(definition_path)
        require(job == expected, "The saved job settings differ from its frozen definition.")
        require(job["spec"]["seed"] == job["seed"] and
                job["spec"]["fresh_weights"] is True and
                job["spec"]["source_weights_used"] is False,
                "Fresh initialization and seed contract changed.")
        for name, checksum in job["runtime_hashes"].items():
            require(ref.sha(Path(job["snapshot"]) / name) == checksum,
                    "Frozen native runtime changed: " + name)


def worker(args):
    plan = ref.read_json(args._plan)
    verify_plan(plan)
    matches = [j for j in plan["jobs"] if j["job_id"] == args._job]
    require(len(matches) == 1, "Unknown worker job.")
    job = matches[0]
    result_path = Path(job["job_dir"]) / "training_result.json"
    if args._worker == "check":
        from seed_training import verify_job
        result = verify_job(job)
        ref.write_json(Path(job["job_dir"]) / "preflight.json", result)
        print("[CHECK PASS] " + job["job_id"] + "; zero optimizer updates", flush=True)
        return
    if args._worker in ("train", "all"):
        from seed_training import train_job
        result = train_job(job)
        ref.write_json(result_path, result)
        if args._worker == "train":
            return
        for split in ("validation", "test"):
            cmd = [sys.executable, str(Path(__file__).resolve()), "--_worker", split,
                   "--_plan", str(args._plan), "--_job", job["job_id"]]
            subprocess.run(cmd, check=True, env=os.environ.copy())
        return
    require(result_path.is_file(), "Train this seed to 40M before evaluation.")
    from seed_evaluation import evaluate_job
    result = evaluate_job(job, ref.read_json(result_path), split=args._worker)
    ref.write_json(Path(job["job_dir"]) / ("evaluation_" + args._worker + "_result.json"), result)


def report(plan):
    rows = []
    for job in plan["jobs"]:
        folder = Path(job["job_dir"])
        item = dict(job_id=job["job_id"], method=job["method"], seed=job["seed"],
                    effective_B=job["effective_B"], planned_nominal_steps=TARGETS[-1])
        for name in ("training_result", "evaluation_validation_result", "evaluation_test_result"):
            p = folder / (name + ".json")
            item[name] = ref.read_json(p) if p.is_file() else None
        rows.append(item)
    path = Path(plan["request"]["output"]) / "experiment_status.json"
    ref.write_json(path, dict(version=VERSION, added_seeds=list(SEEDS), jobs=rows,
        historical_seed42="existing results retained separately; not rerun or fabricated",
        seed_statistics="Use per-seed summaries. Include the existing seed42 results before reporting n=3."))
    print("[STATUS] " + str(path), flush=True)


def launch(args, plan):
    phases = ["validation", "test"] if args.phase == "evaluate" else [args.phase]
    for phase in phases:
        pending = list(plan["jobs"])
        active = {}
        try:
            while pending or active:
                busy = {job["physical_gpu"] for job in active.values()}
                for job in list(pending):
                    if job["physical_gpu"] in busy:
                        continue
                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = str(job["physical_gpu"])
                    env["PYTHONHASHSEED"] = str(job["seed"])
                    cmd = [sys.executable, str(Path(__file__).resolve()), "--_worker", phase,
                           "--_plan", str(args.out / "plan.json"), "--_job", job["job_id"]]
                    proc = subprocess.Popen(cmd, env=env, start_new_session=True)
                    active[proc] = job
                    pending.remove(job)
                    busy.add(job["physical_gpu"])
                    print(f"[START] {job['job_id']} phase={phase} GPU={job['physical_gpu']} pid={proc.pid}", flush=True)
                for proc, job in list(active.items()):
                    code = proc.poll()
                    if code is not None:
                        del active[proc]
                        require(code == 0, f"{job['job_id']} stopped with code {code}. "
                                "Preserve its files and rerun the same command to resume.")
                        print("[DONE] " + job["job_id"] + " " + phase, flush=True)
                if active:
                    time.sleep(0.3)
        except BaseException:
            for proc in active:
                if proc.poll() is None:
                    try:
                        os.killpg(proc.pid, signal.SIGINT)
                    except ProcessLookupError:
                        pass
            for proc in active:
                proc.wait()
            raise


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path.cwd(), help="Original dataset project root.")
    p.add_argument("--manifest", type=Path, help="Original full frozen manifest (2,300 rows).")
    p.add_argument("--out", type=Path, help="New output; default ROOT/main_seeds_8_77.")
    p.add_argument("--gpus", type=int, nargs="+", default=[0], help="Physical GPU indices; one job per GPU.")
    p.add_argument("--phase", choices=("describe", "check", "train", "evaluate", "all", "report"), default="all")
    p.add_argument("--_worker", choices=("check", "train", "all", "validation", "test"), help=argparse.SUPPRESS)
    p.add_argument("--_plan", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--_job", help=argparse.SUPPRESS)
    a = p.parse_args()
    require(bool(a.gpus) and len(a.gpus) == len(set(a.gpus)) and all(g >= 0 for g in a.gpus),
            "GPU indices must be distinct nonnegative integers.")
    a.root = a.root.expanduser().resolve()
    a.out = absolute(a.out or "main_seeds_8_77", a.root)
    return a


def main():
    args = arguments()
    if args._worker:
        require(args._plan is not None and args._job is not None, "Incomplete worker arguments.")
        worker(args)
        return
    if args.phase == "describe":
        templates()
        print(json.dumps(description(), indent=2, ensure_ascii=False))
        return
    with ref.lock(args.out / ".launcher.lock"):
        if args.phase == "report":
            plan = ref.read_json(args.out / "plan.json")
            verify_plan(plan)
            report(plan)
            return
        plan = prepare(args)
        print("[FIXED SEEDS] Strict: 8,77 | Relaxed: 8,77 | 4 new independent 40M runs", flush=True)
        if args.phase != "report":
            launch(args, plan)
        report(plan)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[STOP] Rerun the same command to recover each seed's own saved state.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"[STOP] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
