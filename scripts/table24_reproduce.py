#!/usr/bin/env python3
"""Audit and aggregate saved drones_cpu_timing_v3 results without rerunning plans.

Python standard library only. Input is the completed output directory containing
run_config.json and trials/*.jsonl. Source files are read only. All times are s.
Example:
  python scripts/table24_reproduce.py --input results/cpu_timing/raw/timing_full_v3_t1_r3 --out /tmp/timing_analysis_recomputed
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics as st
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


TIMINGS = ("raw_wall_sec", "refine_validate_wall_sec", "total_plan_wall_sec",
           "attempt_wall_sec", "policy_action_only_wall_sec")
SIGNATURE = ("raw_success", "refined_success", "raw_steps", "refined_steps",
             "active_cells", "raw_coverage_pct", "raw_model_T_sec", "raw_model_Lw_sec",
             "refined_model_T_sec", "refined_model_Lw_sec", "raw_actions_sha256",
             "refined_actions_sha256", "refinement_status", "initial_pair_id",
             "start_r", "start_c", "start_heading", "map_sha256", "file_sha256")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def percentile(values, q):
    """Linear interpolation, equivalent to PERCENTILE.INC / numpy method=linear."""
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (index - lo) * (values[hi] - values[lo])


def describe(values):
    values = list(values)
    return {"n": len(values), "mean": st.mean(values) if values else None,
            "median": st.median(values) if values else None,
            "p95": percentile(values, .95), "min": min(values) if values else None,
            "max": max(values) if values else None}


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f"No rows to export: {path}")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def all_groups(pairs):
    for source in ("synthetic", "osm"):
        for method in ("strict", "relaxed"):
            yield source, method, [p for p in pairs if p["source"] == source and p["method"] == method]


def timing_table(source, method, members, seed=None):
    successful = [p for p in members if p["raw_success"]]
    out = {"domain": source, "method": method}
    if seed is not None:
        out["training_seed"] = seed
    out.update(n_evaluated_pairs=len(members), n_successful_pairs=len(successful),
               n_failed_pairs=len(members)-len(successful),
               completion_pct=100*len(successful)/len(members) if members else None)
    for key, col in (("stage1_mean_sec", "raw_wall_sec"),
                     ("stage2_including_validation_mean_sec", "refine_validate_wall_sec"),
                     ("total_mean_sec", "total_plan_wall_sec"),
                     ("policy_action_selection_mean_sec", "policy_action_only_wall_sec")):
        out[key] = st.mean(p[col+"_median"] for p in successful) if successful else None
    vals = [p["total_plan_wall_sec_median"] for p in successful]
    out.update(total_median_sec=st.median(vals) if vals else None,
               total_p95_sec=percentile(vals, .95),
               total_max_single_trial_sec=max((p["total_plan_wall_sec_max"] for p in successful), default=None),
               total_max_pair_median_sec=max(vals, default=None))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    root, out = args.input.resolve(), args.out.resolve()
    if root == out or root in out.parents:
        raise SystemExit("Output must be separate from the source result directory.")
    config = read_json(root/"run_config.json")
    status = read_json(root/"status.json")
    hardware = read_json(root/"hardware.json")
    models = config["model_keys"]
    records = {r["map_id"]: r for r in config["records"]}
    n_repeats = config["repeats"]
    rows = []
    for model in models:
        with (root/"trials"/(model+".jsonl")).open(encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    expected = {(m, k, r) for m in models for k in records for r in range(1,n_repeats+1)}
    keys = [(r["model_key"], r["map_id"], r["repeat"]) for r in rows]
    actual = set(keys)
    issues = []
    def check(condition, message):
        if not condition:
            issues.append(message)
    check(status["state"] == "completed", "Main run status is not completed")
    check(len(records) == len(config["records"]), "Duplicate selected map IDs")
    check(len(keys) == len(actual), "Duplicate trial keys")
    check(actual == expected, f"Trial membership mismatch: missing={len(expected-actual)}, extra={len(actual-expected)}")
    batches = defaultdict(list)
    initial = defaultdict(set)
    for r in rows:
        key = (r["model_key"],r["map_id"])
        batches[key].append(r)
        initial[r["map_id"]].add((r["initial_pair_id"],r["start_r"],r["start_c"],r["start_heading"]))
        check(r["run_identity"] == config["identity"], f"Run identity differs: {key}")
        rec = records.get(r["map_id"])
        if rec is not None:
            check(r["map_sha256"] == rec["sha256"], f"Map hash differs: {key}")
            check(r["source"] == rec["source"], f"Domain differs: {key}")
        for col in TIMINGS:
            v = r[col]
            check(v is None or (isinstance(v,(int,float)) and math.isfinite(v) and v >= 0), f"Invalid timing: {key}/{col}")
        if r["raw_success"]:
            check(r["refined_success"] is True, f"Missing successful refinement: {key}")
            check(all(r[col] is not None for col in TIMINGS), f"Missing complete-plan timing: {key}")
            check(abs(r["raw_wall_sec"]+r["refine_validate_wall_sec"]-r["total_plan_wall_sec"]) < 1e-9, f"Stage-time algebra: {key}")
            check(r["attempt_wall_sec"] == r["total_plan_wall_sec"], f"Successful attempt time: {key}")
            check(r["refined_steps"] <= r["raw_steps"], f"Refinement increases moves: {key}")
            check(r["refined_model_T_sec"] <= r["raw_model_T_sec"]+1e-6, f"Refinement increases T: {key}")
            check(r["refined_model_Lw_sec"] <= r["raw_model_Lw_sec"]+1e-6, f"Refinement increases Lw: {key}")
        else:
            check(r["refine_validate_wall_sec"] is None and r["total_plan_wall_sec"] is None,
                  f"Failed attempt has complete-plan timing: {key}")
            check(r["attempt_wall_sec"] == r["raw_wall_sec"], f"Failed attempt time: {key}")
            check(r["refinement_status"] == "skipped_raw_incomplete", f"Failed Raw refinement status: {key}")
    check(all(len(v)==1 for v in initial.values()), "Initial states differ among models/repetitions")
    pairs = []
    for key, trials in sorted(batches.items()):
        check(len(trials)==n_repeats, f"Wrong repeat count: {key}")
        signatures = {tuple(r[k] for k in SIGNATURE) for r in trials}
        check(len(signatures)==1, f"Repeated path/metrics differ: {key}")
        r = trials[0]
        item = {k:r[k] for k in ("model_key","method","train_seed","map_id","source","pattern","tier")+SIGNATURE}
        item["n_repeats"] = len(trials)
        for col in TIMINGS:
            vals = [r[col] for r in trials if r[col] is not None]
            item[col+"_median"] = st.median(vals) if vals else None
            item[col+"_min"] = min(vals) if vals else None
            item[col+"_max"] = max(vals) if vals else None
        if item["raw_success"]:
            item["planning_to_same_refined_T_pct"] = 100*item["total_plan_wall_sec_median"]/item["refined_model_T_sec"]
            item["max_trial_to_same_refined_T_pct"] = 100*item["total_plan_wall_sec_max"]/item["refined_model_T_sec"]
            item["total_repeat_range_over_median_pct"] = 100*(item["total_plan_wall_sec_max"]-item["total_plan_wall_sec_min"])/item["total_plan_wall_sec_median"]
        else:
            item["planning_to_same_refined_T_pct"] = None
            item["max_trial_to_same_refined_T_pct"] = None
            item["total_repeat_range_over_median_pct"] = None
        pairs.append(item)
    out.mkdir(parents=True,exist_ok=True)
    audit = {"passed":not issues,"issues":issues,"run_identity":config["identity"],
             "expected_trials":len(expected),"observed_trials":len(rows),
             "unique_trial_keys":len(actual),"model_map_pairs":len(pairs),
             "distinct_maps":len(records),"repeats":n_repeats,
             "source_file_sha256":{p.relative_to(root).as_posix():sha256(p)
                  for p in [root/"run_config.json",root/"status.json",root/"hardware.json",root/"per_trial.csv"]
                  +[root/"trials"/(m+".jsonl") for m in models]}}
    (out/"aggregation_audit.json").write_text(json.dumps(audit,indent=2)+"\n",encoding="utf-8")
    if issues:
        raise SystemExit("Audit failed; see aggregation_audit.json")
    successful = [p for p in pairs if p["raw_success"]]
    failed = [p for p in pairs if not p["raw_success"]]
    successful_trials = [r for r in rows if r["raw_success"]]
    success_models = defaultdict(set)
    for p in successful:
        success_models[p["map_id"]].add(p["model_key"])
    common = {mid for mid, ms in success_models.items() if ms==set(models)}

    table24 = [timing_table(s,m,ps) for s,m,ps in all_groups(pairs)]
    s8 = [timing_table(s,m,[p for p in ps if p["train_seed"]==seed],seed)
          for s,m,ps in all_groups(pairs) for seed in sorted({p["train_seed"] for p in ps})]
    table_common = []
    for source, method, members in all_groups(pairs):
        common_members = [p for p in members if p["map_id"] in common]
        item = timing_table(source,method,common_members)
        item["population"] = "six_model_common_success"
        item["n_common_success_timing_pairs"] = len(common_members)
        item["n_evaluated_pairs"] = len(members)
        item["n_successful_pairs"] = sum(p["raw_success"] for p in members)
        item["n_failed_pairs"] = len(members)-item["n_successful_pairs"]
        item["completion_pct"] = 100*item["n_successful_pairs"]/len(members)
        table_common.append(item)
    write_csv(out/"Table24_CPU_planning_times.csv",table24)
    write_csv(out/"TableS8_seedwise_CPU_planning_times.csv",s8)
    write_csv(out/"common_success_CPU_planning_times.csv",table_common)
    write_csv(out/"per_pair_analysis.csv",pairs)
    write_csv(out/"failed_pairs.csv",failed)
    failure_stats = []
    for source,method,ps in all_groups(pairs):
        for seed in ["pooled"]+sorted({p["train_seed"] for p in ps}):
            scope = ps if seed=="pooled" else [p for p in ps if p["train_seed"]==seed]
            fs = [p for p in scope if not p["raw_success"]]
            vals = [p["attempt_wall_sec_median"] for p in fs]
            failure_stats.append({"domain":source,"method":method,"training_seed":seed,
                "n_evaluated_pairs":len(scope),"n_failed_pairs":len(fs),
                "failed_pct":100*len(fs)/len(scope),
                **{k+"_sec":v for k,v in describe(vals).items() if k!="n"},
                "max_single_trial_sec":max((p["attempt_wall_sec_max"] for p in fs),default=None)})
    write_csv(out/"failed_attempt_statistics.csv",failure_stats)

    comparisons,stage_shares = [],[]
    for source,method,ps in [("all","all",pairs)]+list(all_groups(pairs)):
        ss = [p for p in ps if p["raw_success"]]
        vals = [p["planning_to_same_refined_T_pct"] for p in ss]
        comparisons.append({"domain":source,"method":method,"n_successful_pairs":len(ss),
            **{"planning_to_T_"+k+"_pct":v for k,v in describe(vals).items() if k!="n"},
            "max_individual_planning_to_T_pct":max(p["max_trial_to_same_refined_T_pct"] for p in ss),
            "mean_refined_T_sec":st.mean(p["refined_model_T_sec"] for p in ss),
            "minimum_refined_T_sec":min(p["refined_model_T_sec"] for p in ss),
            "maximum_refined_T_sec":max(p["refined_model_T_sec"] for p in ss)})
        selected_keys = {(p["model_key"],p["map_id"]) for p in ss}
        ts = [r for r in successful_trials if (r["model_key"],r["map_id"]) in selected_keys]
        # math.fsum: exactly rounded sums, identical across Python versions (sum() changed its float accumulation in 3.12)
        total = math.fsum(r["total_plan_wall_sec"] for r in ts)
        raw = math.fsum(r["raw_wall_sec"] for r in ts)
        refine = math.fsum(r["refine_validate_wall_sec"] for r in ts)
        policy = math.fsum(r["policy_action_only_wall_sec"] for r in ts)
        stage_shares.append({"domain":source,"method":method,"n_successful_trials":len(ts),
             "sum_total_sec":total,"sum_stage1_sec":raw,"sum_stage2_sec":refine,
             "stage1_pct_of_summed_total":100*raw/total,
             "stage2_pct_of_summed_total":100*refine/total,
             "policy_pct_of_summed_total":100*policy/total,
             "policy_pct_of_summed_stage1":100*policy/raw})
    write_csv(out/"planning_to_matched_mission_time.csv",comparisons)
    write_csv(out/"stage_time_shares.csv",stage_shares)
    write_csv(out/"longest_50_plans.csv",sorted(successful,key=lambda p:p["total_plan_wall_sec_max"],reverse=True)[:50])
    tier_stats = []
    for source,method,ps in all_groups(pairs):
        for tier in sorted({p["tier"] for p in ps}):
            selected = [p for p in ps if p["tier"]==tier]
            row = timing_table(source,method,selected)
            row["tier"]=tier
            good = [p for p in selected if p["raw_success"]]
            row["active_cells_mean_successful"] = st.mean(p["active_cells"] for p in good)
            row["raw_steps_mean_successful"] = st.mean(p["raw_steps"] for p in good)
            tier_stats.append(row)
    write_csv(out/"timing_by_tier.csv",tier_stats)

    model_summary = [timing_table("all",model.split("_seed")[0],
                                 [p for p in pairs if p["model_key"]==model],
                                 int(model.split("_seed")[1])) for model in models]
    write_csv(out/"model_summary_all_maps.csv",model_summary)
    elapsed = (datetime.fromisoformat(status["ended_at"])-datetime.fromisoformat(status["started_at"])).total_seconds()
    result = {"audit_passed":True,"run_identity":config["identity"],
        "trials":len(rows),"successful_trials":len(successful_trials),"failed_trials":len(rows)-len(successful_trials),
        "model_map_pairs":len(pairs),"successful_pairs":len(successful),"failed_pairs":len(failed),
        "distinct_maps":len(records),"common_success_distinct_maps":len(common),
        "common_success_by_domain":dict(sorted(Counter(records[mid]["source"] for mid in common).items())),
        "cpu_models":hardware["cpu_models"],"cpu_groups":config["cpu_groups"],
        "threads_per_worker":config["threads"],"maximum_concurrent_workers":config["workers"],
        "software":config["software"],"elapsed_batch_wall_sec":elapsed,
        "table24":table24,"table_s8":s8,"failure_statistics":failure_stats,
        "stage_shares":stage_shares,"matched_mission_ratios":comparisons,
        "maximum_single_trial_record":max(successful_trials,key=lambda r:r["total_plan_wall_sec"]),
        "maximum_median_ratio_pair":max(successful,key=lambda p:p["planning_to_same_refined_T_pct"]),
        "repeat_range_over_median_pct":describe(p["total_repeat_range_over_median_pct"] for p in successful),
        "failed_at_5N_count":sum(p["raw_steps"]==5*p["active_cells"] for p in failed),
        "failure_tier_counts":dict(Counter(p["tier"] for p in failed))}
    (out/"analysis_results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:result[k] for k in ("audit_passed","trials","successful_pairs","failed_pairs",
          "common_success_by_domain","table24","matched_mission_ratios","repeat_range_over_median_pct",
          "failure_tier_counts")},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
