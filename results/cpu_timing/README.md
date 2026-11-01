# CPU planning-time measurements (Table 24 and Table S8)

Wall-clock measurements of the complete offline planning pipeline (Stage 1 policy rollout and
Stage 2 connection refinement) behind Section 5.5, Table 24 of the article and Table S8 of the
Supplementary Materials. The measurement code is the package in `src/cpu_timing/`; the aggregation
script is `scripts/table24_reproduce.py`.

## Layout

| Path | Contents |
|---|---|
| `raw/timing_full_v3_t1_r3/per_trial.csv` | All 21,960 measured trials (6 models × 1,220 Test maps × 3 repetitions): timings, success flags, path lengths, modeled T and Lw, action-array hashes, initial condition, worker slot, CPU IDs and timestamps. |
| `raw/timing_full_v3_t1_r3/trials/<model_key>.jsonl` | The same trials as written by the benchmark workers, one JSON record per line (input of the aggregation script). |
| `raw/timing_full_v3_t1_r3/per_map.csv`, `summary.csv`, `paper_table.csv`, `SUMMARY.txt` | Per-model summaries written by the benchmark itself (median / min / max over the three repetitions of each map; per-model statistics). |
| `raw/timing_full_v3_t1_r3/run_config.json`, `hardware.json`, `status.json`, `worker_events.json`, `selected_maps.csv` | Run configuration (threads, CPU IDs, worker slots, map catalogue with SHA-256), CPU topology, run state and load averages, worker start/end times, selected maps. |
| `raw/timing_full_v3_t1_r3/models/<model_key>_metadata.json` | Per-model audit of the checkpoint and native source files (SHA-256), software versions, thread settings and warm-up calls. |
| `analysis/Table24_CPU_planning_times.csv` | Table 24: pooled statistics per domain × method over the successful model–map pairs of the three training seeds. |
| `analysis/TableS8_seedwise_CPU_planning_times.csv` | Table S8: the same statistics per domain × method × training seed. |
| `analysis/per_pair_analysis.csv` | 7,320 model–map pairs: median / min / max of each timing over the three repetitions, route metrics, and the planning-time-to-modeled-T ratios. |
| `analysis/failed_pairs.csv`, `analysis/failed_attempt_statistics.csv` | The 124 failed Relaxed attempts (Synthetic, Tiers 1 and 2) and their attempt times (Table 24 note). |
| `analysis/stage_time_shares.csv` | Stage 1 / Stage 2 shares of the summed time of all successful trials (Section 5.5). |
| `analysis/planning_to_matched_mission_time.csv` | Planning time relative to the modeled completion time T of the same Refined route (Section 5.5). |
| `analysis/common_success_CPU_planning_times.csv`, `timing_by_tier.csv`, `longest_50_plans.csv`, `model_summary_all_maps.csv` | Additional views: six-model common-success set, per-tier descriptives, the 50 longest plans, per-model summaries. |
| `analysis/analysis_results.json`, `analysis/aggregation_audit.json` | All aggregated numbers in one file; membership audit (21,960 trials, 7,320 pairs, no duplicates) with the SHA-256 of the raw input files. |
| `analysis/audits/` | Independent provenance audit of the raw records: checkpoint, map and native-source hashes against this repository at commit `8e19d88`, and a per-case comparison of the recorded route metrics with the published per-map results (`AUDIT_SUMMARY.txt`). |

The benchmark's own log files (`logs/*.log`) and its lock file are not included.

## Measurement

| Item | Value (from `run_config.json`, `hardware.json`, `status.json`) |
|---|---|
| Benchmark | `src/cpu_timing/benchmark_cpu.py`, version `drones_cpu_timing_v3_original_inputs_20261031`; the SHA-256 of the five code files is recorded in `run_config.json` (`benchmark_code_sha256`) and matches `src/cpu_timing/PACKAGE_SHA256.json`. |
| Inputs | The six final checkpoints (40,001,536 interactions; SHA-256 as in `provenance/CHECKPOINTS.json`) and the 1,220 Test maps (720 Synthetic, 500 OSM), whose recorded array and file hashes match `data/manifest_portable.csv` for all 1,220 maps (`analysis/audits/map_hash_checks.csv`). The run read the original server copies of the evaluation package (`input_mode: original`); `analysis/audits/` shows that every loaded source file matches the copy in this repository, except five workspace-directory strings in `test8_original.py` whose planning functions are AST-identical. |
| Timed region | Stage 1: `evaluate_map()` of `src/main_seeds_8_77/ch5_vendor/test8_original.py` (policy rollout with mask computation, observation construction and the diagnostic instrumentation of the evaluation code). Stage 2: `common_refine()` of `src/main_seeds_8_77/ch5_common.py` (connection refinement including the replay verification of the Raw and Refined paths). Both are `time.perf_counter_ns()` wall-clock intervals around the two native calls. |
| Excluded | Model loading, map loading and hash checks, initial-condition selection, one warm-up plan per worker, and output writing. |
| Hardware | Intel Xeon Silver 4214 @ 2.20 GHz (48 logical CPUs, 24 physical cores, 2 sockets); workers pinned to CPUs 0, 1 and 2, which are distinct physical cores of socket 0. Pinning does not reserve the cores against other processes; load averages at start and end are recorded in `status.json`. |
| Threads and concurrency | One PyTorch intra-op and one inter-op thread per worker (`--threads 1`), BLAS/OpenMP thread variables set to 1; at most three workers concurrently (`--workers 3`), one model per worker slot (`model_slots` in `run_config.json`). |
| Repetitions | Three measured repetitions per model–map pair (`--repeats 3`), after one warm-up plan per worker. |
| Software | Python 3.10.18, PyTorch 2.7.1 (CPU), NumPy 2.2.6, Gymnasium 1.1.1, stable-baselines3 2.7.0, sb3-contrib 2.7.0. |
| Run | Started 2026-10-31 09:07 UTC; total wall duration 64,429 s. Run identity `6398ab41…9cbf8d7f`. |

## Aggregation (Table 24, Table S8)

1. Each model–map pair is summarized by the median of its three repetitions, separately for each timing (`raw_wall_sec` → Stage 1, `refine_validate_wall_sec` → Stage 2, `total_plan_wall_sec` → Total, `policy_action_only_wall_sec` → policy action selection).
2. Successful pairs are pooled over the three training seeds within each domain × method group (Table 24) or kept per seed (Table S8).
3. Mean, median and P95 (linear interpolation) are statistics of these per-pair medians; `total_max_single_trial_sec` is the largest individual repetition. Because the Stage 1 and Stage 2 medians are taken separately, their means need not sum to the mean total.
4. Failed Raw attempts (action limit reached; no refinement is run) are excluded from the complete-plan statistics and reported separately with their attempt time.
5. The planning-time ratio uses the total planning time and the modeled completion time T of the same Refined route of the same model–map pair.

Column names: `stage1_mean_sec`, `stage2_including_validation_mean_sec`, `total_mean_sec`, `total_median_sec`, `total_p95_sec`, `total_max_single_trial_sec` and `policy_action_selection_mean_sec` are the Table 24 / Table S8 columns in order; `n_successful_pairs / n_evaluated_pairs` is the "Successful / evaluated pairs" column.

To recompute everything in `analysis/` from the raw records (Python standard library only, no planning is rerun):

```bash
python scripts/table24_reproduce.py --input results/cpu_timing/raw/timing_full_v3_t1_r3 --out /tmp/timing_recomputed
```

## Path placeholder and file identity

As elsewhere in this repository, the absolute root of the original workspace on the measurement server was
replaced by `<ORIGINAL_WORKSPACE>` in the text files of `raw/timing_full_v3_t1_r3/` that contained it
(`run_config.json`, `selected_maps.csv` and the six `models/*_metadata.json`). Their SHA-256 values before the
replacement were:

| File | Original SHA-256 |
|---|---|
| `run_config.json` | `50fecc3db06e509d8342fbc198b78810db42d2c38b7bbfd98f62dcdab5af6241` |
| `selected_maps.csv` | `cfe05361cabfafb3619af0b228b81213dd3bea0fad26e60ca966ad82bed6436b` |
| `models/strict_seed8_metadata.json` | `9e7a7a11928e043a38379dab5deed7262e6285e9b0894b6c9b6a83cfcaeec122` |
| `models/strict_seed42_metadata.json` | `278d2282002136a020fd04fe9d0da291370baca4e5a84c8441c914d03839144d` |
| `models/strict_seed77_metadata.json` | `77f9da22ffa4d4f79cc1190efa221594a71409baeae1f6a1dd081e45588cddef` |
| `models/relaxed_seed8_metadata.json` | `7f4473dda470532ed38531694783b27a9387c4413082f4c38d5001334997a2ad` |
| `models/relaxed_seed42_metadata.json` | `b8878611872f373ed9ea0beec1f16dbb5c1ba63abd240f0b39c386b3233f5902` |
| `models/relaxed_seed77_metadata.json` | `c81fc3e33fc9f4081142e2443a110061b0b777e9c60965e1a19db6b4f0cee1df` |

All other raw files are byte-identical to the benchmark output. The complete benchmark output archive
(`timing_full_v3_t1_r3_results.zip`, including the log files) has SHA-256
`55e8c95d62c5d301ac697896bb7ee866f529aeba92760dab283a2048b7ef0fb7`. The `analysis/` files were regenerated
from the files in `raw/` with the script above; `analysis/audits/` was produced from the unmodified archive, and
its input path was likewise replaced by `<ANALYSIS_WORKSPACE>`.

These are observed offline planning times on the stated maps and hardware, including the diagnostic
instrumentation and replay verification of the evaluation code. They are not worst-case bounds and do not
establish a real-time deadline guarantee (Section 6.4 of the article).
