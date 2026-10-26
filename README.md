# Single-UAV Coverage Path Planning with Priority-Weighted First-Visit Latency and Completion Time

Reproduction materials for the manuscript

> Oh, Y.; Song, C.; Oh, J.; Lee, I. *Single-UAV Coverage Path Planning with Priority-Weighted First-Visit Latency and Completion Time.* Submitted to **Drones** (MDPI), 2026.

Department of Geoinformatics, University of Seoul. Corresponding author: Impyeong Lee (iplee@uos.ac.kr).

The repository holds the fixed evaluation maps, the six final policy checkpoints, the per-map evaluation
outputs behind the result tables, and the experiment packages that produced them. A Zenodo DOI is
minted from each GitHub release; cite the version you used (see `CITATION.cff`).

## What is in the repository

| Path | Contents |
|---|---|
| `data/synthetic/test/` | 720 synthetic priority maps (18 spatial patterns × 5 density tiers × 8 maps), `uint8` 50×50 arrays with priorities 0–5. Folder layout: `<pattern>/tier<k>/all_weight_map/*.npy`. |
| `data/osm_derived/test/real/` | 500 OpenStreetMap-derived proxy-priority maps (Seoul), same format. See `LICENSE-DATA` for the ODbL attribution. |
| `data/manifest_portable.csv` | Map identifier → path inside this repository, with SHA-256 of each file. |
| `data/manifest_original.csv` | The manifest used by the original runs. The absolute workspace root has been replaced by `<ORIGINAL_WORKSPACE>`; everything else is unchanged. |
| `data/flight_logs/` | Twelve DJI Matrice 4E flight records (6 October 2026) of the waypoint flights behind Table 23, in a home-point-relative east–north frame: full records, SI trajectories, the route-to-record binding, the nine Table 23 routes cut from the records, and the planned waypoint routes. See `data/flight_logs/README.md`. |
| `models/{strict,relaxed}/seed{8,42,77}/final_40001536.zip` | Six final MaskablePPO checkpoints (stable-baselines3 / sb3-contrib format) after 40,001,536 environment interactions. SHA-256 values in `provenance/CHECKPOINTS.json`. |
| `results/main_seeds_8_77/` | Raw and refined per-map test results (Synthetic 720 + OSM 500) for seeds 8 and 77, Strict and Relaxed, plus training summaries. |
| `results/mask_controls_seed42/` | No-mask and Boundary-only controls (fresh 40M training, seed 42), raw test results. |
| `results/refinement_b6_common/` | Connection refinement of Relaxed (B = 6) on the common-success map set, with the per-map audit. |
| `results/TABLE_SOURCE_MAP.json` | Which files feed each table of the manuscript (Tables 11–23 of the article and Tables S1–S7 of the Supplementary Materials). |
| `src/` | Experiment packages (see below). Seed-42 per-map results that the tables also use are stored as reference inputs inside these packages. |
| `environment/` | Software and hardware of the original training machine. |
| `provenance/` | Checkpoint hashes, the list of items that were not available when this bundle was assembled, and the old → new name map. |
| `scripts/verify_bundle.py` | Checks every file against `FILES_SHA256SUMS.json`. |
| `scripts/table23_reproduce.py` | Recomputes the measured durations and path-correspondence RMS of Table 23 from the flight records, the route binding and the planned routes. |

## Experiment packages (`src/`)

| Package | Role in the paper |
|---|---|
| `suite_core/` | Core of the experiment suite: the NumPy grid environment and time model (`coverage_core.py`), the none / boundary / progress action constraints (`core_ch4.py`, `cost11_fourway_core.py`), the classical planners BCD, STC and WLG with an exact small-instance reference (`planners_ch4.py`), the common trajectory evaluator (`evaluation_ch4.py`), and the four-way MaskablePPO runner for No-mask, Boundary-only, Relaxed B = 6 and Strict (`run_cost11_fourway.py`). `agents_ch4.py` holds additional native baselines (DDQN, discrete SAC, SAC/TD3) that are not part of the paper's main comparison. Sub-folders hold the seed-42 run protocols. |
| `main_seeds_8_77/` | Strict (B = 0) and Relaxed (B = 6) training to 40M steps for seeds 8 and 77, with evaluation; `runtime/` contains the policy feature extractor (`CNN3CombinedExtractor`) needed to load the checkpoints. `ch5_vendor/test8_original.py` is the eight-planner test with the Random and Priority-Greedy selectors. |
| `refinement_relaxed_b6_40m/` | Fixed-plan connection refinement for Relaxed B = 6 and the B = 0/1/2/4/6 budget comparison. `ch5_vendor/segment_graph_original.py` is the multi-label search over (cell, heading, remaining budget) that preserves first-visit order and arrival headings. |
| `strict_beta_sensitivity_40m/` | Strict 40M, β = 0 / 0.5 / 1.0, raw versus refined. |
| `kappa_sweep_5m/` | κ = 5 / 15 / 20 (and κ = 1 in `upstream/`), 5M steps, validation set of 180 maps. |
| `budget_b1_b2_b4_40m/` | Relaxed with B = 1, 2, 4 at 40M steps. |
| `relaxed_b6_resume_20m_to_40m/` | Continuation of the seed-42 Relaxed B = 6 policy from 20M to 40M steps. |
| `mask_controls_40m/` | No-mask and Boundary-only controls. |

Each package keeps its original Korean run notes (`README_KO.md`) and a `PACKAGE_SHA256.json` listing its files.

## Main configuration

| Item | Value |
|---|---|
| Grid | 50 × 50 cells, priorities 0–5; cell size 16 m for synthetic maps and 15–20 m for OSM-derived maps (encoded in the file name) |
| Time model | straight 11.462 m/s, turn 5.67 m/s (representative speeds from DJI Matrice 4E logs) |
| Reward | `NRG1_C3NT_COST11`: κ = 11, β = 0.5, new-visit reward 1/N, completion reward +2 |
| Policy | MaskablePPO, CNN encoder 3×3 / stride 2 / padding 1: 6→32→64→64, 256-d map feature + 4-d heading + budget scalar = 261 |
| Progress rules | Strict (B = 0) and Relaxed (B = 6); No-mask and Boundary-only as controls |
| Training | 40,001,536 interactions per model; seeds 8, 42, 77 |
| Software | Python 3.10.18, stable-baselines3 2.7.0, sb3-contrib 2.7.0, PyTorch 2.7.1, Gymnasium 1.1.1, NumPy 2.2.6 (`environment/`) |

## Using the files

```python
import numpy as np
m = np.load("data/synthetic/test/single_cluster/tier3/all_weight_map/"
            "syn50_test_single_cluster_tier3_000_active40x26.npy")   # (50, 50) uint8, 0-5
```

Loading a checkpoint needs the feature-extractor class to be importable:

```python
import sys; sys.path.insert(0, "src/main_seeds_8_77/runtime/strict")
import nrg1_c3nt_cost5_features                      # registers CNN3CombinedExtractor
from sb3_contrib import MaskablePPO
model = MaskablePPO.load("models/strict/seed42/final_40001536.zip", device="cpu")
```

Integrity check of the whole tree:

```bash
python scripts/verify_bundle.py
```

## Scope and limitations

This is a materials archive, not a turnkey pipeline. Please note:

- The experiment runners were written to execute inside the original workspace (`uav_chapter4_suite_v2`)
  and refer to run folders by their original identifiers (`chapter4_*`, `chapter5_*`). Those strings in
  the protocol files are provenance records; the folders themselves are not part of this repository.
- Not included: the synthetic-map generator scripts, training logs and intermediate checkpoints, and
  the scripts that aggregate per-map CSVs into the manuscript tables (`scripts/table23_reproduce.py`
  covers Table 23, whose two measured columns are reproduced exactly).
  `provenance/PACKER_MISSING_ITEMS.json` lists the items that were unavailable when the bundle was
  assembled; the flight records, their route binding and the planned routes have since been added under
  `data/flight_logs/`.
  `results/TABLE_SOURCE_MAP.json` marks which tables are fully bound to files here.
- Flight positions are published in a local metre frame whose origin is withheld (see
  `data/flight_logs/README.md`).
- Identical result files are stored once; the table map references the retained copy.

## License

- Code: MIT (`LICENSE`).
- Synthetic maps, checkpoints and result files: CC BY 4.0 (`LICENSE-DATA`).
- OSM-derived maps: derived from OpenStreetMap data, © OpenStreetMap contributors, ODbL 1.0 (`LICENSE-DATA`).
