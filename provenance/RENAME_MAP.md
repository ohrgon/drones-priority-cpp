# Name changes made for the public release

The materials were first collected by an automated packer from the authors' working directory.
Before publication the internal names below were replaced by descriptive ones, and the same
replacement was applied inside every text file (JSON protocols, CSV, Markdown, Python), so all
cross-references stay consistent. Binary files (`.npy`, `.zip`, `.gz`) were not modified.

## Directories

| Original name | Public name |
|---|---|
| `src/project/` | `src/suite_core/` |
| `src/project/chapter4_5m_seed42/` | `src/suite_core/protocols_5m_seed42/` |
| `src/project/chapter4_COST11_STRICT_BETAS_50M_v2_seed42/` | `src/suite_core/strict_beta_runs_seed42/` |
| `src/MAIN_SEEDS_8_77/` | `src/main_seeds_8_77/` |
| `src/CH5_B6_40M_REFINED_v1/` | `src/refinement_relaxed_b6_40m/` |
| `src/CH5_STRICT40M_BETA_REFINED_v1/` | `src/strict_beta_sensitivity_40m/` |
| `src/CH5_TABLE514_KAPPA5_15_20_v1/` | `src/kappa_sweep_5m/` |
| `src/COST11_B124_BETA05_40M_FIX_NATIVE/` | `src/budget_b1_b2_b4_40m/` |
| `src/COST11_B6_BETA05_RESUME20M_TO40M_v1/` | `src/relaxed_b6_resume_20m_to_40m/` |
| `src/COST11_TABLE55_CONTROLS40M_FRESH_FINALONLY_v2/` | `src/mask_controls_40m/` |
| `results/sources/root0/chapter4_MAIN_COST11_40M_seed8_77/` | `results/main_seeds_8_77/` |
| `results/sources/root0/chapter4_COST11_CONTROLS_40M_FRESH_FINALONLY_seed42/` | `results/mask_controls_seed42/` |
| `results/sources/root0/chapter5_B6_40M_COMMON_REFINEMENT_VAL180_TEST1220_v1/` | `results/refinement_b6_common/` |
| `audit/` | `provenance/` (`MISSING.json` → `PACKER_MISSING_ITEMS.json`; the packer's own `SLIM_REPORT.json` and `SELECTED_FILES.csv` were dropped) |

## Files

| Original name | Public name |
|---|---|
| `src/COST11_B124_BETA05_40M_FIX_NATIVE/README_#Uc2e4#Ud589#Ubc29#Ubc95.md` (mangled Korean file name) | `src/budget_b1_b2_b4_40m/README_KO.md` |

## Path placeholders

The absolute root of the original workspace (a user home directory on the training server) was
replaced everywhere by `<ORIGINAL_WORKSPACE>`. The original project folder was
`<ORIGINAL_WORKSPACE>/uav_chapter4_suite_v2`.

Because of this replacement `data/manifest_original.csv` is no longer byte-identical to the manifest
used on the server. Its original SHA-256 was
`6c3853577b5a8b96178e89caa81311091b6aa17c37f6cb11dfdeb4f24a77e68a` (812,818 bytes).
`data/manifest_portable.csv` maps every map to its path in this repository and is the one to use.

## Names that were intentionally left as they were

Strings such as `chapter4_COST11_5M_to_20M_seed42`, `chapter4_COST11_B6_beta05_EXISTING20M_to40M_CP5M_VAL1M_seed42`
or `CH5_Selected9_Flight_Error_Audit` still appear inside protocol and provenance files. They are the
identifiers of upstream run folders on the original server that are **not** part of this repository;
renaming them would break the provenance record.

`PACKAGE_SHA256.json` inside each package and `FILES_SHA256SUMS.json` at the root were regenerated
after the changes above.
