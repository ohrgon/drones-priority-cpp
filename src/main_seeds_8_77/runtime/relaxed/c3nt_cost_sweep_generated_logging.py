"""Logging-only compatibility launcher for the ORIGINAL NRG1_C3NT runner.

Add this file NEXT TO run_nrg1_c3nt.py; DO NOT replace any frozen source.
Use the same original command/options, replacing only the script name.
The wrapper fixes variable episode-CSV diagnostic columns. It does not modify
core transitions, rewards, observations, truncations, PPO, or recovery files.
Existing CSV headers/bytes are not rewritten. Original optimizer-boundary
recovery remains responsible for rolling logs back to committed offsets.
A separate audit and pre-recovery log backup document the runtime logger patch.
"""
from __future__ import annotations
import csv
from datetime import datetime, timezone
import functools
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping
FIX_VERSION = 'c3nt_cost11_episode_csv_fixed_schema_sweep_v1'
C3NT_VERSION = 'nrg1_cnn3_noclock_externalcap_cost11_sweep_v1_20260926'
RUN_ID = 'ppo_relaxedB6_NRG1_C3NT_COST11_b0.5_g1_l0.95_budgetobs_seed42'
NATIVE_REASON = 'c3nt_native_termination_reason'
STEP_METADATA = ('mask_phase', 'mask_cardinality', 'remaining_priority_before', 'traversal_time_sec', 'is_new_active_cell', 'is_turn', 'is_valid_movement', 'executed_grid_action')
OPTIONAL_DIAGNOSTICS = (NATIVE_REASON,) + STEP_METADATA
SKIP_INFO_KEYS = {'terminal_observation', 'TimeLimit.truncated'}

def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda : f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def csv_header(path: Path) -> list[str] | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        fields = next(csv.reader(f), None)
    if not fields:
        return None
    require(all(fields) and len(set(fields)) == len(fields), f'Empty/duplicate CSV field names: {path}')
    return fields

def normalize_episode_row(row: Mapping[str, Any], fields: list[str] | None, path: Path) -> tuple[dict[str, Any], list[str], list[str]]:
    """Keep data values intact; reconcile only explicitly known diagnostics.

    An existing header is immutable to preserve byte-offset recovery. If it
    lacks an optional diagnostic, only that redundant diagnostic is omitted.
    A new CSV includes all known diagnostics in a deterministic field order.
    """
    require(isinstance(row, Mapping), 'Episode logger expected a mapping.')
    clean = dict(row)
    require(clean.get('nrg_version') == C3NT_VERSION, f"Logging fix is only for C3NT, found {clean.get('nrg_version')!r}: {path}")
    require(clean.get('c3nt_cost_multiplier') == 11.0, 'Wrong cost multiplier in episode row.')
    required = {'is_success', 'termination_reason', 'c3nt_terminated', 'c3nt_truncated', 'total_reward', 'steps', 'training_seed', 'timesteps', 'c3nt_cost_multiplier'}
    require(required.issubset(clean), f'Episode row is missing required fields {sorted(required - set(clean))}: {path}')
    require(not set(clean) & SKIP_INFO_KEYS, 'Vector-only keys reached the episode logger; review callback filtering.')
    if NATIVE_REASON not in clean:
        require(not bool(clean['c3nt_truncated']), 'A capped episode lacks its native reason: source differs from C3NT v1.')
        clean[NATIVE_REASON] = clean['termination_reason']
    missing_steps = [k for k in STEP_METADATA if k not in clean]
    if missing_steps:
        require(int(clean['steps']) == 0 and bool(clean['is_success']), f'Non-singleton episode lacks step diagnostics {missing_steps}: {path}')
        for key in missing_steps:
            clean[key] = None
    if fields is None:
        fields = [k for k in clean if k not in OPTIONAL_DIAGNOSTICS] + list(OPTIONAL_DIAGNOSTICS)
    else:
        fields = list(fields)
    expected = set(fields)
    removable = set(OPTIONAL_DIAGNOSTICS) - expected
    omitted = [key for key in OPTIONAL_DIAGNOSTICS if key in removable and key in clean]
    for key in omitted:
        clean.pop(key)
    (missing, extra) = (expected - set(clean), set(clean) - expected)
    require(not missing and (not extra), f'Episode CSV schema differs beyond the known C3NT diagnostics. No unknown fields were discarded.\npath={path}\nmissing={sorted(missing)}\nextra={sorted(extra)}\nKeep logs/checkpoints; provide this mismatch report for inspection.')
    return ({key: clean[key] for key in fields}, fields, omitted)

class EpisodeCSVCompatibility:
    """Delegate all file writes and flushing to the actual original CSVLog."""

    def __init__(self, original: Any, path: Path, *, verbose: bool=True):
        self._native = original
        self._path = Path(path)
        self._fields = csv_header(self._path)
        self._verbose = verbose
        self._reported: set[tuple[str, ...]] = set()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._native, name)

    def append(self, row: Mapping[str, Any]) -> Any:
        (normalized, fields, omitted) = normalize_episode_row(row, self._fields, self._path)
        self._fields = fields
        if omitted and tuple(omitted) not in self._reported:
            self._reported.add(tuple(omitted))
            if self._verbose:
                print('[CSV HEADER PRESERVED] optional diagnostics not present in old header: ' + ', '.join(omitted), flush=True)
                print('  Existing is_success / termination_reason / c3nt_truncated / reward and movement metrics remain unchanged.', flush=True)
        return self._native.append(normalized)

def finish_logger(logger: Any) -> None:
    close = getattr(logger, 'close', None)
    if callable(close):
        close()

def episode_rows_for_check(trainer: Any, spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Real frozen core: success, truncation, exact-cap success and singleton."""
    import numpy as np
    cfg = trainer.ExperimentEnvConfig(**spec['environment'])
    require(cfg.grid_size == 50, 'Expected 50x50 original C3NT config.')
    layouts = []
    for case in ('success', 'cap', 'success_at_cap', 'singleton'):
        arr = np.zeros((50, 50), dtype=np.uint8)
        arr[1, 1] = 1
        if case == 'success':
            arr[1, 2] = 5
        elif case == 'cap':
            arr[48, 48] = 5
        elif case == 'success_at_cap':
            arr[1, 11] = 5
        layouts.append((case, arr))
    rows = []
    for (case, arr) in layouts:
        core = trainer.make_core(cfg)
        core.reset(arr, start=(1, 1), heading=1, seed=42, map_name='logfix_' + case)
        info = None
        for _ in range(max(1, int(core.max_steps))):
            action = 0 if core.complete_at_reset else 1
            (_, _, terminated, truncated, info) = core.step(action)
            if terminated or truncated:
                break
        require(info is not None and (terminated or truncated), 'Schema probe did not stop.')
        require(bool(truncated) == (case == 'cap'), 'Schema probe reached wrong stop kind.')
        rows.append(dict(training_seed=42, env_rank=0, episode_global=len(rows) + 1, timesteps=8192 * (len(rows) + 1), **{k: v for (k, v) in info.items() if k not in SKIP_INFO_KEYS}))
    return rows

def verify_csv_roundtrip(native_class: Any, rows: list[dict[str, Any]], destination: Path) -> dict:
    """Exercise the ACTUAL CSVLog with new and both legacy header variants."""
    require(len(rows) >= 4, 'Need all four stop cases.')
    (success, capped) = (rows[0], rows[1])
    schema_delta = sorted(set(success) ^ set(capped))
    require(set(schema_delta) <= {NATIVE_REASON}, f'Success/cap keys differ in unexpected fields: {schema_delta}')
    trials = []
    for (kind, initial) in (('fresh', None), ('old_success_first', success), ('old_cap_first', capped)):
        path = Path(destination) / kind / 'training_episodes.csv'
        path.parent.mkdir(parents=True, exist_ok=True)
        original_header_bytes = None
        if initial is not None:
            old = native_class(path)
            old.append(dict(initial))
            finish_logger(old)
            original_header_bytes = path.read_bytes().splitlines(keepends=True)[0]
        count = int(initial is not None)
        for sequence in (rows, list(reversed(rows))):
            logger = EpisodeCSVCompatibility(native_class(path), path, verbose=False)
            for row in sequence:
                logger.append(row)
                count += 1
            finish_logger(logger)
        with path.open(encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            saved = list(reader)
            fields = reader.fieldnames or []
        require(len(saved) == count and all((None not in r for r in saved)), 'CSV row shape corrupted.')
        expected = ([initial] if initial is not None else []) + rows + list(reversed(rows))
        for (before, after) in zip(expected, saved):
            require(float(after['total_reward']) == float(before['total_reward']), 'Raw reward changed.')
            require(after['termination_reason'] == str(before['termination_reason']), 'Stop reason changed.')
            require(after['is_success'] == str(before['is_success']), 'Success changed.')
            require(after['c3nt_truncated'] == str(before['c3nt_truncated']), 'Truncation changed.')
        if original_header_bytes is not None:
            require(path.read_bytes().splitlines(keepends=True)[0] == original_header_bytes, 'Legacy header bytes changed: recovery offsets would be invalid.')
        trials.append(dict(case=kind, rows=len(saved), columns=len(fields), status='PASS'))
    path = Path(destination) / 'fresh' / 'training_episodes.csv'
    bad = dict(success)
    bad['unexpected_metric_change'] = 1
    try:
        normalize_episode_row(bad, csv_header(path), path)
    except RuntimeError as exc:
        require('unexpected_metric_change' in str(exc), 'Wrong rejection for unknown schema drift.')
    else:
        raise RuntimeError('Unknown schema change was silently ignored.')
    return dict(version=FIX_VERSION, native_csvlog_module=native_class.__module__, success_cap_key_difference=schema_delta, cases=trials, written_test_rows=sum((r['rows'] for r in trials)), unknown_schema_change_rejected=True, raw_reward_and_stop_flags_unchanged=True, existing_header_bytes_preserved=True, optimizer_updates=0)

def install(trainer, utils):
    """Only episode-log diagnostics are normalized; other CSVs use native CSVLog."""
    if getattr(trainer, '_cost5_csv_fixed', False):
        return trainer._cost5_native_csvlog
    native = trainer.CSVLog
    require(native is utils.CSVLog, 'Unexpected CSVLog substitution before COST5 setup.')

    def factory(path, *args, **kwargs):
        logger = native(path, *args, **kwargs)
        return EpisodeCSVCompatibility(logger, Path(path)) if Path(path).name == 'training_episodes.csv' else logger
    trainer.CSVLog = factory
    trainer._cost5_native_csvlog = native
    trainer._cost5_csv_fixed = True
    return native
