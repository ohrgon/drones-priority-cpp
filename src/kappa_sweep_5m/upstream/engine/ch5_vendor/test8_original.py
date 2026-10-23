#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Eight ORIGINAL-path planners on Test_syn720 + Test_osm500, with action choices.

Run in the ORIGINAL training environment, beside uav_chapter4_suite_v2:
    python -u test_8methods_action_choices.py --root .
    python -u test_8methods_action_choices.py --root . --phase check

No training, checkpoint tuning, validation rollout or connection refinement.
Uses authenticated existing COST11 snapshots, registered Strict 40M checkpoints,
original Relaxed 5M copies, and the existing native BCD/WLG implementation.
Never recursively scans experiment directories. Original inputs are read-only.
Only use trusted local SB3/PyTorch checkpoints (deserialization can execute code).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from copy import deepcopy
import csv
from dataclasses import replace
import hashlib
import html
import importlib
import importlib.metadata
import io
import itertools
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
import zipfile

sys.dont_write_bytecode = True
VERSION = 'test8_native_action_choices_v1_20261002'
DEFAULT_OUT = 'chapter5_TEST8_action_choices_B0_40M_B6_5M'
STRICT_DIR = 'strict_beta_runs_seed42'
RELAXED_DIR = 'chapter4_COST11_5M_to_20M_seed42'
STRICT_PLAN_VERSION = 'cost11_strict50m_native_v2_20261001'
STRICT_LAUNCHER_SHA = '8f1610a290be3120804e4d48c7e5b77a431f52fe0b776fec5b288514de44a351'
STRICT_CORE_VERSION = 'cnn3_noclock_cost11_strict_progress_v1_20260929'
RELAXED_CORE_VERSION = 'nrg1_cnn3_noclock_externalcap_cost11_sweep_v1_20260926'
TRAINER = 'trainers_ch4_c3nt_cost_sweep'
CORE = 'c3nt_cost_sweep_generated_core'
FEATURES = 'nrg1_c3nt_cost5_features'
LOGGER = 'c3nt_cost_sweep_generated_logging'
BASE_REFERENCE = 'ppo_progress_FULL_b0.5_c0.03_o0.006_preference4_priority_mass_seed42'
MILLION = 1_000_000
T = 'mission_time_sec'
LW = 'priority_weighted_first_visit_latency_sec'
EPS = 1e-7
INITIAL_SEED = 20260907
EVAL_SEED = 42
MOVE_TUPLES = ((-1, 0), (0, 1), (1, 0), (0, -1))
PHYSICS = dict(grid_size=50, cell_size_m=16.0, v_straight=11.462, v_turn=5.670,
               step_budget_factor=5.0, history_alpha=.96, initialization='priority_mass')
EXPECTED_COUNTS = {('synthetic', 'train'): 900, ('synthetic', 'validation'): 180,
                   ('synthetic', 'test'): 720, ('osm', 'test'): 500}
COMMON_HASHES = {
    'coverage_core.py': '866d939c45fc51e6b9471582418017c5fe6f23b144aec76f9bfd4f331fde1c9d',
    'core_ch4.py': '4874b3553e87f592e1c7e25c2e35ec78a216d4a7830a5f57029f725c5ef0d3ca',
    'core_ch4_bounded_b6.py': 'da866974297f7187063a267ce3b6332072d38e68700a4e9d44c7e4ed51ec912d',
    'full_v2_utils.py': '8a4aa1ea1adeeaac473bc38966eb9983f1826caf5d04b0cd955ffd253bbd903a',
}
PLANNER_SHA = '80a21d7693a810a646a8b297f1bb9c0f6fa79cd7b46e77ce27ac92a980092b06'
METHODS = {
    'b0_random': dict(label='B0 Random', kind='random', family='strict', beta=None, B=0, nominal_m=0),
    'b0_priority_greedy': dict(label='B0 Priority-Greedy', kind='priority_greedy', family='strict', beta=None, B=0, nominal_m=0),
    'ppo_b0_beta0_40m': dict(label='PPO B0 beta=0 40M', kind='ppo', family='strict', beta=0.0, B=0, nominal_m=40),
    'ppo_b0_beta05_40m': dict(label='PPO B0 beta=0.5 40M', kind='ppo', family='strict', beta=0.5, B=0, nominal_m=40),
    'wlg': dict(label='WLG', kind='weighted_latency_greedy', family='classical', beta=None, B=None, nominal_m=0),
    'bcd': dict(label='BCD', kind='bcd', family='classical', beta=None, B=None, nominal_m=0),
    'ppo_b6_beta05_5m': dict(label='PPO B6 beta=0.5 5M', kind='ppo', family='relaxed', beta=0.5, B=6, nominal_m=5),
    'ppo_b6_beta0_5m': dict(label='PPO B6 beta=0 5M', kind='ppo', family='relaxed', beta=0.0, B=6, nominal_m=5),
}
METRICS = [T, LW, 'path_length_m', 'turn_count', 'revisit_steps', 'zero_transit_steps',
           'priority_arrival_25_sec', 'priority_arrival_50_sec', 'priority_arrival_75_sec',
           'priority_arrival_90_sec', 'masked_entropy_mean', 'multiaction_normalized_entropy_mean']


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def safe(v):
    if isinstance(v, Path): return str(v)
    if isinstance(v, dict): return {str(k): safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [safe(x) for x in v]
    if hasattr(v, 'tolist'): return safe(v.tolist())
    if hasattr(v, 'item'): return safe(v.item())
    if isinstance(v, float) and not math.isfinite(v): return None
    return v


def digest(v):
    return hashlib.sha256(json.dumps(safe(v), ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def read(path):
    with Path(path).open(encoding='utf-8-sig') as f: return json.load(f)


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))


def text_write(path, text):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.is_symlink(), 'Symlink output refused: ' + str(path))
    tmp = path.with_name(path.name + '.writing.' + str(os.getpid()))
    try:
        with tmp.open('w', encoding='utf-8', newline='') as f:
            f.write(text); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists(): tmp.unlink()


def write(path, value):
    text_write(path, json.dumps(safe(value), ensure_ascii=False, sort_keys=True,
                               indent=2, allow_nan=False) + '\n')


def freeze(path, value):
    if Path(path).exists(): require(read(path) == safe(value), 'Frozen output differs; choose a new --out: ' + str(path))
    else: write(path, value)


def csv_write(path, rr, fields=None):
    rr = list(rr)
    fields = fields or list(dict.fromkeys(k for r in rr for k in r)) or ['status']
    buf = io.StringIO(newline=''); w = csv.DictWriter(buf, fieldnames=fields, extrasaction='raise')
    w.writeheader()
    for r in rr:
        w.writerow({k: json.dumps(safe(v), ensure_ascii=False) if isinstance(v, (dict, list, tuple)) else safe(v)
                    for k, v in r.items()})
    text_write(path, '\ufeff' + buf.getvalue())


def num(v):
    try:
        x = float(v); return x if math.isfinite(x) else None
    except (TypeError, ValueError): return None


def truth(v):
    if v is True or str(v).lower() in ('1', '1.0', 'true'): return True
    if v is False or str(v).lower() in ('0', '0.0', 'false'): return False
    raise ValueError('Invalid success flag: ' + repr(v))


def mean(vv):
    vv = [num(v) for v in vv]; vv = [v for v in vv if v is not None]
    return math.fsum(vv) / len(vv) if vv else None


def close(a, b, atol=EPS, rtol=1e-10):
    a, b = num(a), num(b)
    return a is not None and b is not None and math.isclose(a, b, abs_tol=atol, rel_tol=rtol)


def resolve(path, root):
    p = Path(path).expanduser(); return (p if p.is_absolute() else root / p).resolve()


@contextmanager
def lock(path):
    import fcntl
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as f:
        try: fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e: raise RuntimeError('Another process holds: ' + str(path)) from e
        try: yield
        finally: fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def check_physics(env):
    for k, v in PHYSICS.items():
        actual = env.get(k)
        require(actual == v if isinstance(v, str) else close(actual, v), 'Physics/initialization differs: ' + k)


def check_cost11_spec(spec, family, beta):
    env = spec['environment']; rd = spec['reward_definition']; check_physics(env)
    require(spec.get('algorithm') == 'ppo' and spec.get('seed') == 42, 'Not the requested original PPO seed42')
    require(close(spec.get('beta'), beta) and close(env.get('tradeoff_beta'), beta), 'Wrong PPO beta')
    require(env.get('mask_mode') == ('progress' if family == 'strict' else 'relaxed'), 'Wrong native mask family')
    require(env.get('detour_budget_max') == 6, 'Unexpected observation budget capacity; do not change B0 checkpoint input')
    require(close(rd.get('cost_multiplier'), 11) and close(rd.get('success_terminal'), 2) and
            close(rd.get('failure_terminal'), 0) and rd.get('first_visit') == 'delta_new_active_count / active_cell_count',
            'Not FULL replacement COST11 reward')
    for k in ('coverage_reward_scale', 'objective_cost_scale', 'success_terminal_reward', 'failure_terminal_penalty'):
        require(close(env.get(k), 0), 'Legacy reward must be disabled for native COST11: ' + k)
    require(spec.get('mission_time_observation') is False and spec.get('remaining_step_observation') is False,
            'Unexpected clock input')


def validate_manifest(path):
    rr = rows(path)
    require(Counter((r['source'], r['split']) for r in rr) == EXPECTED_COUNTS,
            'Manifest must be Train900 / Validation180 / Test_syn720 / Test_osm500. Never use the backup split.')
    for field in ('map_id', 'sha256', 'path'):
        require(len({r[field] for r in rr}) == len(rr), 'Duplicate/cross-split manifest identity: ' + field)
    for split, per_stratum in (('train', 10), ('validation', 2), ('test', 8)):
        syn = [r for r in rr if r['source'] == 'synthetic' and r['split'] == split]
        strata = Counter((r['pattern'], r['tier']) for r in syn)
        require(len({r['pattern'] for r in syn}) == 18 and len({r['tier'] for r in syn}) == 5 and
                len(strata) == 90 and set(strata.values()) == {per_stratum},
                'Synthetic pattern/tier split differs: ' + split)
    test = [r for r in rr if r['split'] == 'test']
    require(Counter(r['source'] for r in test) == {'synthetic': 720, 'osm': 500}, 'Wrong test split')
    require(all(r['split'] == 'test' for r in test), 'Validation must not enter test')
    return rr, sorted(test, key=lambda r: (0 if r['source'] == 'synthetic' else 1, str(r['tier']), r['pattern'], r['map_id']))


def validate_runtime_sources(sources):
    for name, info in sources.items():
        require(Path(name).name == name and name.endswith('.py'), 'Unsafe runtime basename')
        p = Path(info['path'])
        require(p.is_file() and sha(p) == info['sha256'], 'Frozen native runtime missing/changed: ' + str(p))
    for name, expected in COMMON_HASHES.items():
        if name in sources:
            require(sources[name]['sha256'] == expected, 'Not the inspected original base core: ' + name)


def snapshot_sources(folder, hashes):
    out = {}
    for name, expected in hashes.items():
        if name.endswith('.py'):
            out[name] = dict(path=str((Path(folder) / name).resolve()), sha256=expected)
    required = {x + '.py' for x in (TRAINER, CORE, LOGGER, FEATURES, 'coverage_core', 'core_ch4',
                                   'core_ch4_bounded_b6', 'full_v2_utils', 'resume_ch4')}
    require(required <= set(out), 'Incomplete authenticated COST11 runtime: ' + repr(sorted(required - set(out))))
    validate_runtime_sources(out)
    return out


def registered_path(run, cp):
    name = Path(str(cp['path']).replace('\\', '/')).name
    if cp.get('final'): return run / 'last_model.zip'
    a, b = run / 'checkpoints' / name, run / name
    return a if a.is_file() or not b.is_file() else b


def choose_strict40(job, case):
    """Use registry/history only, not mtime/best score/latest_model fallback."""
    protocol = read(case / 'continuation_protocol.json')
    run = case / 'runs' / protocol['spec']['run_id']
    registered = read(run / 'checkpoints.json')
    entries = {}
    for cp in job.get('history', []): entries[int(cp['num_timesteps'])] = dict(cp)
    for cp in registered:
        if cp.get('source_baseline'): continue
        step = int(cp['num_timesteps'])
        if step not in entries:
            entries[step] = dict(cp, path=str(registered_path(run, cp).resolve()), origin='continuation')
    block = int(job['settings']['num_envs']) * int(job['settings']['ppo']['n_steps'])
    require(0 < block < MILLION, 'Invalid optimizer-boundary checkpoint window')
    matches = [v for step, v in entries.items() if 40*MILLION <= step < 40*MILLION + block]
    require(len(matches) == 1, f'Need one REGISTERED 40M checkpoint for {case.name}; found {len(matches)}. No last_model substitution.')
    cp = matches[0]
    cp.update(nominal_m=40, optimizer_block=block, selection='fixed_40M_first_completed_update_not_test_selected')
    check_checkpoint(cp, job['settings'])
    return cp, protocol


def check_checkpoint(cp, settings):
    p = Path(cp['path']); require(p.is_file(), 'Registered checkpoint missing: ' + str(p))
    require(sha(p) == cp['sha256'], 'Checkpoint hash mismatch: ' + str(p))
    with zipfile.ZipFile(p) as z:
        require({'data', 'policy.pth'} <= set(z.namelist()), 'Not an SB3 policy checkpoint')
        d = json.loads(z.read('data'))
    require(int(d['num_timesteps']) == int(cp['num_timesteps']), 'Checkpoint JSON/registry step mismatch')
    require(d.get('seed') == 42 and d.get('n_envs') == settings['num_envs'], 'Checkpoint seed/env count differs')
    for k in ('n_steps', 'batch_size', 'n_epochs', 'gamma', 'gae_lambda', 'ent_coef', 'vf_coef', 'max_grad_norm'):
        require(close(d.get(k), settings['ppo'][k]), 'Checkpoint/native PPO setting differs: ' + k)


def strict_inputs(root, strict_root, manifest_sha):
    pp = strict_root / 'plan.json'; plan = read(pp)
    require(plan.get('version') == STRICT_PLAN_VERSION and plan.get('launcher_sha256') == STRICT_LAUNCHER_SHA,
            'Strict plan is not from the inspected run_strict_50m.py revision')
    require(Path(plan['root']).resolve() == root and Path(plan['out']).resolve() == strict_root,
            'Root/Strict output differs from the original training plan')
    output = {}
    for key, beta in [('strict_b0', 0.0), ('strict_b05', .5)]:
        job = plan['jobs'][key]; case = Path(job['case_dir']).resolve()
        require(case == strict_root / key, 'Wrong Strict case directory')
        require(job['manifest_sha256'] == manifest_sha and sha(job['manifest']) == manifest_sha,
                'Strict source manifest differs from canonical manifest')
        check_cost11_spec(job['spec'], 'strict', beta)
        cp, proto = choose_strict40(job, case)
        require(proto['spec']['environment'] == job['spec']['environment'], 'Strict continuation changed environment')
        require(proto['runtime_hashes'] == job['runtime_hashes'], 'Strict runtime protocol differs from source plan')
        src = snapshot_sources(case / 'source_snapshot', job['runtime_hashes'])
        output[beta] = dict(spec=job['spec'], settings=job['settings'], software=job['software'], checkpoint=cp,
                            sources=src, runtime_kind='strict', native_version=STRICT_CORE_VERSION,
                            software_evidence=str(pp)+'::jobs.'+key+'.software',
                            protocol_sources={str(pp): sha(pp), str(case / 'continuation_protocol.json'): sha(case / 'continuation_protocol.json')},
                            strict50m_source_sha256=proto['spec']['strict50m']['source_sha256'])
    return output


def relaxed_inputs(root, relaxed_root, manifest_sha):
    output = {}
    for key, beta in [('relaxed_b0', 0.0), ('relaxed_b05', .5)]:
        case = relaxed_root / key; pp = case / 'continuation_protocol.json'; proto = read(pp)
        require(proto.get('version') == 'cost11_three_5m_to_20m_v1_20260930', 'Unexpected Relaxed continuation version')
        require(proto['source_manifest_sha256'] == manifest_sha and sha(case / 'manifest.csv') == manifest_sha,
                'Relaxed original source split differs from canonical manifest')
        spec, settings = proto['request']['spec'], proto['request']['settings']
        check_cost11_spec(spec, 'relaxed', beta)
        origin = spec['continuation']; step = int(origin['source_step'])
        block = settings['num_envs'] * settings['ppo']['n_steps']
        require(5*MILLION <= step < 5*MILLION+block, 'source_5m is not a registered nominal 5M checkpoint')
        cp = dict(path=str((case / 'source_5m' / 'last_model.zip').resolve()), sha256=proto['source_model_sha256'],
                  num_timesteps=step, nominal_m=5, optimizer_block=block,
                  source_run_id=origin['source_run_id'], selection='authenticated_original_source_5m_not_continued_last_model')
        require(origin['source_model_sha256'] == cp['sha256'], 'Relaxed source checkpoint identity conflict')
        check_checkpoint(cp, settings)
        src = snapshot_sources(case / 'source_snapshot', proto['snapshot_hash'])
        # The original continuation copied this runtime and did not alter the environment/CNN.
        software_path = case / 'runs' / spec['run_id'] / 'software.json'
        software = read(software_path) if software_path.is_file() else None
        if software is None:
            software_path = root/'protocols_5m_seed42'/'completed_model_test_full'/'test_protocol.json'
            software = read(software_path)['evaluation_versions']
            software_evidence = str(software_path)+'::evaluation_versions (recorded evaluation runtime)'
        else:
            software_evidence = str(software_path)+' (continuation runtime metadata)'
        output[beta] = dict(spec=spec, settings=settings, software=software, software_evidence=software_evidence,
                            checkpoint=cp, sources=src,
                            runtime_kind='relaxed', native_version=RELAXED_CORE_VERSION,
                            protocol_sources={str(pp): sha(pp)})
    return output


def find_verified_source(name, expected, candidates):
    for folder in dict.fromkeys(Path(p).resolve() for p in candidates):
        p = folder / name
        if p.is_file() and sha(p) == expected:
            return dict(path=str(p), sha256=expected)
    raise RuntimeError(f'No authenticated {name}; expected SHA={expected}. Checked exact directories: {candidates}')


def classical_inputs(root, manifest_sha, strict_input, replay):
    base = root / 'protocols_5m_seed42'; report = base / 'completed_model_test_full'
    ep = report / 'test_protocol.json'; p = read(ep)
    cp = report / 'classical_priority_planners_protocol.json'; c = read(cp)
    rp = base / 'runs' / c['reference_run'] / 'run_spec.json'; spec = read(rp)
    require(p['manifest_sha256'] == manifest_sha, 'Classical test source uses a different manifest')
    require(digest(spec) == c['reference_spec_sha256'], 'Classical reference environment/spec changed')
    require(c['initial_seed'] == INITIAL_SEED and c['evaluation_seed'] == EVAL_SEED and
            c['initialization'] == 'priority_mass', 'Classical initial conditions differ')
    check_physics(spec['environment'])
    require(p['runtime_sha256'].get('planners_ch4.py') == PLANNER_SHA, 'Different BCD/WLG implementation')
    sources = {}
    for name, expected in p['runtime_sha256'].items():
        sources[name] = find_verified_source(name, expected, [root, base / 'source_snapshot'])
    # The native, side-effect-free Strict-mask query is used for diagnostics only.
    sources['core_ch4_bounded_b6.py'] = strict_input['sources']['core_ch4_bounded_b6.py']
    validate_runtime_sources(sources)
    return dict(spec=spec, settings=None, software=p['evaluation_versions'], checkpoint=None, sources=sources,
                runtime_kind='classical', native_version=None, software_evidence=str(ep)+'::evaluation_versions',
                protocol_sources={str(ep): sha(ep), str(cp): sha(cp), str(rp): sha(rp)},
                legacy_report=str(report) if replay else None)


def runtime_identity(inp):
    return digest({n: x['sha256'] for n, x in inp['sources'].items()})[:20]


def copy_runtime(inp, out):
    folder = out / '_runtime' / runtime_identity(inp)
    folder.mkdir(parents=True, exist_ok=True)
    for name, src in inp['sources'].items():
        dst = folder / name
        if dst.exists(): require(sha(dst) == src['sha256'], 'Private native runtime changed: ' + str(dst))
        else:
            tmp = dst.with_name(dst.name + '.copying.' + str(os.getpid()))
            shutil.copyfile(src['path'], tmp)
            require(sha(tmp) == src['sha256'], 'Source changed during snapshot copy')
            os.replace(tmp, dst)
    return str(folder.resolve())


def build_plan(args):
    root, out = args.root, args.out
    manifest = root / 'protocols_5m_seed42' / 'manifest.csv'
    rr, tests = validate_manifest(manifest); mh = sha(manifest)
    strict = strict_inputs(root, args.strict_root, mh)
    relaxed = relaxed_inputs(root, args.relaxed_root, mh)
    classical = classical_inputs(root, mh, strict[0.0], not args.no_legacy_replay)
    if args.limit_per_source is not None:
        tests = [r for source in ('synthetic', 'osm')
                 for r in [x for x in tests if x['source'] == source][:args.limit_per_source]]
    jobs = {}
    for key, meta in METHODS.items():
        inp = deepcopy(classical if meta['family'] == 'classical' else
                       relaxed[meta['beta']] if meta['family'] == 'relaxed' else
                       strict[meta['beta'] if meta['kind'] == 'ppo' else 0.0])
        if meta['kind'] != 'ppo': inp['checkpoint'] = None
        inp['snapshot'] = copy_runtime(inp, out)
        jobs[key] = dict(meta, key=key, **inp)
    env_sig = {k: PHYSICS[k] for k in PHYSICS}
    plan = dict(version=VERSION, script_sha256=sha(__file__), root=str(root), out=str(out),
                manifest=str(manifest), manifest_sha256=mh,
                full_partition_counts={s+'/'+t: n for (s,t),n in EXPECTED_COUNTS.items()},
                tests=tests, jobs=jobs, threads=args.threads, device='cpu',
                evaluation_seed=EVAL_SEED, initial_seed=INITIAL_SEED,
                physics=env_sig, smoke_test_only=args.limit_per_source is not None,
                limit_per_source=args.limit_per_source, raw_only=True, connection_refinement=False,
                evaluation_scope='Test_syn + Test_osm only; no validation',
                action_definition='pre-action native mask cardinality for EACH executed action; terminal sentinel excluded',
                classical_action_definition='boundary-feasible primitive moves; NOT internal planner search/target candidates',
                greedy_definition='max raw priority of an admissible destination; tie: straight then U,R,D,L; not immediate reward',
                random_definition='uniform among B0 native-mask actions; per-map PCG64 seed stable_seed(42,map_sha,random,0)',
                test_access_note='Existing test sets have been inspected during development; split exclusion does not restore untouched-test status.',
                comparison_note='Strict PPO 40M vs Relaxed PPO 5M is the requested configuration comparison, not a matched-training-budget mask ablation.')
    freeze(out/'plan.json', plan)
    csv_write(out/'selected_checkpoints.csv', [dict(method=k, beta=v['beta'], B=v['B'], nominal_m=v['nominal_m'],
             actual_step=v['checkpoint']['num_timesteps'], sha256=v['checkpoint']['sha256'], path=v['checkpoint']['path'],
             selection=v['checkpoint']['selection']) for k,v in jobs.items() if v['checkpoint']])
    csv_write(out/'test_manifest.csv', tests)
    for key, job in jobs.items():
        params = dict(version=VERSION, script_sha256=plan['script_sha256'], method=key,
                      method_definition={k:job[k] for k in ('kind','family','beta','B','nominal_m')},
                      environment=job['spec']['environment'], checkpoint=job['checkpoint'],
                      runtime_sha256={n:s['sha256'] for n,s in job['sources'].items()},
                      manifest_sha256=mh, test_signature=[(r['map_id'],r['sha256'],r['source'],r['split']) for r in tests],
                      evaluation_seed=EVAL_SEED, initial_seed=INITIAL_SEED, device='cpu', threads=args.threads,
                      raw_only=True, greedy_definition=plan['greedy_definition'], random_definition=plan['random_definition'],
                      legacy_report=job.get('legacy_report'))
        freeze(out/'methods'/key/'identity.json', dict(sha256=digest(params), parameters=params))
    return plan


def verify_frozen_inputs(plan, job):
    require(plan['version'] == VERSION and plan['script_sha256'] == sha(__file__), 'Code changed; do not mix caches')
    require(sha(plan['manifest']) == plan['manifest_sha256'], 'Canonical manifest changed')
    for name, spec in job['sources'].items():
        require(sha(Path(job['snapshot'])/name) == spec['sha256'], 'Private snapshot hash differs: '+name)
    if job['checkpoint']:
        require(sha(job['checkpoint']['path']) == job['checkpoint']['sha256'], 'Selected checkpoint changed')


def native_runtime(plan, job):
    """Same factory/installer as the uploaded native launcher; no generic-core fallback."""
    import numpy as np
    snapshot = Path(job['snapshot']).resolve()
    sys.path.insert(0, str(snapshot)); importlib.invalidate_caches()
    needed = ['coverage_core', 'core_ch4', 'core_ch4_bounded_b6', 'full_v2_utils']
    if job['runtime_kind'] == 'classical': needed += ['planners_ch4']
    else: needed += [TRAINER, CORE, FEATURES, LOGGER]
    mods = {}
    for name in needed:
        mod = importlib.import_module(name)
        require(Path(mod.__file__).resolve().parent == snapshot, 'Wrong native module loaded: ' + name)
        mods[name] = mod
    for name, mod in list(sys.modules.items()):
        file = getattr(mod, '__file__', None)
        if file and Path(file).suffix == '.py' and Path(file).resolve().parent == snapshot:
            require(Path(file).name in job['sources'] and sha(file) == job['sources'][Path(file).name]['sha256'],
                    'Unverified native dependency: ' + str(file))
    packages = ['numpy'] if job['runtime_kind'] == 'classical' else ['numpy','torch','gymnasium','stable-baselines3','sb3-contrib']
    for name in packages:
        require(job['software'].get(name) == importlib.metadata.version(name),
                f'Use original training environment: {name} expected {job["software"].get(name)}, got {importlib.metadata.version(name)}. No automatic upgrade.')
    if job['runtime_kind'] == 'classical':
        env = deepcopy(job['spec']['environment']); env['mask_mode'] = 'boundary'
        cfg = mods['core_ch4'].ExperimentEnvConfig(**env)
        factory = mods['core_ch4'].make_core
        native_core = None
    else:
        import torch
        torch.set_num_threads(plan['threads'])
        trainer, native_core, feature, logger = (mods[TRAINER], mods[CORE], mods[FEATURES], mods[LOGGER])
        require(native_core.VERSION == job['native_version'] and close(native_core.COST_MULTIPLIER,11) and
                close(native_core.NORMALIZATION_STEP_FACTOR,5), 'Wrong native COST11 version')
        require(logger.C3NT_VERSION == native_core.VERSION, 'Logger/native core mismatch')
        logger.install(trainer, mods['full_v2_utils'])
        cfg = trainer.ExperimentEnvConfig(**job['spec']['environment']); factory = trainer.make_core
        require(isinstance(factory(cfg), native_core.C3NoClockMixin), 'Native replacement reward not installed')
    require(tuple(mods['coverage_core'].MOVES) == MOVE_TUPLES, 'Unexpected action index convention')
    return SimpleNamespace(cfg=cfg, factory=factory, core=native_core, modules=mods,
                           initial=mods['core_ch4'], bounded=mods['core_ch4_bounded_b6'],
                           utils=mods['full_v2_utils'], moves=MOVE_TUPLES)


def stable_seed(*items):
    # Matches the original evaluation_ch4 / run_strict_50m initialization convention.
    return int(hashlib.sha256(json.dumps(items, sort_keys=True, default=str).encode()).hexdigest()[:8],16)


def initial_condition(rt, rec, array):
    mode = 'priority_mass'
    c = rt.initial.ExperimentCore(replace(rt.cfg, initialization=mode))
    c.reset(array, seed=stable_seed('initial',rec['sha256'],mode,0,INITIAL_SEED))
    pair = hashlib.sha256(json.dumps([rec['sha256'],mode,0,INITIAL_SEED,c.start_pos,c.start_heading],sort_keys=True).encode()).hexdigest()
    return dict(start=list(c.start_pos), heading=c.start_heading, initialization=mode, pair_id=pair)


def priority_action(core, mask, moves=MOVE_TUPLES):
    """NEW requested Greedy, deliberately NOT native core.greedy_action()."""
    import numpy as np
    legal = np.flatnonzero(mask).tolist()
    require(legal, 'No B0 action available')
    r,c = core.drone_pos
    return max(legal, key=lambda a:(float(core.weight_map[r+moves[a][0],c+moves[a][1]]), int(a==core.heading), -a))


def load_model(rt, job):
    from sb3_contrib import MaskablePPO
    cp = job['checkpoint']
    require(cp is not None and sha(cp['path']) == cp['sha256'], 'Missing/changed selected policy')
    model = MaskablePPO.load(cp['path'], device='cpu')
    require(model.num_timesteps == int(cp['num_timesteps']), 'Loaded policy has wrong number of training steps')
    for k in ('n_steps','batch_size','n_epochs','gamma','gae_lambda','ent_coef','vf_coef','max_grad_norm'):
        require(close(getattr(model,k),job['settings']['ppo'][k]), 'Loaded PPO setting mismatch: '+k)
    require(model.seed == 42 and model.n_envs == job['settings']['num_envs'], 'Loaded PPO seed/env mismatch')
    if job['family']=='strict':
        require(getattr(model,'strict50m_source_sha256',None)==job['strict50m_source_sha256'], 'Strict 40M continuation provenance mismatch')
    rt.core.audit_model(model); rt.modules[FEATURES].audit_cnn3_model(model)
    model.policy.set_training_mode(False)
    return model


def policy_action(model, obs, mask):
    import torch
    import numpy as np
    with torch.no_grad():
        ob,_=model.policy.obs_to_tensor(obs)
        dist=model.policy.get_distribution(ob, action_masks=mask.reshape(1,-1))
        p=dist.distribution.probs[0].detach().cpu().numpy().astype(float)
    require(p.shape==(4,) and np.isfinite(p).all() and abs(p.sum()-1)<1e-5 and
            (p>=0).all() and np.all(p[~mask]<1e-6), 'Invalid native masked policy distribution')
    action=int(np.argmax(p)); require(mask[action], 'Argmax chose a masked action')
    return action,p


def safe_folder(value):
    return re.sub(r'[^A-Za-z0-9_.-]', '_', value)[:100]+'_'+hashlib.sha256(value.encode()).hexdigest()[:12]


def legacy_run_id(job):
    return ('BCD' if job['kind']=='bcd' else 'WeightedLatencyGreedy')+'_model_free_priority_mass_seed42'


def legacy_header(job):
    """Only known exact BCD/WLG cache paths. No model-directory recursion."""
    if not job.get('legacy_report'): return None
    folder=Path(job['legacy_report'])/'baselines'/legacy_run_id(job)
    p=folder/'baseline_identity.json'
    if not p.is_file(): return None
    d=read(p); par=d['parameters']
    require(d.get('sha256')==digest(par), 'Legacy baseline identity digest failed')
    require(par.get('planner_key')==job['kind'] and par.get('run_id')==legacy_run_id(job), 'Wrong legacy planner')
    require(par.get('environment')==job['spec']['environment'] and par.get('initial_seed')==INITIAL_SEED and
            par.get('evaluation_seed')==EVAL_SEED and par.get('initialization')=='priority_mass', 'Legacy environment/initialization differs')
    return dict(folder=folder, identity=d, maps={x[0]:x[1] for x in par['maps']})


def legacy_path(header, rec, ic):
    if header is None: return None
    import numpy as np
    require(header['maps'].get(rec['map_id'])==rec['sha256'], 'Legacy map differs')
    folder=header['folder']/'maps'/safe_folder(rec['map_id'])
    needed=['trajectory.npz','metrics.json','initial_condition.json','map_complete.json','baseline_result_identity.json']
    if not all((folder/n).is_file() for n in needed): return None
    complete=read(folder/'map_complete.json')
    required={'trajectory.npz','first_visits.csv','initial_condition.json','metrics.json'}
    require(required<=set(complete.get('files',{})), 'Legacy commit missing artifact hashes')
    for n in required:
        require((folder/n).is_file() and sha(folder/n)==complete['files'][n], 'Legacy artifact hash mismatch: '+str(folder/n))
    ident=read(folder/'baseline_result_identity.json')
    expected=digest(dict(run_identity_sha256=header['identity']['sha256'],map_id=rec['map_id'],map_sha256=rec['sha256']))
    require(ident['sha256']==expected, 'Legacy map identity mismatch')
    old_ic=read(folder/'initial_condition.json')
    require(old_ic==safe(ic), 'Legacy initial condition differs')
    metric=read(folder/'metrics.json')
    require(metric['map_id']==rec['map_id'] and metric['map_sha256']==rec['sha256'] and metric['initial_pair_id']==ic['pair_id'], 'Legacy row identity mismatch')
    with np.load(folder/'trajectory.npz',allow_pickle=False) as z:
        tr={n:z[n].copy() for n in ('actions','positions','time_sec','pre_action_masks','first_visit_time_sec')}
    n=len(tr['actions'])
    require(tr['actions'].shape==(n,) and tr['positions'].shape==(n+1,2) and
            tr['pre_action_masks'].shape==(n,4) and tr['time_sec'].shape==(n+1,), 'Legacy trajectory shapes differ')
    require(tr['actions'].dtype.kind in 'iu' and np.all((tr['actions']>=0)&(tr['actions']<4)), 'Invalid stored primitive actions')
    return dict(**tr, metric=metric, folder=str(folder), trajectory_sha256=complete['files']['trajectory.npz'])


def hist_stats(hist, prefix):
    require(len(hist)==5 and all(num(x) is not None and float(x)==int(x) and int(x)>=0 for x in hist),
            'Histogram must contain five nonnegative integer counts')
    h=[int(x) for x in hist]; require(h[0]==0, 'Histogram contains an empty nonterminal mask')
    n=sum(h); total=sum(k*h[k] for k in range(1,5))
    out={prefix+'_k'+str(k)+'_count':h[k] for k in range(1,5)}
    out.update({prefix+'_sum':total, prefix+'_mean':total/n if n else None,
                prefix+'_single_ratio':h[1]/n if n else None,
                prefix+'_multi_ratio':sum(h[2:])/n if n else None})
    return out


def evaluate_map(rt, job, model, rec, array, ic, old=None):
    """Instrument the native environment; never count API calls or terminal sentinels."""
    import numpy as np
    core=rt.factory(rt.cfg)
    obs,_=core.reset(array,start=tuple(ic['start']),heading=int(ic['heading']),seed=EVAL_SEED,map_name=rec['map_id'])
    rng=np.random.default_rng(stable_seed(EVAL_SEED,rec['sha256'],'random',0))
    planner=None
    if job['family']=='classical' and old is None:
        planner=rt.modules['planners_ch4'].ClassicalPlanner(job['kind'],None)
    hist=np.zeros(5,np.int64); bhist=np.zeros(5,np.int64); shist=np.zeros(5,np.int64)
    trace=dict(actions=[],positions=[list(core.drone_pos)],time_sec=[float(core.elapsed_time_sec)],
               pre_action_masks=[],boundary_masks=[],strict_masks=[],strict_phase=[],budget_before=[],
               priorities_before=[],destination_priority=[],heading_before=[],entropy=[],probabilities=[])
    phase_hist=defaultdict(lambda:np.zeros(5,np.int64)); ent=[]; norm=[]
    error=None; chosen_strict=0; budget_open_steps=0; compute=0.0
    if core.complete_at_reset and job['family']!='classical':
        # Native COST11 singleton reward acknowledgment: ZERO executed movements.
        core.step(0)
    while not (core._done or core.complete_at_reset):
        step=len(trace['actions'])
        if old is not None and step>=len(old['actions']):
            error=old['metric'].get('planner_error') or 'stored_planner_exhausted'
            break
        mask=np.asarray(core.action_masks(),dtype=bool).copy()
        boundary=np.asarray(core.feasible_mask(),dtype=bool).copy()
        # Query the ORIGINAL native no-cache routine; same state, no transition.
        strict,phase=rt.bounded.ExperimentCore.strict_mask_details_no_cache(core)
        strict=np.asarray(strict,dtype=bool).copy()
        require(mask.shape==boundary.shape==strict.shape==(4,) and mask.any() and strict.any(), 'Invalid nonterminal native mask')
        require(not np.any(mask & ~boundary) and not np.any(strict & ~boundary), 'Mask exceeds physical boundary')
        budget=int(getattr(core,'detour_budget',0)) if job['family']!='classical' else -1
        if job['family']=='strict':
            require(np.array_equal(mask,strict) and budget==0, 'B0 must equal native Strict, with zero effective budget')
        elif job['family']=='relaxed':
            require(0<=budget<=6 and np.array_equal(mask,boundary if budget>0 else strict), 'Native B6 mask/budget mismatch')
        else: require(np.array_equal(mask,boundary), 'BCD/WLG must not be filtered by B0/B6')
        before_pos=tuple(core.drone_pos); before_h=int(core.heading); before_t=float(core.elapsed_time_sec)
        before_priority=1-float(core.remaining_priority_mass)/float(core.total_priority_mass)
        tick=time.perf_counter(); p=None
        if old is not None:
            action=int(old['actions'][step])
            require(np.array_equal(mask,old['pre_action_masks'][step]) and
                    np.array_equal(before_pos,old['positions'][step]) and close(before_t,old['time_sec'][step]), 'Legacy replay pre-state/mask mismatch')
        elif job['kind']=='random':
            p=mask.astype(float)/int(mask.sum()); action=int(rng.choice(4,p=p))
        elif job['kind']=='priority_greedy':
            action=priority_action(core,mask); p=np.eye(4)[action]
        elif job['kind']=='ppo': action,p=policy_action(model,obs,mask)
        else:
            action=planner.act(core)
            if action is None: error='planner_exhausted'; break
            require(not isinstance(action,bool) and isinstance(action,(int,np.integer)), 'Planner returned non-integer action')
            action=int(action)
        compute+=time.perf_counter()-tick
        require(action in range(4) and mask[action], 'Algorithm chose an inadmissible primitive move')
        k=int(mask.sum()); kb=int(boundary.sum()); ks=int(strict.sum())
        hist[k]+=1; bhist[kb]+=1; shist[ks]+=1; phase_hist[phase][k]+=1
        chosen_strict+=int(strict[action]); budget_open_steps+=int(budget>0)
        h=None
        if p is not None:
            p=np.asarray(p,float)
            require(p.shape==(4,) and np.isfinite(p).all() and np.all(p>=0) and abs(float(p.sum())-1)<1e-5 and np.all(p[~mask]<1e-6), 'Invalid behavior probabilities')
            h=-float(np.sum(p[p>0]*np.log(p[p>0]))); ent.append(h)
            if k>1: norm.append(h/math.log(k))
        nr,nc=before_pos[0]+rt.moves[action][0],before_pos[1]+rt.moves[action][1]
        trace['actions'].append(action); trace['pre_action_masks'].append(mask)
        trace['boundary_masks'].append(boundary); trace['strict_masks'].append(strict)
        trace['strict_phase'].append(phase); trace['budget_before'].append(budget)
        trace['priorities_before'].append(before_priority); trace['destination_priority'].append(int(core.weight_map[nr,nc]))
        trace['heading_before'].append(before_h); trace['entropy'].append(float('nan') if h is None else h)
        trace['probabilities'].append(np.full(4,np.nan) if p is None else p)
        obs,reward,terminated,truncated,info=core.step(action)
        require(not(terminated and truncated) and len(trace['actions'])<=int(core.max_steps), 'Native step termination/cap inconsistency')
        trace['positions'].append(list(core.drone_pos)); trace['time_sec'].append(float(core.elapsed_time_sec))
        require(tuple(core.drone_pos)==(nr,nc), 'Native move differs from selected action')
        if old is not None:
            require(np.array_equal(core.drone_pos,old['positions'][step+1]) and close(core.elapsed_time_sec,old['time_sec'][step+1]), 'Legacy replay transition mismatch')
    core.assert_consistent(); row=safe(core.info(final=True)); n=len(trace['actions'])
    require(int(row['steps'])==n==sum(hist) and np.array_equal(np.asarray(core.mask_counts),hist), 'Action-count instrumentation/native histogram mismatch')
    require(int(row['invalid_action_count'])==0 and int(row['max_steps'])==math.ceil(5*int(row['active_cells'])), 'Wrong physics/movement cap')
    require(close(row['time_straight_sec'],16/11.462) and close(row['time_turn_sec'],16/5.670), 'Physical move time changed')
    if job['family']!='classical':
        require(row.get('nrg_version')==job['native_version'] and close(row.get('c3nt_cost_multiplier'),11) and
                close(row.get('c3nt_normalization_step_factor'),5), 'Not native COST11 result')
        require(num(row.get('nrg_return_identity_error')) is not None and abs(float(row['nrg_return_identity_error']))<1e-6,
                'Native replacement reward identity failed')
    if truth(row['is_success']):
        require(close(row['coverage_pct'],100) and num(row[LW]) is not None and 0<=float(row[LW])<=float(row[T])+EPS, 'Invalid completed-path exact metrics')
    else:
        require(num(row.get(LW)) is None, 'Unfinished-path Lw must be undefined, not a fabricated complete latency')
        if job['family']!='classical':
            require(row['termination_reason']=='external_step_cap' and n==int(row['max_steps']), 'Unexpected native failure cause')
    if job['family']=='strict': require(chosen_strict==n and close(row.get('nrg_detour_ratio'),0), 'B0 made a non-Strict move')
    if old is not None:
        require(n==len(old['actions']), 'Legacy path finished at a different step')
        for field in ('steps','is_success','coverage_pct',T,'turn_count','revisit_steps','max_steps',LW):
            a,b=num(row.get(field)),num(old['metric'].get(field))
            require((a is None and b is None) or close(a,b), 'Legacy replay metric differs: '+field)
        require(np.allclose(core.first_visit_time,old['first_visit_time_sec'],equal_nan=True,rtol=1e-10,atol=EPS), 'Legacy first-visit timings differ')
    cp=job['checkpoint']
    row.update(method=job['key'], label=job['label'], family=job['family'], beta=job['beta'],
               effective_B=job['B'], native_config_budget_capacity=getattr(rt.cfg,'detour_budget_max',None),
               nominal_m=job['nominal_m'], checkpoint_step=int(cp['num_timesteps']) if cp else 0,
               checkpoint_sha256=cp['sha256'] if cp else None,
               map_id=rec['map_id'], map_sha256=rec['sha256'], source=rec['source'], split=rec['split'],
               pattern=rec['pattern'], tier=rec['tier'], initial_pair_id=ic['pair_id'],
               start_r=ic['start'][0],start_c=ic['start'][1],start_heading=ic['heading'],
               decision_steps=n, evaluation_seed=EVAL_SEED, initial_seed=INITIAL_SEED,
               deterministic_action_selection=(job['kind']!='random'), evaluation_mode='native_legacy_path_replay' if old else 'native_rollout',
               old_trajectory_sha256=old['trajectory_sha256'] if old else None, planner_error=error,
               stage='raw_stage1_no_connection_refinement',
               action_set_kind='boundary_feasible_not_internal_planner_candidates' if job['family']=='classical' else 'native_progress_constraint_mask',
               internal_planner_candidate_count=None, native_reward_beta=rt.cfg.tradeoff_beta,
               masked_entropy_mean=mean(ent), multiaction_normalized_entropy_mean=mean(norm),
               policy_compute_time_sec=compute, chosen_strict_actions=chosen_strict,
               chosen_strict_ratio=chosen_strict/n if n else None, budget_open_steps=budget_open_steps,
               policy_mask_actions_mean=None if job['family']=='classical' else (sum(k*hist[k] for k in range(1,5))/n if n else None))
    for prefix,histogram in [('allowed_actions',hist),('boundary_actions',bhist),('strict_counterfactual_actions',shist)]:
        row.update(hist_stats(histogram,prefix))
    for phase in ('direct','progress'):
        hh=phase_hist[phase]; row['strict_phase_'+phase+'_steps']=int(sum(hh))
        row['strict_phase_'+phase+'_allowed_sum']=int(sum(k*hh[k] for k in range(1,5)))
    arrays={
        'actions':np.asarray(trace['actions'],np.int8), 'positions':np.asarray(trace['positions'],np.int16).reshape(-1,2),
        'time_sec':np.asarray(trace['time_sec'],np.float64), 'heading_before':np.asarray(trace['heading_before'],np.int8),
        'pre_action_masks':np.asarray(trace['pre_action_masks'],bool).reshape(-1,4),
        'boundary_masks':np.asarray(trace['boundary_masks'],bool).reshape(-1,4),
        'strict_masks':np.asarray(trace['strict_masks'],bool).reshape(-1,4),
        'strict_phase':np.asarray(trace['strict_phase'],dtype='<U8'), 'budget_before':np.asarray(trace['budget_before'],np.int16),
        'covered_priority_ratio_before':np.asarray(trace['priorities_before'],np.float64),
        'destination_priority':np.asarray(trace['destination_priority'],np.uint8),
        'entropy':np.asarray(trace['entropy'],np.float64), 'probabilities':np.asarray(trace['probabilities'],np.float32).reshape(-1,4),
        'allowed_action_count':np.sum(np.asarray(trace['pre_action_masks'],bool).reshape(-1,4),axis=1).astype(np.uint8),
        'boundary_action_count':np.sum(np.asarray(trace['boundary_masks'],bool).reshape(-1,4),axis=1).astype(np.uint8),
        'strict_action_count':np.sum(np.asarray(trace['strict_masks'],bool).reshape(-1,4),axis=1).astype(np.uint8),
        'priority_map':np.asarray(array).copy(), 'first_visit_time_sec':core.first_visit_time.copy(),
        'first_visit_step':core.first_visit_step.copy(),
    }
    return safe(row),arrays


def map_key(rec): return hashlib.sha256(rec['map_id'].encode()).hexdigest()


def validate_cached_row(row, identity, rec, job, ic):
    require(row['method']==job['key'] and row['map_id']==rec['map_id'] and row['map_sha256']==rec['sha256'], 'Cache method/map mismatch')
    require(row['source']==rec['source'] and row['split']=='test' and row['initial_pair_id']==ic['pair_id'], 'Cache split/initial condition mismatch')
    cp=job['checkpoint']; require(row.get('checkpoint_sha256')==(cp['sha256'] if cp else None), 'Cache checkpoint mismatch')
    require(row.get('checkpoint_step')==(int(cp['num_timesteps']) if cp else 0), 'Cache training step mismatch')
    require(int(row['decision_steps'])==int(row['steps']), 'Cached action denominator mismatch')
    for prefix in ('allowed_actions','boundary_actions','strict_counterfactual_actions'):
        hist=[0]+[int(row[prefix+'_k'+str(k)+'_count']) for k in range(1,5)]
        require(sum(hist)==int(row['decision_steps']), 'Cache action histogram mismatch')
        actual=hist_stats(hist,prefix)
        for k,v in actual.items(): require(row.get(k)==v or close(row.get(k),v), 'Cache statistic mismatch: '+k)


def save_npz(path, arrays):
    import numpy as np
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.writing.'+str(os.getpid()))
    try:
        with tmp.open('wb') as f:
            np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if tmp.exists():tmp.unlink()


def worker(plan, key):
    import numpy as np
    job=plan['jobs'][key];out=Path(plan['out']);base=out/'methods'/key
    with lock(base/'.worker.lock'):
        verify_frozen_inputs(plan,job);rt=native_runtime(plan,job)
        ident=read(base/'identity.json')['sha256'];rr=[];model=None
        already_complete=(base/'complete.json').is_file()
        if already_complete:
            done=read(base/'complete.json')
            require(done['identity']==ident and done['n_maps']==len(plan['tests']) and
                    (base/'per_map.csv').is_file() and sha(base/'per_map.csv')==done['per_map_sha256'],
                    'Completed method checksum/identity differs; original results are not overwritten')
        header=legacy_header(job) if job['family']=='classical' else None
        reused=0;replayed=0
        for i,rec in enumerate(plan['tests'],1):
            array=np.load(rec['path'],allow_pickle=False)
            require(rt.utils.array_hash(array)==rec['sha256'], 'Test map array changed: '+rec['map_id'])
            ic=initial_condition(rt,rec,array)
            dest=base/'maps'/map_key(rec);commit=dest/'result.json';trace=dest/'trace.npz'
            if commit.is_file():
                cache=read(commit);row=cache['row']
                require(cache['identity']==ident and cache['row_sha256']==digest(row) and
                        trace.is_file() and sha(trace)==cache['trace_sha256'], 'Changed/incomplete committed action cache: '+str(dest))
                validate_cached_row(row,ident,rec,job,ic);reused+=1
            else:
                old=legacy_path(header,rec,ic) if header else None
                if old is None and job['kind']=='ppo' and model is None: model=load_model(rt,job)
                row,arrays=evaluate_map(rt,job,model,rec,array,ic,old)
                save_npz(trace,arrays)
                write(commit,dict(identity=ident,row_sha256=digest(row),trace_sha256=sha(trace),row=row))
                replayed+=int(old is not None)
            rr.append(row)
            if i==1 or i%25==0 or i==len(plan['tests']):
                if not already_complete: csv_write(base/'per_map.csv',rr)
                status=dict(method=key,completed=i,expected=len(plan['tests']),success=sum(truth(r['is_success']) for r in rr),
                            cached=reused,replayed=replayed,last_map=rec['map_id'])
                write(base/'progress.json',status)
                print(f'[TEST] {key} {i}/{len(plan["tests"])} success={status["success"]} cached={reused} replay={replayed}',flush=True)
        if not already_complete:
            csv_write(base/'per_map.csv',rr)
            write(base/'complete.json',dict(identity=ident,n_maps=len(rr),per_map_sha256=sha(base/'per_map.csv')))
        print('[DONE METHOD]',key,flush=True)


def group_summary(rr, target_n, key, scope, *, time_ids=None, group=None):
    good=[r for r in rr if truth(r['is_success'])]
    timed=[r for r in good if time_ids is None or r['map_id'] in time_ids]
    statrows=rr if time_ids is None else timed
    meta=METHODS[key];n=len(rr);s=len(good)
    out=dict(scope=scope,method=key,label=meta['label'],family=meta['family'],B=meta['B'],beta=meta['beta'],nominal_m=meta['nominal_m'],
             N_target=target_n,N_evaluated=n,evaluation_complete=(n==target_n),success_S=s,
             success_pct=100*s/target_n if n==target_n and target_n else None,
             success_pct_of_evaluated=100*s/n if n else None,coverage_pct=mean(r['coverage_pct'] for r in rr),
             time_mean_n=len(timed),time_mean_population='own_success' if time_ids is None else 'all_requested_methods_common_success',
             action_mean_population='all_evaluated_prefixes_including_failures' if time_ids is None else 'same_common_success_maps',
             action_map_n=sum(int(r['decision_steps'])>0 for r in statrows),zero_decision_map_n=sum(int(r['decision_steps'])==0 for r in statrows),
             action_total_steps=sum(int(r['decision_steps']) for r in statrows),
             action_set_kind='boundary_feasible_not_internal_planner_candidates' if meta['family']=='classical' else 'native_progress_constraint_mask')
    if group:out.update(group)
    for field in METRICS:out[field]=mean(r.get(field) for r in timed)
    tv,lw=out[T],out[LW]
    out['Lw_over_T_ratio_of_means']=lw/tv if tv and lw is not None else None
    out['mean_mapwise_Lw_over_T']=mean(float(r[LW])/float(r[T]) for r in timed if num(r.get(T)) not in (None,0))
    den=out['action_total_steps']
    for prefix in ('allowed_actions','boundary_actions','strict_counterfactual_actions'):
        total=math.fsum(float(r[prefix+'_sum']) for r in statrows)
        out[prefix+'_map_mean']=mean(r[prefix+'_mean'] for r in statrows)
        out[prefix+'_pooled_step_mean']=total/den if den else None
        for k in range(1,5):
            count=sum(int(r[prefix+'_k'+str(k)+'_count']) for r in statrows)
            out[prefix+'_k'+str(k)+'_count']=count
            out[prefix+'_k'+str(k)+'_ratio']=count/den if den else None
        out[prefix+'_multiaction_step_ratio']=sum(out[prefix+'_k'+str(k)+'_count'] for k in (2,3,4))/den if den else None
    out['policy_mask_actions_map_mean']=None if meta['family']=='classical' else out['allowed_actions_map_mean']
    out['internal_planner_candidate_count']=None
    out['chosen_strict_step_ratio']=sum(int(r['chosen_strict_actions']) for r in statrows)/den if den else None
    out['budget_open_step_ratio']=sum(int(r['budget_open_steps']) for r in statrows)/den if den and meta['family']!='classical' else None
    for phase in ('direct','progress'):
        pn=sum(int(r['strict_phase_'+phase+'_steps']) for r in statrows)
        ps=sum(int(r['strict_phase_'+phase+'_allowed_sum']) for r in statrows)
        out['strict_phase_'+phase+'_steps']=pn;out['strict_phase_'+phase+'_mean_actions']=ps/pn if pn else None
    return out


def validate_pair(a,b):
    for field in ('map_id','map_sha256','initial_pair_id','source','split','pattern','tier','start_r','start_c','start_heading','active_cells','max_steps'):
        require(str(a[field])==str(b[field]), 'Paired map/initial mismatch: '+field+' '+str(a.get('map_id')))
    for field in ('time_straight_sec','time_turn_sec'):require(close(a[field],b[field]), 'Paired physics mismatch')


def sign(x): return -1 if x < -EPS else 1 if x > EPS else 0


def pair_summary(a,b,scope,ma,mb,target_n):
    ia={r['map_id']:r for r in a};ib={r['map_id']:r for r in b};ids=sorted(set(ia)&set(ib))
    events=Counter();paired=[]
    for mid in ids:
        ra,rb=ia[mid],ib[mid];validate_pair(ra,rb)
        sa,sb=truth(ra['is_success']),truth(rb['is_success'])
        events['both_success' if sa and sb else 'A_only_success' if sa else 'B_only_success' if sb else 'both_failed']+=1
        if sa and sb:
            dt=float(rb[T])-float(ra[T]);dl=float(rb[LW])-float(ra[LW]);st,sl=sign(dt),sign(dl)
            q='B_better_both' if st==-1 and sl==-1 else 'A_better_both' if st==1 and sl==1 else 'B_faster_A_lower_Lw' if st==-1 and sl==1 else 'A_faster_B_lower_Lw' if st==1 and sl==-1 else 'tie_in_one_or_both'
            events[q]+=1
            paired.append(dict(scope=scope,A=ma,B=mb,map_id=mid,source=ra['source'],tier=ra['tier'],pattern=ra['pattern'],
                               T_A=ra[T],T_B=rb[T],Lw_A=ra[LW],Lw_B=rb[LW],delta_T_B_minus_A=dt,delta_Lw_B_minus_A=dl,
                               actions_A=ra['allowed_actions_mean'],actions_B=rb['allowed_actions_mean'],
                               delta_mean_actions_B_minus_A=(float(rb['allowed_actions_mean'])-float(ra['allowed_actions_mean'])) if ra['allowed_actions_mean'] is not None and rb['allowed_actions_mean'] is not None else None,
                               quadrant=q))
    out=dict(scope=scope,A=ma,B=mb,N_target=target_n,N_both_evaluated=len(ids),n_common_success=len(paired),
             delta_definition='B minus A; seconds; negative favors B',
             mean_delta_T=mean(r['delta_T_B_minus_A'] for r in paired),mean_delta_Lw=mean(r['delta_Lw_B_minus_A'] for r in paired))
    for k in ('both_success','A_only_success','B_only_success','both_failed','B_better_both','A_better_both','B_faster_A_lower_Lw','A_faster_B_lower_Lw','tie_in_one_or_both'):
        out[k]=events[k]
    return out,paired


def load_results(plan):
    out=Path(plan['out']);allrows={};status={}
    expected={r['map_id']:r for r in plan['tests']}
    for key,job in plan['jobs'].items():
        base=out/'methods'/key;p=base/'per_map.csv'
        if not p.is_file():allrows[key]=[];status[key]='not_started';continue
        # Prefer committed JSON rows to CSV text, preserving null and bool types.
        rr=[];identity=read(base/'identity.json')['sha256']
        for rec in plan['tests']:
            path=base/'maps'/map_key(rec)/'result.json'
            if not path.exists():continue
            c=read(path);r=c['row']
            require(c['identity']==identity and digest(r)==c['row_sha256'], 'Report input identity/digest differs')
            require(r['split']=='test' and r['map_id'] in expected and r['map_sha256']==rec['sha256'], 'Non-test/changed report row')
            require(r['method']==key,'Report method mismatch')
            require((path.parent/'trace.npz').is_file() and sha(path.parent/'trace.npz')==c['trace_sha256'],
                    'Report trace missing/changed: '+str(path.parent))
            validate_cached_row(r,identity,rec,job,dict(pair_id=r['initial_pair_id']))
            rr.append(r)
        complete=base/'complete.json'
        if complete.is_file():
            c=read(complete)
            require(c['identity']==identity and c['per_map_sha256']==sha(p) and c['n_maps']==len(plan['tests'])==len(rr), 'Completed report checksum/count differs')
        allrows[key]=rr;status[key]='complete' if complete.is_file() else 'partial'
    return allrows,status


def html_table(rr,columns):
    text='<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(c)+'</th>' for c in columns)+'</tr></thead><tbody>'
    for r in rr:
        text+='<tr>'
        for c in columns:
            v=r.get(c)
            s='NR' if v is None else f'{v:,.4f}' if isinstance(v,float) else str(v)
            text+='<td>'+html.escape(s)+'</td>'
        text+='</tr>'
    return text+'</tbody></table></div>'


def report(plan):
    out=Path(plan['out']);dest=out/'report';dest.mkdir(parents=True,exist_ok=True)
    with lock(out/'.report.lock'):
        allrows,status=load_results(plan);own=[];common=[];bygroup=[];pairs=[];paired_rows=[];common_lists=[]
        scopes=[('Test_syn',{'synthetic'}),('Test_osm',{'osm'}),('Combined_Test',{'synthetic','osm'})]
        for scope,sources in scopes:
            target=[r for r in plan['tests'] if r['source'] in sources];target_n=len(target)
            groups={k:[r for r in rr if r['source'] in sources] for k,rr in allrows.items()}
            good=[{r['map_id'] for r in rr if truth(r['is_success'])} for rr in groups.values()]
            shared=set.intersection(*good) if good else set()
            common_lists += [dict(scope=scope,map_id=mid) for mid in sorted(shared)]
            for key,rr in groups.items():
                own.append(group_summary(rr,target_n,key,scope))
                common.append(group_summary(rr,target_n,key,scope,time_ids=shared))
            for ma,mb in itertools.combinations(METHODS,2):
                s,pr=pair_summary(groups[ma],groups[mb],scope,ma,mb,target_n);pairs.append(s)
                # Combined is a duplicate union, not a third independent paired sample.
                if scope!='Combined_Test':paired_rows.extend(pr)
        synth_targets=[r for r in plan['tests'] if r['source']=='synthetic']
        for dims in [('tier',),('pattern',),('tier','pattern')]:
            targets=defaultdict(list)
            for r in synth_targets:targets[tuple(r[d] for d in dims)].append(r)
            for vals,tr in sorted(targets.items()):
                ids={r['map_id'] for r in tr};extra=dict(zip(dims,vals),grouping='x'.join(dims))
                for key,rr in allrows.items():
                    sr=[r for r in rr if r['map_id'] in ids]
                    bygroup.append(group_summary(sr,len(tr),key,'Test_syn',group=extra))
        csv_write(dest/'summary_by_split.csv',own)
        csv_write(dest/'common_success_by_split.csv',common)
        csv_write(dest/'common_success_map_ids.csv',common_lists)
        action_fields=['scope','method','label','family','B','beta','nominal_m','N_target','N_evaluated','success_S','action_map_n','zero_decision_map_n','action_total_steps','action_set_kind',
                       'allowed_actions_map_mean','allowed_actions_pooled_step_mean','allowed_actions_k1_ratio','allowed_actions_k2_ratio','allowed_actions_k3_ratio','allowed_actions_k4_ratio',
                       'allowed_actions_multiaction_step_ratio','policy_mask_actions_map_mean','boundary_actions_map_mean','strict_counterfactual_actions_map_mean','chosen_strict_step_ratio']
        csv_write(dest/'action_choices_by_split.csv',[{k:r[k] for k in action_fields} for r in own])
        csv_write(dest/'tier_pattern_summary.csv',bygroup)
        csv_write(dest/'paired_T_Lw_summary.csv',pairs)
        csv_write(dest/'paired_T_Lw_per_map.csv',paired_rows)
        csv_write(dest/'all_per_map.csv',[r for key in METHODS for r in allrows[key]])
        complete=all(v=='complete' for v in status.values())
        write(dest/'report_status.json',dict(complete=complete,methods=status,n_rows=sum(len(rr) for rr in allrows.values()),
                                           expected_rows=8*len(plan['tests']),smoke_test_only=plan['smoke_test_only'],
                                           native_rollout_rows=sum(r.get('evaluation_mode')=='native_rollout' for rr in allrows.values() for r in rr),
                                           authenticated_legacy_replay_rows=sum(r.get('evaluation_mode')=='native_legacy_path_replay' for rr in allrows.values() for r in rr),
                                           stage='Raw / no connection refinement'))
        images=[];plot_error=None
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            for scope,_ in scopes:
                for field,label,data in [
                    ('allowed_actions_map_mean','Mean admissible actions (equal map weights)',own),
                    ('allowed_actions_k1_ratio','Single-action fraction (pooled executed steps)',own),
                    ('success_pct','Complete-coverage success (%)',own),
                    (T,'Mission time T: all-method common-success maps (s)',common),
                    (LW,'Exact Lw: all-method common-success maps (s)',common)]:
                    sr=[r for r in data if r['scope']==scope]
                    fig,ax=plt.subplots(figsize=(10,5))
                    vals=[float('nan') if r.get(field) is None else r[field] for r in sr]
                    ax.bar(range(len(sr)),vals)
                    ax.set_xticks(range(len(sr)),[r['label'] for r in sr],rotation=30,ha='right')
                    title=scope+(' — partial' if not complete else '')
                    if field in (T,LW): title+=' | common n='+str(sr[0]['time_mean_n'])
                    ax.set(ylabel=label,title=title)
                    if field.startswith('allowed_actions'):
                        ax.set_xlabel('BCD/WLG: boundary-feasible moves; others: progress-constraint admissibility',fontsize=8)
                    ax.grid(axis='y',alpha=.2);fig.tight_layout()
                    name=scope+'_'+field+'.png';fig.savefig(dest/name,dpi=140);plt.close(fig);images.append(name)
        except ImportError as e:plot_error=str(e)
        document='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Test8 action choices</title>
<style>body{font:15px system-ui,sans-serif;line-height:1.7;margin:30px auto;max-width:1400px;padding:0 20px}table{border-collapse:collapse;font-size:12px}th,td{border-bottom:1px solid #ddd;padding:7px;text-align:right;white-space:nowrap}th{background:#f3f3f3}.scroll{overflow:auto}img{max-width:100%}code{background:#f3f3f3;padding:2px 4px}</style>
<h1>8-method Test: B0 / B6, Raw paths and action availability</h1>'''
        document+='<p><b>'+('완료' if complete else '중간 결과 — 누락된 평가를 실패로 처리하지 않음')+('</b> · 소규모 확인용, 최종 Test가 아님' if plan['smoke_test_only'] else '</b>')+'</p>'
        scope_counts=Counter(r['source'] for r in plan['tests'])
        document+=f'<p>Train/Validation 제외. Test_syn {scope_counts["synthetic"]} 및 Test_osm {scope_counts["osm"]}을 먼저 보고하고 Combined {len(plan["tests"])}은 보조 합계로 사용합니다. 이번 실험은 모두 Raw 경로이며 Connection Refinement가 없습니다.</p>'
        document+='<p><b>액션 수 = 행동 실행 직전의 허용 이동 수</b>입니다. 최종으로 고른 한 행동의 개수나 softmax 양수 확률 개수가 아닙니다. 종료 sentinel과 초기 완주 지도는 스텝 분모에 넣지 않습니다. 실패 경로도 실제 실행한 prefix의 액션 통계에는 포함합니다.</p>'
        document+='<p>지도 균등 평균은 지도별 평균을 다시 평균합니다. 스텝 가중 평균은 모든 실행 스텝을 모아 평균합니다. 두 분모를 구분하십시오. T/Lw는 완주 지도에서만 계산하며, common 표는 8방법 공통 성공 지도입니다.</p>'
        document+='<p><b>BCD/WLG의 값은 boundary-feasible 이동 수</b>로, 내부 목표 후보 수나 탐색 branching 수가 아닙니다. policy-mask 열은 해당없음(NR), boundary 열은 실제 계측값입니다. 두 baseline에 B0/B6를 강제하지 않았습니다.</p>'
        document+='<p>B0 Greedy는 허용 이웃의 원래 priority 최댓값 → 동률이면 직진 → U,R,D,L입니다. 기존 immediate-reward Greedy와 다릅니다. Random은 지도별 고정 seed에서 허용 행동에 균등확률을 줍니다.</p>'
        document+='<p>Strict PPO 40M와 Relaxed PPO 5M는 학습량이 다릅니다. 이 결과를 mask 하나만의 인과효과로 해석하지 않습니다. 액션 수는 각 방법이 실제 방문한 상태에 조건부인 값입니다. Test는 이전 개발 과정에서 열람되었으므로 새로운 미열람 Test라고 표현하지 않습니다.</p>'
        for scope,_ in scopes:
            document+='<h2>'+scope+'</h2><h3>Action availability</h3>'
            document+=html_table([r for r in own if r['scope']==scope],['label','N_evaluated','success_S','action_total_steps','allowed_actions_map_mean','allowed_actions_pooled_step_mean','allowed_actions_k1_ratio','allowed_actions_multiaction_step_ratio','boundary_actions_map_mean','strict_counterfactual_actions_map_mean'])
            document+='<h3>Own-success T/Lw</h3>'+html_table([r for r in own if r['scope']==scope],['label','N_target','N_evaluated','success_S','success_pct','time_mean_n',T,LW,'Lw_over_T_ratio_of_means'])
            document+='<h3>All-method common-success T/Lw</h3>'+html_table([r for r in common if r['scope']==scope],['label','time_mean_n',T,LW,'allowed_actions_map_mean'])
            document+=''.join('<img src="'+n+'" alt="'+n+'">' for n in images if n.startswith(scope+'_'))
        document+='<h2>CSV</h2><p>action_choices_by_split.csv · summary_by_split.csv · common_success_by_split.csv · tier_pattern_summary.csv · paired_T_Lw_summary.csv · paired_T_Lw_per_map.csv · all_per_map.csv</p>'
        document+='<p>개별 map의 실행 전 mask·행동·위치·시간·budget은 methods/METHOD/maps/HASH/trace.npz에 있습니다. --phase export-steps --method METHOD --map-id MAP_ID로 CSV를 만들 수 있습니다.</p>'
        if plot_error:document+='<p>PNG 미생성: '+html.escape(plot_error)+'</p>'
        document+='</html>';text_write(dest/'index.html',document)
        zp=dest/'UPLOAD_TEST8_action_choices.zip';tmp=zp.with_name(zp.name+'.writing')
        with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(dest.iterdir()):
                if p.is_file() and p.suffix!='.zip' and '.writing' not in p.name:z.write(p,'report/'+p.name)
            for n in ('plan.json','selected_checkpoints.csv','test_manifest.csv'):z.write(out/n,n)
            for key in METHODS:
                for n in ('identity.json','complete.json','per_map.csv'):
                    p=out/'methods'/key/n
                    if p.is_file():z.write(p,'methods/'+key+'/'+n)
            # No policy weights or 9760 trajectory binaries in the upload summary.
        os.replace(tmp,zp)
        print('[REPORT]',dest/'index.html',flush=True);print('[UPLOAD]',zp,flush=True)


def export_steps(plan, method, map_id):
    import numpy as np
    require(method in METHODS, '--method must name one of the 8 methods')
    rec=next((r for r in plan['tests'] if r['map_id']==map_id),None)
    require(rec is not None, '--map-id must exactly match test_manifest.csv')
    folder=Path(plan['out'])/'methods'/method/'maps'/map_key(rec)
    commit=read(folder/'result.json');path=folder/'trace.npz'
    require(sha(path)==commit['trace_sha256'], 'Trace checksum mismatch')
    with np.load(path,allow_pickle=False) as z:
        rr=[]
        for i,a in enumerate(z['actions']):
            rr.append(dict(step=i+1,pre_r=int(z['positions'][i,0]),pre_c=int(z['positions'][i,1]),
                           pre_heading=int(z['heading_before'][i]),pre_time_sec=float(z['time_sec'][i]),
                           allowed_count=int(z['allowed_action_count'][i]),boundary_count=int(z['boundary_action_count'][i]),
                           strict_counterfactual_count=int(z['strict_action_count'][i]),
                           allowed_U=int(z['pre_action_masks'][i,0]),allowed_R=int(z['pre_action_masks'][i,1]),
                           allowed_D=int(z['pre_action_masks'][i,2]),allowed_L=int(z['pre_action_masks'][i,3]),
                           chosen_action=int(a),budget_before=int(z['budget_before'][i]),strict_phase=str(z['strict_phase'][i]),
                           destination_priority=int(z['destination_priority'][i]),
                           post_r=int(z['positions'][i+1,0]),post_c=int(z['positions'][i+1,1]),post_time_sec=float(z['time_sec'][i+1])))
    csv_write(folder/'steps.csv',rr);print('[EXPORTED]',folder/'steps.csv',flush=True)


def launch(plan, selected, max_workers):
    out=Path(plan['out']);logs=out/'logs';logs.mkdir(exist_ok=True)
    pending=list(selected);running=[];failed=[];last=0
    try:
        while pending or running:
            while pending and len(running)<max_workers:
                key=pending.pop(0);lp=logs/(key+'.log');f=lp.open('a',encoding='utf-8',buffering=1)
                env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='';env['MPLBACKEND']='Agg'
                for n in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[n]=str(plan['threads'])
                cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--_worker',key,'--_plan',str(out/'plan.json')]
                p=subprocess.Popen(cmd,cwd=plan['root'],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                running.append((key,p,f,lp));print(f'[LAUNCH] {key} CPU log={lp}',flush=True)
            for item in running[:]:
                key,p,f,lp=item
                if p.poll() is not None:
                    f.close();running.remove(item)
                    print(f'[EXIT] {key} code={p.returncode}',flush=True)
                    if p.returncode:failed.append(dict(method=key,code=p.returncode,log=str(lp)))
            if time.monotonic()-last>30:
                for key,p,f,lp in running:
                    sp=out/'methods'/key/'progress.json'
                    if sp.is_file():
                        s=read(sp);print(f'[PROGRESS] {key}: {s["completed"]}/{s["expected"]} success={s["success"]}',flush=True)
                last=time.monotonic()
            if running:time.sleep(1)
        write(out/'worker_status.json',dict(failed=failed,selected=selected))
        return failed
    except KeyboardInterrupt:
        for _,p,_,_ in running:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        for _,p,f,_ in running:
            p.wait();f.close()
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--out',type=Path)
    p.add_argument('--strict-root',type=Path)
    p.add_argument('--relaxed-root',type=Path)
    p.add_argument('--phase',choices=['all','check','eval','report','export-steps'],default='all')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--methods',nargs='+',choices=list(METHODS),default=list(METHODS))
    p.add_argument('--no-legacy-replay',action='store_true',help='Do not replay authenticated legacy BCD/WLG paths; recompute native planners')
    p.add_argument('--limit-per-source',type=int,help='Small diagnostic only; requires an explicit new --out, never labelled full Test')
    p.add_argument('--method',choices=list(METHODS),help='For --phase export-steps')
    p.add_argument('--map-id',help='Exact map_id from test_manifest.csv, for export-steps')
    p.add_argument('--_worker',choices=list(METHODS),help=argparse.SUPPRESS)
    p.add_argument('--_plan',type=Path,help=argparse.SUPPRESS)
    args=p.parse_args()
    if args._worker:
        require(args._plan is not None,'Worker needs a plan');worker(read(args._plan),args._worker);return 0
    args.root=args.root.expanduser().resolve();require(args.root.is_dir(),'Original suite root not found')
    args.out=resolve(args.out or DEFAULT_OUT,args.root)
    require(args.out!=args.root,'Output cannot be the source suite root')
    args.strict_root=resolve(args.strict_root or STRICT_DIR,args.root)
    args.relaxed_root=resolve(args.relaxed_root or RELAXED_DIR,args.root)
    for protected in (args.strict_root,args.relaxed_root,args.root/'protocols_5m_seed42'):
        require(args.out!=protected and protected not in args.out.parents,
                'Use a NEW output outside the original training/results directories: '+str(protected))
    require(1<=args.workers<=8 and 1<=args.threads<=32,'workers:1..8; threads:1..32')
    if args.limit_per_source is not None:
        require(args.limit_per_source>0 and '--out' in sys.argv, '--limit-per-source requires a separate explicit --out')
    if args.phase in ('report','export-steps'):
        plan=read(args.out/'plan.json');require(plan['version']==VERSION and plan['script_sha256']==sha(__file__),'Wrong report source/version')
        if args.phase=='report':report(plan)
        else:
            require(args.method is not None and args.map_id is not None,'export-steps requires --method and --map-id')
            export_steps(plan,args.method,args.map_id)
        return 0
    with lock(args.out/'.launcher.lock'):
        print('[CHECK] exact source paths / registered checkpoints / frozen runtime / canonical split',flush=True)
        plan=build_plan(args)
        print('[SCOPE]',Counter(r['source'] for r in plan['tests']),'Validation excluded',flush=True)
        for key,job in plan['jobs'].items():
            cp=job['checkpoint'];step=cp['num_timesteps'] if cp else 0
            print(f'[PLAN] {key}: B={job["B"]} beta={job["beta"]} nominal={job["nominal_m"]}M actual={step:,}',flush=True)
        print('[ACTIONS] native pre-action masks; terminal sentinels excluded; BCD/WLG=boundary-feasible, not planner search candidates',flush=True)
        if args.phase=='check':
            write(args.out/'check.json',dict(passed=True,n_test=len(plan['tests']),validation_rollouts=0,model_inference=0,checkpoints_fixed=True))
            print('[CHECK PASS] metadata/files only; no training or Test inference',flush=True);return 0
        failed=launch(plan,args.methods,args.workers)
        report(plan)
        if failed:raise RuntimeError('Some methods failed; other committed results are retained. See worker_status.json / logs: '+str(failed))
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except KeyboardInterrupt:
        print('\n[PAUSED] Re-run the same command; committed per-map results/trace caches are retained.',file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        import traceback
        traceback.print_exc();print('[ERROR]',exc,file=sys.stderr);raise SystemExit(1)
