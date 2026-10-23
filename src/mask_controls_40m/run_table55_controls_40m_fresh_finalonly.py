#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""No-mask / Boundary-only: FRESH 0->40M, final save only, Raw Test1220.

Place beside the ORIGINAL run_cost11_fourway.py in uav_chapter4_suite_v2.
  python run_table55_controls_40m_fresh_finalonly.py --root . --gpus 1 2

Both controls start from fresh independent weights under the SAME COST11/FULL/
beta=0.5/CNN3-no-clock/H5/seed42 protocol used by run_cost11_fourway.py.
No prior No-mask/Boundary model is searched or loaded. No Validation, preflight
rollout, intermediate checkpoint, recovery checkpoint, model selection, or
connection refinement. Only the final ~40M model is serialized, then the fixed
Synthetic720 and OSM500 Raw tests are evaluated. Original files remain read-only.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from contextlib import contextmanager
from copy import deepcopy
import csv
import difflib
import hashlib
import html
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time
import traceback
import zipfile

VERSION = 'table55_controls40m_fresh_finalonly_v2_20261006'
TARGET = 40_000_000
CASES = ('none', 'boundary')
LABELS = {'none': 'No-mask', 'boundary': 'Boundary-only'}
DEFAULT_OUT = 'mask_controls_seed42'
DEFAULT_SOURCE = 'chapter4_relaxedB6_C3NT_COST_SWEEP_seed42/cost11'
DEFAULT_RESUME = 'chapter4_C3NT_COST11_FOURWAY_seed42'
RUNNER_SHA = '13cc4195f52d8e4c1e9e60bb110bb21ac2c8d11c5dcfdc638fb557000f48a707'
TKEY = 'mission_time_sec'
LKEY = 'priority_weighted_first_visit_latency_sec'
PPO_KEYS = ('n_steps', 'batch_size', 'n_epochs', 'gamma', 'gae_lambda',
            'ent_coef', 'vf_coef', 'max_grad_norm', 'target_kl')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if hasattr(value, 'tolist'):
        return safe(value.tolist())
    if hasattr(value, 'item'):
        return safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def digest(value):
    return hashlib.sha256(json.dumps(safe(value), sort_keys=True,
                         separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.is_symlink(), 'Symlink output refused: ' + str(path))
    temp = path.with_name(path.name + '.writing.' + str(os.getpid()))
    try:
        with temp.open('w', encoding='utf-8', newline='') as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def write(path, value):
    write_text(path, json.dumps(safe(value), ensure_ascii=False, sort_keys=True,
                               indent=2, allow_nan=False) + '\n')


def csv_read(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def csv_write(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ['status']
    s = io.StringIO(newline='')
    w = csv.DictWriter(s, fieldnames=fields)
    w.writeheader()
    for row in rows:
        w.writerow({k: json.dumps(safe(v), ensure_ascii=False)
                    if isinstance(v, (dict, list, tuple)) else safe(v)
                    for k, v in row.items()})
    write_text(path, '\ufeff' + s.getvalue())


def canon(split):
    s = str(split).lower()
    return 'validation' if s in ('validation', 'val', 'valid') else s


def truth(value):
    if value is True or str(value).lower() in ('true', '1', '1.0'):
        return True
    if value is False or str(value).lower() in ('false', '0', '0.0'):
        return False
    raise ValueError('Invalid success flag: ' + repr(value))


def number(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def mean(values):
    v = [number(x) for x in values]
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


@contextmanager
def lock(path):
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another process owns ' + str(path)) from exc
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def resolve(root, value):
    p = Path(value).expanduser()
    return (p if p.is_absolute() else root / p).resolve()


def overlap(a, b):
    return a == b or a in b.parents or b in a.parents


def original(root):
    path = root / 'run_cost11_fourway.py'
    require(path.is_file() and sha(path) == RUNNER_SHA,
            'Expected the uploaded original run_cost11_fourway.py (do not overwrite it).')
    spec = importlib.util.spec_from_file_location('_table55_original_runner', path)
    require(spec and spec.loader, 'Cannot load original runner.')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    for name in mod.PACKAGE:
        require((root / name).is_file(), 'Missing original suite module: ' + name)
    return mod


def model_metadata(path):
    with zipfile.ZipFile(path) as z:
        require({'data', 'policy.pth', 'policy.optimizer.pth', '_stable_baselines3_version'}
                <= set(z.namelist()), 'Incomplete policy/Adam ZIP: ' + str(path))
        require(z.getinfo('data').file_size < 32_000_000, 'Oversized SB3 metadata.')
        data = json.loads(z.read('data'))
        require(z.read('_stable_baselines3_version').decode().strip() == '2.7.0',
                'Use the original SB3 2.7.0 environment.')
    return data


def inspect_resume(base, ctx, root, case, case_root):
    """Accept only the authenticated original SAME-case completed 5M run."""
    p = read(case_root / 'protocol.json')
    req = p.get('request', {})
    require(req.get('version') == base.VERSION and req.get('case') == case,
            'Not the original SAME-case COST11 run: ' + str(case_root))
    require(req.get('source_protocol_sha256') == base.sha(ctx['root'] / 'protocol.json'),
            '5M source is bound to another original COST11 protocol.')
    require(p.get('manifest_sha256') == sha(case_root / 'manifest.csv') ==
            sha(ctx['root'] / 'manifest.csv'), '5M manifest/order differs.')
    for n, h in p['snapshot_hash'].items():
        require(sha(base.child(case_root / 'source_snapshot', n)) == h,
                '5M source snapshot changed: ' + n)
    for n, h in req['package_hash'].items():
        require(n in base.PACKAGE and sha(root / n) == h,
                '5M package differs from installed original: ' + n)
    run = case_root / 'runs' / base.run_id(case)
    sp, st, status = (read(run / n) for n in
                     ('run_spec.json', 'training_settings.json', 'status.json'))
    expected_sp, expected_st, _ = base.make_plan(ctx, case)
    clean_sp = {k: v for k, v in sp.items() if k != 'nrg1_protocol_sha256'}
    require(clean_sp == expected_sp and sp.get('nrg1_protocol_sha256') == sha(case_root / 'protocol.json'),
            '5M reward/environment/CNN/spec mismatch: ' + case)
    clean_st = {k: v for k, v in st.items() if k != 'device'}
    require(clean_st == {k: v for k, v in expected_st.items() if k != 'device'},
            '5M training settings differ: ' + case)
    require(status.get('state') == 'completed',
            'Existing 5M run is incomplete. Preserve it; select --start fresh explicitly or finish the old run: ' + str(run))
    path = run / 'last_model.zip'
    regs = [c for c in read(run / 'checkpoints.json') if c.get('final')]
    require(len(regs) == 1 and regs[0]['sha256'] == sha(path), '5M final registry mismatch.')
    data = model_metadata(path)
    step = int(data['num_timesteps'])
    block = int(st['num_envs']) * int(st['ppo']['n_steps'])
    require(5_000_000 <= step < 5_000_000 + block and step == status['actual_timesteps'],
            'Expected a completed nominal 5M model; not COST5/20M/other training budget.')
    require(data['seed'] == 42 and data['n_envs'] == st['num_envs'], '5M seed/env count mismatch.')
    for k in PPO_KEYS:
        require(data[k] == st['ppo'][k], '5M PPO parameter mismatch: ' + k)
    require(data['_n_updates'] == status['ppo_epoch_counter'], '5M update counter mismatch.')
    files = [case_root / 'protocol.json', case_root / 'manifest.csv', path]
    files += [run / n for n in ('run_spec.json', 'training_settings.json', 'status.json', 'checkpoints.json')]
    files += [run / f'map_order_rank{i}.csv' for i in range(st['num_envs'])]
    require(all(f.is_file() for f in files), 'Missing 5M sampler history or metadata.')
    return dict(case_root=str(case_root), run=str(run), model=str(path), step=step,
                sha256=sha(path), episodes=int(status['episodes']),
                epoch_counter=int(status['ppo_epoch_counter']), settings=st, spec=sp,
                training_identity=status['identity'],
                protected={str(f): sha(f) for f in files})


def choose_resume(base, ctx, args, case):
    # Fresh-only runner: prior No-mask/Boundary weights are intentionally not used.
    return None


def patch_train_sb3(source):
    """Checked AST edit of the server's frozen native train_sb3, not a new PPO."""
    tree = ast.parse(source)
    candidates = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train_sb3']
    require(len(candidates) == 1, 'Native train_sb3 definition missing.')
    f = deepcopy(candidates[0])
    original_f = ast.unparse(f)
    defs = {n.name: n for n in f.body if isinstance(n, ast.FunctionDef)}
    require({'save_model', 'save_resume'} <= set(defs), 'Unrecognized native save interface.')
    adapter = [n for n in ast.walk(f) if isinstance(n, ast.ClassDef) and n.name == 'Adapter']
    require(len(adapter) == 1, 'Native Adapter changed.')
    adapter_before = ast.dump(adapter[0], include_attributes=False)
    hits = Counter()

    class Edit(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if node.name == 'save_resume':
                hits['removed_save_resume_definition'] += 1
                return None
            node = self.generic_visit(node)
            if node.name == 'save_model':
                require([a.arg for a in node.args.args] == ['model', 'final'], 'save_model signature changed.')
                node.body[:0] = ast.parse('''
if not final:
    raise RuntimeError('Intermediate model save is disabled.')
if int(model.num_timesteps) < int(settings['total_timesteps']):
    raise RuntimeError('Refusing to save a non-final model.')
''').body
                hits['final_save_guard'] += 1
            return node

        def visit_Assign(self, node):
            if any(isinstance(t, ast.Name) and t.id == 'store' for t in node.targets):
                require(isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'ResumeStore',
                        'Unexpected native store initialization.')
                node.value = ast.Constant(None)
                hits['disabled_store'] += 1
            elif ast.unparse(node.value) == 'store.load()':
                node.value = ast.Constant(None)
                hits['disabled_store_load'] += 1
            elif ast.unparse(node.value) == 'SaveAfterUpdate(**common)':
                node.value = ast.parse('_t55_build_model(SaveAfterUpdate, common, env, spec, settings, out, records)', mode='eval').body
                hits['model_constructor'] += 1
            elif any(isinstance(t, ast.Name) and t.id == 'episode_count' for t in node.targets):
                require(isinstance(node.value, ast.IfExp) and ast.unparse(node.value.test) == 'state',
                        'Unexpected episode counter expression.')
                node.value.orelse = ast.parse("settings['table55']['source_episodes']", mode='eval').body
                hits['source_episode_counter'] += 1
            return self.generic_visit(node)

        def visit_If(self, node):
            test = ast.unparse(node.test)
            if test in ('self.num_timesteps >= next_cp',
                        'self.num_timesteps >= next_resume or stop.requested'):
                hits['periodic_branch'] += 1
                return None
            node = self.generic_visit(node)
            if not node.body:
                node.body = [ast.Pass()]
            return node

        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'save_resume':
                hits['removed_save_resume_call'] += 1
                return None
            return self.generic_visit(node)

        def visit_Call(self, node):
            node = self.generic_visit(node)
            if isinstance(node.func, ast.Attribute) and node.func.attr == 'learn':
                kw = [k for k in node.keywords if k.arg == 'reset_num_timesteps']
                require(len(kw) == 1 and ast.unparse(kw[0].value) == 'not bool(state)',
                        'Unknown native learn/resume convention.')
                kw[0].value = ast.Constant(False)
                hits['keep_cumulative_steps'] += 1
            return node

    f = Edit().visit(f)
    expected = {'removed_save_resume_definition': 1, 'final_save_guard': 1,
                'disabled_store': 1, 'disabled_store_load': 1, 'model_constructor': 1,
                'source_episode_counter': 1, 'periodic_branch': 2,
                'removed_save_resume_call': 2, 'keep_cumulative_steps': 1}
    require(dict(hits) == expected, 'Native trainer structure differs; no training started: ' + str(dict(hits)))
    ast.fix_missing_locations(f)
    adapter_after = next(n for n in ast.walk(f) if isinstance(n, ast.ClassDef) and n.name == 'Adapter')
    require(ast.dump(adapter_after, include_attributes=False) == adapter_before, 'Native Adapter was modified.')
    calls = [n for n in ast.walk(f) if isinstance(n, ast.Call)]
    saves = [n for n in calls if ast.unparse(n.func) == 'save_model']
    require(len(saves) == 1 and any(k.arg == 'final' and isinstance(k.value, ast.Constant)
            and k.value.value is True for k in saves[0].keywords), 'Final save must be called exactly once.')
    require(not any(ast.unparse(n.func) in ('save_resume', 'store.commit', 'store.load') for n in calls),
            'A recovery write/load survived patching.')
    require(sum(ast.unparse(n.func) == 'super().train' for n in calls) == 1, 'Native optimizer update changed.')
    require(sum(ast.unparse(n.func) == 'model.save' for n in calls) == 1, 'Unexpected model serialization path.')
    result = ast.unparse(f) + '\n'
    compile(result, 'train_sb3_finalonly.py', 'exec')
    return result, dict(hits=hits, adapter_ast_unchanged=True,
                        optimizer_update_preserved=True, final_save_calls=1,
                        recovery_save_calls=0, source_function_sha256=digest(original_f))


def verify_protected(job):
    for p, h in job['protected'].items():
        require(Path(p).is_file() and sha(p) == h, 'Protected source changed: ' + p)


def make_plan(args, base, ctx):
    pp = args.out / 'plan.json'
    if pp.exists():
        plan = read(pp)
        require(plan['version'] == VERSION and plan['launcher_sha256'] == sha(__file__),
                'New runner version cannot be mixed with an existing output.')
        require(plan['root'] == str(args.root) and plan['source_root'] == str(ctx['root']) and
                plan['cases'] == list(args.cases), 'Use original root/cases/output options.')
        if args.phase in ('all', 'train'):
            if args.gpus:
                require(plan['gpus'] == args.gpus, 'Training GPU assignment changed.')
            if args.device:
                require(all(j['device'] == args.device for j in plan['jobs'].values()), 'Training device changed.')
        for job in plan['jobs'].values():
            verify_protected(job)
        return plan
    require(args.phase in ('all', 'train'), 'No existing final-only plan: train first.')
    args.out.mkdir(parents=True, exist_ok=True)
    require(not any(p.name not in ('.launcher.lock', 'TABLE55_SOURCE_DEBUG.zip') for p in args.out.iterdir()), 'Use an empty NEW --out folder.')
    devices = [f'cuda:{g}' for g in args.gpus] if args.gpus else [args.device or ctx['settings']['device']]
    jobs = {}
    for i, case in enumerate(args.cases):
        # Deliberately ignore/delete no prior weights: every control is a fresh independent 0->40M run.
        resume = None
        sp, st, _ = base.make_plan(ctx, case, devices[i % len(devices)])
        old_sp = deepcopy(sp)
        st.update(total_timesteps=TARGET, checkpoint_every_timesteps=10**15,
                  resume_every_timesteps=10**15, retain_completed_resume=False)
        if resume:
            # Keep the completed 5M learning-rate floor; never restart the high LR.
            floor = st['ppo']['learning_rate_floor']
            st['ppo']['learning_rate_initial'] = floor
        meta = dict(version=VERSION, cumulative_target=TARGET, validation=False,
                    intermediate_checkpoints=False, recovery_checkpoints=False,
                    source_step=0, source_episodes=0,
                    start='fresh_independent_weights',
                    lr_policy='original_schedule_over_40m',
                    source_environment_rng_restored=False, test_count=1220, raw_only=True)
        sp.update(run_id=base.run_id(case) + '__40M_fresh_finalonly', stage='table55_controls40m_fresh_finalonly',
                  table55_execution=meta)
        st['table55'] = meta
        require(sp['environment'] == old_sp['environment'] and sp['encoder_ablation'] == old_sp['encoder_ablation'],
                'Environment/CNN changed.')
        protected = {str(args.root / n): sha(args.root / n) for n in base.PACKAGE}
        protected[str(ctx['root'] / 'protocol.json')] = sha(ctx['root'] / 'protocol.json')
        protected[str(ctx['root'] / 'manifest.csv')] = sha(ctx['root'] / 'manifest.csv')
        if resume:
            protected.update(resume['protected'])
        jobs[case] = dict(case=case, device=st['device'], case_dir=str(args.out / case),
                          spec=sp, settings=st, resume=resume,
                          evaluation_seed=int(ctx['protocol']['request']['evaluation']['initial_seed']),
                          protected=protected)
    plan = dict(version=VERSION, root=str(args.root), source_root=str(ctx['root']), out=str(args.out),
                launcher_sha256=sha(__file__), cases=list(args.cases), jobs=jobs, gpus=args.gpus,
                target=TARGET, start_option='fresh_only', validation_rollouts=0,
                train_count=900, synthetic_test_count=720, osm_test_count=500,
                eval_threads=args.eval_threads)
    write(pp, plan)
    return plan


def prepare_runtime(plan, job):
    base = original(Path(plan['root']))
    ctx = base.source_context(Path(plan['source_root']))
    case_dir = Path(job['case_dir'])
    case_dir.mkdir(parents=True, exist_ok=True)
    snap = case_dir / 'source_snapshot'
    pp = case_dir / 'protocol.json'
    if pp.exists():
        p = read(pp)
        require(p['job_digest'] == digest(job), 'Frozen job changed.')
        for name, h in p['snapshot_hash'].items():
            require(sha(base.child(snap, name)) == h, 'Private source snapshot changed: ' + name)
        require(sha(case_dir / 'manifest.csv') == p['manifest_sha256'], 'Private manifest changed.')
    else:
        require(not snap.exists(), 'Unfinished snapshot creation; preserve output and choose a new --out.')
        hashes = base.prepare_snapshot(ctx, snap, job['case'])
        shutil.copy2(ctx['root'] / 'manifest.csv', case_dir / 'manifest.csv')
        write(pp, dict(version=VERSION, job_digest=digest(job), snapshot_hash=hashes,
                       manifest_sha256=sha(case_dir / 'manifest.csv'),
                       validation_executed=False, raw_only=True))
    rt, _ = base.runtime(snap, job['case'])
    trainer, utils, initial, core, features, _ = rt
    records = utils.load_manifest(case_dir / 'manifest.csv', 50, verify_files=False)
    counts = Counter((r.source, canon(r.split)) for r in records)
    require(counts == {('synthetic', 'train'): 900, ('synthetic', 'validation'): 180,
                       ('synthetic', 'test'): 720, ('osm', 'test'): 500}, 'Manifest partition changed.')
    tr = [r for r in records if canon(r.split) == 'train' and r.source == 'synthetic']
    # The Validation map arrays are not opened; no Validation rollout is requested.
    return base, trainer, utils, initial, core, features, records, tr


def tensor_digest(value):
    import torch
    h = hashlib.sha256()
    def walk(v):
        if isinstance(v, torch.Tensor):
            a = v.detach().cpu().contiguous()
            h.update((str(a.dtype) + str(tuple(a.shape))).encode())
            h.update(a.numpy().tobytes())
        elif isinstance(v, dict):
            for k in sorted(v, key=lambda x: (type(x).__name__, repr(x))):
                h.update(repr(k).encode()); walk(v[k])
        elif isinstance(v, (list, tuple)):
            h.update(str(len(v)).encode())
            for x in v:
                walk(x)
        else:
            h.update((type(v).__name__ + ':' + repr(v)).encode())
    walk(value)
    return h.hexdigest()


def rebuild_orders(utils, tr, resume):
    st = resume['settings']
    states = []
    for rank in range(st['num_envs']):
        rr = csv_read(Path(resume['run']) / f'map_order_rank{rank}.csv')
        require(rr, 'Missing source sampler history.')
        positions = [int(r['sequence_position']) for r in rr]
        require(positions == list(range(positions[0], positions[-1] + 1)), 'Sampler log has gaps.')
        require(all(int(r['episode']) == p and int(r['rank']) == rank for r, p in zip(rr, positions)),
                'Sampler episode/rank mismatch.')
        order = utils.BalancedMapOrder(tr, st['map_sequence_seed'] + 42*1009 + rank)
        by_pos = dict(zip(positions, rr))
        for pos in range(1, positions[-1] + 1):
            rec = tr[order.next_index()]
            if pos in by_pos:
                require(rec.map_id == by_pos[pos]['map_id'], 'Source sampler replay differs.')
        states.append(dict(order=order.state_dict(), episode=positions[-1]))
    return states


def check_model(model, settings):
    for k in PPO_KEYS:
        require(getattr(model, k) == settings['ppo'][k], 'PPO parameter mismatch: ' + k)
    require(model.n_envs == settings['num_envs'] and model.seed == 42, 'PPO seed/env count mismatch.')
    require(set(model.observation_space.spaces) == {'map', 'heading', 'detour_budget'},
            'Unexpected observation; clock/beta input must not be added.')


def install_patch(trainer, utils, features, tr, job):
    source = Path(trainer.__file__).read_text(encoding='utf-8-sig')
    patch, audit = patch_train_sb3(source)
    dest = Path(job['case_dir']) / 'train_sb3_finalonly.py'
    write_text(dest, patch)
    write_text(dest.with_suffix('.diff'), ''.join(difflib.unified_diff(
        source.splitlines(True), patch.splitlines(True), fromfile=Path(trainer.__file__).name,
        tofile='train_sb3_finalonly.py (function-only override)')))
    write(Path(job['case_dir']) / 'save_policy_audit.json', audit)
    resume = job['resume']
    if resume:
        require(trainer.run_identity(resume['spec'], resume['settings'], tr) == resume['training_identity'],
                'Source train record order/spec/settings identity mismatch.')
    orders = rebuild_orders(utils, tr, resume) if resume else None

    def build(cls, common, env, spec, settings, out, records):
        if resume is None:
            model = cls(**common)
            check_model(model, settings)
            return model
        from stable_baselines3.common.save_util import load_from_zip_file
        require(sha(resume['model']) == resume['sha256'], 'Source model changed.')
        _, parameters, _ = load_from_zip_file(resume['model'], device='cpu', load_data=False)
        require(parameters and {'policy', 'policy.optimizer'} <= set(parameters), 'Policy/Adam missing.')
        expected = tensor_digest(parameters)
        model = cls.load(resume['model'], env=env, device=settings['device'], force_reset=True)
        require(tensor_digest(model.get_parameters()) == expected, 'Policy/Adam were not loaded exactly.')
        require(model.num_timesteps == resume['step'] and model._n_updates == resume['epoch_counter'],
                'Source counters changed.')
        opt = model.policy.optimizer.state_dict()
        require(opt['state'] and all(float(x['step']) > 0 for x in opt['state'].values()), 'Missing Adam moments.')
        lr = settings['ppo']['learning_rate_floor']
        require(all(abs(float(g['lr']) - lr) <= 1e-12 for g in opt['param_groups']),
                'Completed source optimizer is not at the expected LR floor; no silent schedule change.')
        model.learning_rate = float(lr)
        model._setup_lr_schedule()
        model.tensorboard_log = str(Path(out) / 'tensorboard') if settings.get('tensorboard') else None
        model._last_obs = None
        model._last_episode_starts = None
        for rank, adapter in enumerate(env.envs):
            adapter.stream.order.load_state_dict(orders[rank]['order'])
            adapter.stream.episode = orders[rank]['episode']
            adapter.stream.total_steps = resume['step'] // settings['num_envs']
        check_model(model, settings)
        if hasattr(features, 'audit_cnn3_model'):
            features.audit_cnn3_model(model)
        require(tensor_digest(model.get_parameters()) == expected, 'Optimizer state changed during initialization.')
        write(Path(out) / 'load_audit.json', dict(source_step=resume['step'], source_sha256=resume['sha256'],
              policy_adam_exact=True, source_environment_rng_restored=False,
              sampler_positions=[x['episode'] for x in orders], learning_rate=lr,
              subsequent_recovery_saves=False))
        print(f'[LOAD 5M] {job["case"]}: {model.num_timesteps:,} -> cumulative {TARGET:,}; LR={lr:g}', flush=True)
        return model

    trainer.__dict__['_t55_build_model'] = build
    exec(compile(patch, str(dest), 'exec'), trainer.__dict__)


def final_model(job):
    run = Path(job['case_dir']) / 'runs' / job['spec']['run_id']
    status = read(run / 'status.json')
    require(status.get('state') == 'completed' and status.get('validation_executed') is False,
            'No completed final-only 40M model.')
    cp = run / 'last_model.zip'
    regs = read(run / 'checkpoints.json')
    require(len(regs) == 1 and regs[0].get('final') and regs[0]['sha256'] == sha(cp),
            'Only one registered final model is allowed.')
    meta = model_metadata(cp)
    block = job['settings']['num_envs'] * job['settings']['ppo']['n_steps']
    require(TARGET <= int(meta['num_timesteps']) < TARGET + block and
            meta['num_timesteps'] == status['actual_timesteps'], 'Final cumulative step mismatch.')
    require(not list((run / 'checkpoints').glob('*')) and not list((run / 'resume').rglob('*')),
            'Unexpected intermediate/recovery files in final-only output.')
    return run, cp, meta


def train_worker(plan, job):
    verify_protected(job)
    base, trainer, utils, initial, core, features, records, tr = prepare_runtime(plan, job)
    run = Path(job['case_dir']) / 'runs' / job['spec']['run_id']
    status_path = run / 'status.json'
    if status_path.exists():
        if read(status_path).get('state') == 'completed':
            final_model(job)
            print('[TRAIN SKIP] completed ' + job['case'], flush=True)
            return
        raise RuntimeError('Interrupted final-only training has no recovery checkpoint. Preserve logs and use a NEW --out to restart fresh 0->40M with a NEW --out.')
    require(not run.exists() or not any(run.iterdir()), 'Nonempty unrecognized run directory.')
    # Static patch compilation precedes model construction; this is not Validation.
    install_patch(trainer, utils, features, tr, job)
    import numpy as np
    import torch
    import importlib.metadata
    for package in ('stable-baselines3', 'sb3-contrib'):
        require(importlib.metadata.version(package) == '2.7.0', 'Use the original 2.7.0 environment: ' + package)
    torch.set_num_threads(int(job['settings'].get('torch_threads', 4)))
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    run.mkdir(parents=True, exist_ok=True)
    sp, st = deepcopy(job['spec']), deepcopy(job['settings'])
    write(run / 'run_spec.json', sp)
    write(run / 'training_settings.json', st)
    write(run / 'software.json', trainer.software_record())
    status_base = dict(identity=trainer.run_identity(sp, st, tr), version=VERSION,
                       validation_executed=False, test_executed=False,
                       requested_timesteps=TARGET, source_step=st['table55']['source_step'],
                       intermediate_checkpoints=False, recovery_checkpoints=False,
                       source_environment_rng_restored=False)
    write(status_path, dict(status_base, state='training'))
    print(f'[TRAIN] {job["case"]} {st["table55"]["source_step"]:,} -> TOTAL {TARGET:,} | {job["device"]}', flush=True)
    print('[DISABLED] Validation / preflight rollout / intermediate checkpoints / recovery saves / refinement', flush=True)
    try:
        trainer._check_settings(st)
        result = trainer.train_sb3(sp, st, tr, run)
        write(status_path, dict(status_base, **{k: v for k, v in result.items() if k not in status_base},
                               state='completed'))
        final_model(job)
        verify_protected(job)
        print('[TRAIN DONE] ' + job['case'] + ': final model saved once; no Validation.', flush=True)
    except BaseException as exc:
        write(status_path, dict(status_base, state='interrupted_no_recovery', error=repr(exc),
                               traceback=traceback.format_exc()))
        raise


def validate_row(row, job):
    require(row['source'] in ('synthetic', 'osm') and canon(row['split']) == 'test',
            'Non-Test row in final evaluation.')
    require(row.get('mask_mode') == job['case'], 'Wrong evaluation mask.')
    require(int(row['steps']) == int(row['valid_move_count']) + int(row['invalid_action_count']),
            'Attempts and actual movements were mixed.')
    require(int(row['max_steps']) == math.ceil(5 * int(row['active_cells'])), 'H5 changed.')
    require(number(row['coverage_pct']) is not None and 0 <= float(row['coverage_pct']) <= 100.000001,
            'Invalid coverage.')
    require(number(row[TKEY]) is not None and float(row[TKEY]) >= 0, 'Invalid elapsed time.')
    if truth(row['is_success']):
        require(abs(float(row['coverage_pct']) - 100) < 1e-5 and number(row[LKEY]) is not None,
                'Invalid success/latency.')
    # Failed-run elapsed time remains in per-map evidence, not a completed-time mean.
    require(not truth(row['is_success']) or 0 <= float(row[LKEY]) <= float(row[TKEY]) + 1e-6,
            'Invalid completed latency.')


def evaluate_one(trainer, initial, job, model, rec, array):
    import numpy as np
    import torch
    cfg = trainer.ExperimentEnvConfig(**job['spec']['environment'])
    ic = initial.initial_condition(rec, cfg, job['spec']['initialization'], 0, job['evaluation_seed'])
    core = trainer.make_core(cfg)
    obs, _ = core.reset(array, start=tuple(ic['start']), heading=ic['heading'], seed=42, map_name=rec.map_id)
    counts, entropy, forced, steps = 0, 0.0, 0, 0
    if core.complete_at_reset:
        core.step(0)
    else:
        while not core._done:
            mask = np.asarray(core.action_masks(), dtype=bool)
            require(mask.shape == (4,) and mask.any(), 'Empty native mask.')
            if job['case'] == 'none':
                require(mask.all(), 'No-mask must expose all four actions in Test as in training.')
            with torch.no_grad():
                tensor, _ = model.policy.obs_to_tensor(obs)
                distribution = model.policy.get_distribution(tensor, action_masks=mask.reshape(1, -1))
                p = distribution.distribution.probs[0].detach().cpu().numpy().astype(float)
            require(np.isfinite(p).all() and abs(p.sum() - 1) < 1e-5 and np.all(p[~mask] < 1e-6),
                    'Invalid native action distribution.')
            action = int(np.argmax(p))
            require(mask[action], 'Argmax selected masked action.')
            positive = p[p > 0]
            entropy += -float(np.sum(positive * np.log(positive)))
            counts += int(mask.sum()); forced += int(mask.sum() == 1)
            obs, _, terminated, truncated, _ = core.step(action)
            steps += 1
            require(steps <= int(core.max_steps) + 1 and not (terminated and truncated), 'Native stop rule changed.')
    core.assert_consistent()
    row = core.info(final=True)
    row.update(map_id=rec.map_id, map_sha256=rec.sha256, source=rec.source, split='test',
               pattern=rec.pattern, tier=rec.tier, method=LABELS[job['case']], case=job['case'],
               beta=0.5, raw_or_refined='Raw', deterministic=True,
               initial_pair_id=ic['pair_id'], start_r=ic['start'][0], start_c=ic['start'][1],
               start_heading=ic['heading'], mask_cardinality_mean=counts/steps if steps else None,
               forced_action_ratio=forced/steps if steps else None,
               masked_entropy_mean=entropy/steps if steps else None)
    validate_row(row, job)
    return safe(row)


def aggregate(rr):
    good = [r for r in rr if truth(r['is_success'])]
    result = dict(total_n=len(rr), success_n=len(good),
                  success_pct=100*len(good)/len(rr) if rr else None,
                  coverage_pct=mean(r['coverage_pct'] for r in rr), time_mean_n=len(good),
                  raw_T_s=mean(r[TKEY] for r in good), raw_Lw_s=mean(r[LKEY] for r in good))
    for k in ('steps', 'valid_move_count', 'invalid_action_count', 'mask_cardinality_mean',
              'forced_action_ratio', 'masked_entropy_mean'):
        result['all_maps_mean_' + k] = mean(r.get(k) for r in rr)
    for k in ('path_length_m', 'turn_count', 'revisit_steps', 'zero_transit_steps'):
        result['successful_maps_mean_' + k] = mean(r.get(k) for r in good)
    return result


def evaluate_worker(plan, job):
    verify_protected(job)
    _, trainer, utils, initial, core, features, records, tr = prepare_runtime(plan, job)
    run, cp, meta = final_model(job)
    chosen = sorted([r for r in records if canon(r.split) == 'test'],
                    key=lambda r: (r.source, str(r.tier), r.pattern, r.map_id))
    require(len(chosen) == 1220 and Counter(r.source for r in chosen) == {'synthetic': 720, 'osm': 500},
            'Test must be exactly Synthetic720 + OSM500; no Validation.')
    for field in ('map_id', 'sha256', 'path'):
        require(len({getattr(r, field) for r in chosen}) == 1220 and
                not {getattr(r, field) for r in chosen} & {getattr(r, field) for r in tr},
                'Test duplicate or train/Test overlap: ' + field)
    root = Path(job['case_dir']) / 'test1220'
    identity = digest(dict(job=digest(job), model=sha(cp), source_hash=read(Path(job['case_dir'])/'protocol.json')['snapshot_hash'],
                           eval_device='cpu', eval_threads=plan['eval_threads'],
                           maps=[(r.map_id, r.sha256) for r in chosen]))
    root.mkdir(parents=True, exist_ok=True)
    ip = root / 'identity.json'
    if ip.exists():
        require(read(ip)['identity'] == identity, 'Evaluation cache belongs to another model/protocol.')
    else:
        write(ip, dict(identity=identity, final_model_sha256=sha(cp), actual_timesteps=meta['num_timesteps'],
                       validation_count=0, synthetic_test_count=720, osm_test_count=500))
    import numpy as np
    import torch
    from sb3_contrib import MaskablePPO
    torch.set_num_threads(int(plan['eval_threads']))
    model = None
    rr = []
    for i, rec in enumerate(chosen, 1):
        dest = root / 'maps' / (hashlib.sha256(rec.map_id.encode()).hexdigest() + '.json')
        array = np.load(rec.path, allow_pickle=False)
        require(utils.array_hash(array) == rec.sha256, 'Test map changed: ' + rec.map_id)
        if dest.exists():
            cache = read(dest); row = cache['row']
            require(cache['identity'] == identity and cache['row_sha256'] == digest(row) and
                    row['map_id'] == rec.map_id and row['map_sha256'] == rec.sha256, 'Corrupt Test cache.')
            validate_row(row, job)
        else:
            if model is None:
                model = MaskablePPO.load(cp, device='cpu')
                check_model(model, job['settings'])
                if hasattr(features, 'audit_cnn3_model'):
                    features.audit_cnn3_model(model)
                model.policy.set_training_mode(False)
            row = evaluate_one(trainer, initial, job, model, rec, array)
            row.update(actual_timesteps=int(meta['num_timesteps']), final_model_sha256=sha(cp))
            write(dest, dict(identity=identity, row_sha256=digest(row), row=row))
        rr.append(row)
        if i == 1 or i % 25 == 0 or i == 1220:
            csv_write(root / 'per_map.csv', rr)
            print(f'[TEST] {job["case"]}: {i}/1220; success={sum(truth(r["is_success"]) for r in rr)}', flush=True)
    require(len(rr) == len({r['map_id'] for r in rr}) == 1220, 'Partial Test result.')
    summaries = [dict(case=job['case'], method=LABELS[job['case']], source=domain,
                      actual_timesteps=int(meta['num_timesteps']), **aggregate([r for r in rr if r['source'] == domain]))
                 for domain in ('synthetic', 'osm')]
    csv_write(root / 'summary.csv', summaries)
    write(root / 'complete.json', dict(identity=identity, rows=1220,
          files={n: sha(root/n) for n in ('per_map.csv', 'summary.csv', 'identity.json')}))
    status = read(run / 'status.json'); status['test_executed'] = True
    write(run / 'status.json', status)
    verify_protected(job)


def report(plan):
    root = Path(plan['out']); dest = root / 'report'
    all_rows, table_rows, groups, missing = [], [], [], []
    for case in plan['cases']:
        job = plan['jobs'][case]; ev = Path(job['case_dir']) / 'test1220'
        if not (ev / 'complete.json').is_file():
            missing.append(case); continue
        completed = read(ev / 'complete.json')
        for name, h in completed['files'].items():
            require(sha(ev/name) == h, 'Report input changed: ' + str(ev/name))
        rr = csv_read(ev / 'per_map.csv')
        require(len(rr) == 1220 and all(canon(r['split']) == 'test' for r in rr), 'Incomplete Test data.')
        all_rows.extend(rr)
        for domain, n in (('synthetic', 720), ('osm', 500)):
            subset = [r for r in rr if r['source'] == domain]
            require(len(subset) == n, 'Wrong domain denominator.')
            table_rows.append(dict(case=case, method=LABELS[case], source=domain, training_budget='40M',
                              actual_timesteps=int(float(subset[0]['actual_timesteps'])),
                              path_stage='Raw', **aggregate(subset)))
        grouped = defaultdict(list)
        for r in rr:
            grouped[(r['source'], str(r['tier']), r['pattern'])].append(r)
        for (domain, tier, pattern), subset in sorted(grouped.items()):
            groups.append(dict(case=case, source=domain, tier=tier, pattern=pattern, **aggregate(subset)))
    csv_write(dest / 'Table5_5_controls_40M.csv', table_rows)
    csv_write(dest / 'all_controls_test_per_map.csv', all_rows)
    csv_write(dest / 'pattern_tier_summary.csv', groups)
    write(dest / 'report_status.json', dict(complete=not missing, missing_cases=missing,
          expected_rows=len(plan['cases'])*1220, actual_rows=len(all_rows), validation_rows=0,
          strict_relaxed_results='not retrained or overwritten; merge these control rows with existing 40M Raw results'))
    header = ['Source', 'Method', 'Actual steps', 'Success/total', 'Coverage (%)', 'n (time)', 'Raw T (s)', 'Raw Lw (s)']
    def fmt(v):
        return '—' if number(v) is None else f'{float(v):,.2f}'
    lines = ['# Table 5.5: No-mask / Boundary-only, COST11 FULL 40M Raw', '',
             '**' + ('COMPLETE' if not missing else 'PARTIAL: missing ' + ', '.join(missing)) + '**', '',
             '| ' + ' | '.join(header) + ' |', '|' + '|'.join(['---']*len(header)) + '|']
    html_rows = []
    for r in table_rows:
        vals = [r['source'], r['method'], str(r['actual_timesteps']), f"{r['success_n']}/{r['total_n']}",
                fmt(r['coverage_pct']), str(r['time_mean_n']), fmt(r['raw_T_s']), fmt(r['raw_Lw_s'])]
        lines.append('| ' + ' | '.join(vals) + ' |')
        html_rows.append('<tr>' + ''.join('<td>'+html.escape(v)+'</td>' for v in vals) + '</tr>')
    lines += ['', 'Coverage uses ALL maps. Completed T/Lw use each method\'s own successful maps only.',
              'A zero-success row has undefined T/Lw (—), not zero seconds.',
              'No Validation or refinement was run. Training logs are retained; intermediate model/recovery saves are disabled.',
              'Existing Strict/Relaxed 40M rows are not regenerated or relabeled by this package.', '']
    write_text(dest / 'Table5_5_controls_40M.md', '\n'.join(lines))
    write_text(dest / 'report.html', '<!doctype html><meta charset="utf-8"><title>Table 5.5 controls 40M</title>'
      '<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;line-height:1.6}'
      'table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ccc;text-align:right}'
      'th{background:#eee}</style><h1>Table 5.5 · Controls · 40M · Raw</h1><p>'
      + ('Complete' if not missing else 'Partial: ' + html.escape(', '.join(missing)))
      + '</p><table><thead><tr>' + ''.join('<th>'+html.escape(v)+'</th>' for v in header)
      + '</tr></thead><tbody>' + ''.join(html_rows) + '</tbody></table>'
      + '<p>Validation: 0 maps. Synthetic Test: 720 maps; OSM Test: 500 maps per model. '
      'Time means are conditional on success. — means undefined. Strict/Relaxed are not retrained.</p>')
    zp = root / 'TABLE55_CONTROLS40M_FINALONLY_REVIEW.zip'
    temp = zp.with_suffix('.writing.zip')
    with zipfile.ZipFile(temp, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(root / 'plan.json', 'plan.json')
        for p in sorted(dest.iterdir()):
            if p.is_file():
                z.write(p, 'report/' + p.name)
        for case in plan['cases']:
            cdir = root / case
            for name in ('protocol.json', 'save_policy_audit.json', 'train_sb3_finalonly.py'):
                p = cdir / name
                if p.is_file():
                    z.write(p, case + '/' + name)
            rdir = cdir / 'runs' / plan['jobs'][case]['spec']['run_id']
            for name in ('status.json','run_spec.json','training_settings.json','software.json',
                         'checkpoints.json','load_audit.json','model_architecture.json','training_updates.csv'):
                p = rdir/name
                if p.is_file():
                    z.write(p, case + '/run_metadata/' + name)
    os.replace(temp, zp)
    print('[REPORT] ' + str(zp), flush=True)


def launch(plan, phase):
    root = Path(plan['out']); logs = root / 'logs'; logs.mkdir(exist_ok=True)
    pending, active, failed = list(plan['cases']), {}, []
    try:
        while pending or active:
            busy = {x[1] for x in active.values()}
            for case in list(pending):
                job = plan['jobs'][case]
                slot = job['device'] if phase == 'train' else 'cpu_' + case
                if slot in busy:
                    continue
                env = os.environ.copy(); env['PYTHONUNBUFFERED'] = '1'; env['MPLBACKEND'] = 'Agg'
                if phase == 'test':
                    env['CUDA_VISIBLE_DEVICES'] = ''
                logfile = logs / (case + '_' + phase + '.log')
                f = logfile.open('a', encoding='utf-8', buffering=1)
                cmd = [sys.executable, str(Path(__file__).resolve()), '--_worker', phase,
                       '--_plan', str(root/'plan.json'), '--case', case]
                proc = subprocess.Popen(cmd, cwd=plan['root'], env=env, stdout=f,
                                        stderr=subprocess.STDOUT, start_new_session=True)
                active[proc] = (case, slot, f); busy.add(slot); pending.remove(case)
                print(f'[LAUNCH {phase}] {case} {slot} pid={proc.pid}; log={logfile}', flush=True)
            for proc, (case, slot, f) in list(active.items()):
                if proc.poll() is not None:
                    f.close(); del active[proc]
                    print(f'[EXIT {phase}] {case}: {proc.returncode}', flush=True)
                    if proc.returncode:
                        failed.append((case, proc.returncode))
            if active:
                time.sleep(0.5)
    except BaseException:
        for proc in active:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
        for proc, (_, _, f) in active.items():
            try:
                proc.wait()
            finally:
                f.close()
        raise
    require(not failed, 'Worker failure: ' + str(failed) + '. Read logs; no intermediate model was saved.')


def debug_bundle(root, source_root, out, error):
    """On an incompatible/missing native source, collect only small code/metadata."""
    try:
        out.mkdir(parents=True, exist_ok=True)
        path = out / 'TABLE55_SOURCE_DEBUG.zip'
        with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
            z.writestr('ERROR.txt', error)
            z.write(__file__, 'new_runner.py')
            names = ['run_cost11_fourway.py','cost11_fourway_core.py','cost11_fourway_checks.py','cost11_fourway_report.py']
            for name in names:
                p = root/name
                if p.is_file():
                    z.write(p, 'suite/' + name)
            for p in (source_root/'source_snapshot').glob('*.py'):
                if p.stat().st_size < 2_000_000:
                    z.write(p, 'original_source_snapshot/' + p.name)
            for p in [source_root/'protocol.json'] + list((source_root/'runs').glob('*/training_settings.json')):
                if p.is_file():
                    z.write(p, 'source_metadata/' + str(p.relative_to(source_root)))
        print('[SOURCE DEBUG] ' + str(path), file=sys.stderr)
    except Exception as exc:
        print('[DEBUG PACK FAILED] ' + repr(exc), file=sys.stderr)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, default=Path('.'), help='Existing uav_chapter4_suite_v2 directory.')
    p.add_argument('--source-root', type=Path, default=Path(DEFAULT_SOURCE), help='Original completed COST11 source root.')
    p.add_argument('--out', '--out-root', dest='out', type=Path, default=Path(DEFAULT_OUT))
    p.add_argument('--phase', choices=('all','train','test','report'), default='all')
    p.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    g = p.add_mutually_exclusive_group(); g.add_argument('--gpus', nargs='+', type=int); g.add_argument('--device')
    p.add_argument('--eval-threads', type=int, default=1)
    p.add_argument('--_worker', choices=('train','test'), help=argparse.SUPPRESS)
    p.add_argument('--_plan', type=Path, help=argparse.SUPPRESS)
    p.add_argument('--case', choices=CASES, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args._worker:
        require(args._plan and args.case, 'Internal worker arguments missing.')
        plan = read(args._plan)
        require(plan['launcher_sha256'] == sha(__file__), 'Runner changed after plan creation.')
        job = plan['jobs'][args.case]
        try:
            with lock(Path(job['case_dir']) / '.worker.lock'):
                if args._worker == 'train':
                    train_worker(plan, job)
                else:
                    evaluate_worker(plan, job)
        except Exception:
            debug_bundle(Path(plan['root']), Path(plan['source_root']), Path(job['case_dir']), traceback.format_exc())
            raise
        return
    args.root = args.root.expanduser().resolve()
    args.out = resolve(args.root, args.out)
    source_root = resolve(args.root, args.source_root)
    require(args.out != args.root and not overlap(args.out, source_root), 'Output must be separate from suite/source.')
    require(args.root.is_dir() and args.eval_threads >= 1, 'Invalid root/eval threads.')
    require(len(args.cases) == len(set(args.cases)), 'Duplicate cases.')
    if args.gpus:
        require(min(args.gpus) >= 0 and len(set(args.gpus)) == len(args.gpus), 'Invalid GPU IDs.')
    try:
        with lock(args.out / '.launcher.lock'):
            base = original(args.root)
            ctx = base.source_context(source_root)
            plan = make_plan(args, base, ctx)
            for case, job in plan['jobs'].items():
                print(f'[PLAN] {LABELS[case]}: FRESH 0 -> {TARGET:,}; {job["device"]}', flush=True)
            print('[PLAN] Fresh independent weights; Validation=0; intermediate/recovery checkpoints=0; final model=1/case; Raw Test=720+500/case', flush=True)
            if args.phase in ('all','train'):
                launch(plan, 'train')
            if args.phase in ('all','test'):
                launch(plan, 'test')
            if args.phase in ('all','test','report'):
                report(plan)
    except Exception:
        debug_bundle(args.root, source_root, args.out, traceback.format_exc())
        raise


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n[STOP] No intermediate/recovery model was saved. Completed final models and Test caches remain.', file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        traceback.print_exc()
        print('[ERROR] ' + str(exc), file=sys.stderr)
        raise SystemExit(1)
