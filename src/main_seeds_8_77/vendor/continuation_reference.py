#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Continue the three reviewed COST11 PPO runs from 5M to TOTAL 20M (fix1).

Fixes the reversed spec/settings arguments that caused KeyError: 'ppo' before
model.learn(). The exact known v1 output can be retried in the SAME directory:
no source checkpoints, native recovery, frozen plan or protocol are removed.
The executable revision is recorded separately in active_launcher.json.

Uses each experiment's frozen, authenticated trainer/core/CNN/logger snapshot.
Does not train a fresh policy, alter the environment/reward/mask, or run test.
Default physical GPUs: 1 -> relaxed_b05, 2 -> relaxed_b0, 3 -> strict_b05.

  python continue_cost11_three_models_20m_fixed.py --root . --phase check
  python continue_cost11_three_models_20m_fixed.py --root .
  python continue_cost11_three_models_20m_fixed.py --phase report --out chapter4_COST11_5M_to_20M_seed42

Use the ORIGINAL trusted checkpoints and original Python/SB3 training environment.
An SB3 checkpoint uses Python serialization: never load an untrusted model.
Only the initial 5M->extension boundary restarts episodes/RNG; sampler positions are
reconstructed from the original per-rank logs. Later interruptions recover the full
extension environment, sampler, optimizer, and RNG at a committed update boundary.
"""
from __future__ import annotations

import argparse
import ast
import csv
import difflib
import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import io
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace

# Keep the frozen experiment identity unchanged. This release changes only the
# assertion bridge and adds preflight/launcher-compatibility checks, not training.
VERSION = 'cost11_three_5m_to_20m_v1_20260930'
BUGFIX_VERSION = 'cost11_20m_fix1_assert_bridge_20260930'
KNOWN_V1_LAUNCHER_SHA256 = 'be3fad2406a3301a18678c75b30fcaa81321169a864ea822a13aaae48683806b'
DEFAULT_OUT = 'chapter4_COST11_5M_to_20M_seed42'
CASES = {
    'relaxed_b05': {
        'label': 'Relaxed FULL COST11 beta=0.5 B=6', 'beta': 0.5, 'mask': 'relaxed',
        'relative_root': 'chapter4_relaxedB6_C3NT_COST_SWEEP_seed42/cost11',
        'run_id': 'ppo_relaxedB6_NRG1_C3NT_COST11_b0.5_g1_l0.95_budgetobs_seed42'},
    'relaxed_b0': {
        'label': 'Relaxed FULL COST11 beta=0 B=6', 'beta': 0.0, 'mask': 'relaxed',
        'relative_root': 'chapter4_C3NT_COST11_BETA_0_07_09_1_seed42/beta0',
        'run_id': 'ppo_relaxedB6_C3NT_COST11_FULL_H5_b0_g1_l0.95_budgetobs_seed42'},
    'strict_b05': {
        'label': 'Strict FULL COST11 beta=0.5', 'beta': 0.5, 'mask': 'progress',
        'relative_root': 'chapter4_C3NT_COST11_STRICT_BETA_0_05_1_seed42/beta0p5',
        'run_id': 'ppo_progress_C3NT_COST11_FULL_H5_b0.5_g1_l0.95_budgetobs_strictsweep_seed42'},
}
TRAINER = 'trainers_ch4_c3nt_cost_sweep'
CORE = 'c3nt_cost_sweep_generated_core'
FEATURES = 'nrg1_c3nt_cost5_features'
LOGGER = 'c3nt_cost_sweep_generated_logging'
# All three authenticated trainers in the uploaded diagnostics have this hash.
TRAINER_SHA256 = '0049fb0f18f07cd80f2ec96a6ad16510843d01dcd1703ec7250435c80908585e'
CORE_VERSIONS = {
    'relaxed': 'nrg1_cnn3_noclock_externalcap_cost11_sweep_v1_20260926',
    'progress': 'cnn3_noclock_cost11_strict_progress_v1_20260929',
}
RUNTIME_FILES = [TRAINER+'.py', CORE+'.py', FEATURES+'.py', LOGGER+'.py',
                 'core_ch4.py', 'core_ch4_bounded_b6.py', 'coverage_core.py',
                 'full_v2_utils.py', 'resume_ch4.py']


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def plain(obj):
    if isinstance(obj, Path): return str(obj)
    if isinstance(obj, dict): return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)): return [plain(v) for v in obj]
    if hasattr(obj, 'tolist'): return plain(obj.tolist())
    if hasattr(obj, 'item'): return plain(obj.item())
    if isinstance(obj, float) and not math.isfinite(obj): return None
    return obj


def digest(obj):
    return hashlib.sha256(json.dumps(plain(obj), ensure_ascii=True, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.is_symlink(), 'Refusing symlink output: '+str(path))
    temp = path.with_name(path.name+f'.writing.{os.getpid()}')
    try:
        with temp.open('w', encoding='utf-8', newline='') as f:
            f.write(text); f.flush(); os.fsync(f.fileno())
        os.replace(temp, path)
        if os.name == 'posix':
            fd = os.open(str(path.parent), os.O_RDONLY)
            try: os.fsync(fd)
            finally: os.close(fd)
    finally:
        if temp.exists(): temp.unlink()


def write_json(path, value):
    atomic_text(path, json.dumps(plain(value), ensure_ascii=False, indent=2,
                                sort_keys=True, allow_nan=False)+'\n')


def freeze(path, value):
    if Path(path).exists():
        require(read_json(path) == plain(value),
                'Frozen settings/code/source/GPU changed: '+str(path)+
                '\nUse the SAME command and code to resume; do not overwrite the old run.')
    else:
        write_json(path, value)


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, values, fields=None):
    fields = fields or list(dict.fromkeys(k for r in values for k in r)) or ['status']
    s = io.StringIO(newline=''); w = csv.DictWriter(s, fieldnames=fields)
    w.writeheader()
    for r in values:
        w.writerow({k: json.dumps(plain(v), ensure_ascii=False) if isinstance(v, (list, dict)) else v
                    for k, v in plain(r).items() if k in fields})
    atomic_text(path, '\ufeff'+s.getvalue())


def within(a, b):
    """Whether resolved path a lies within resolved directory b (inclusive)."""
    a, b = Path(a).resolve(), Path(b).resolve()
    return a == b or b in a.parents


@contextmanager
def lock(path):
    import fcntl  # Target server: Linux. No stale lock-file deletion is needed.
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.is_symlink(), 'Symlink lock: '+str(path))
    with path.open('a+b') as f:
        try: fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e: raise RuntimeError('Another process owns '+str(path)) from e
        try: yield
        finally: fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def locate(base, key, explicit, output):
    c = CASES[key]
    if explicit:
        found = Path(explicit).expanduser().resolve()
        require((found/'runs'/c['run_id']/'run_spec.json').is_file(),
                'Explicit root is not the requested source case: '+str(found))
        return found
    direct = base/c['relative_root']
    if (direct/'runs'/c['run_id']/'run_spec.json').is_file(): return direct.resolve()
    found = set()
    skip = {'source_snapshot','__pycache__','.git','maps','resume','checkpoints',
            'tensorboard','site-packages','.venv','node_modules','raw','source_5m'}
    for directory, dirs, filenames in os.walk(base, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in skip and not d.startswith('_cost11_diagnostics')
                         and not within(Path(directory)/d, output)
                         and not (Path(directory)/d).is_symlink())
        p = Path(directory)
        if p.name == c['run_id'] and p.parent.name == 'runs' and 'run_spec.json' in filenames:
            found.add(p.parent.parent.resolve()); dirs[:] = []
    require(len(found) == 1, f'{key}: expected one source; found {len(found)}: '+
            ', '.join(map(str, sorted(found)))+'\nUse --relaxed-b05-root / --relaxed-b0-root / --strict-b05-root.')
    return next(iter(found))


def check_zip_metadata(path, registered, source_spec, settings, status):
    with zipfile.ZipFile(path) as z:
        require({'data', 'policy.pth', 'policy.optimizer.pth', '_stable_baselines3_version'} <= set(z.namelist()),
                'Missing policy/optimizer metadata in checkpoint: '+str(path))
        require(z.getinfo('data').file_size < 32_000_000, 'Oversized SB3 data JSON.')
        d = json.loads(z.read('data'))
        require(z.read('_stable_baselines3_version').decode().strip() == '2.7.0',
                'Source checkpoint is not the reviewed SB3 2.7.0 run.')
    expected = {k:settings['ppo'][k] for k in ('n_steps','batch_size','n_epochs','gamma',
                'gae_lambda','ent_coef','vf_coef','max_grad_norm','target_kl')}
    expected.update(num_timesteps=int(registered['num_timesteps']), n_envs=settings['num_envs'],
                    seed=source_spec['seed'], _total_timesteps=settings['total_timesteps'])
    for k,v in expected.items(): require(d.get(k) == v, f'Source checkpoint {k} changed: {d.get(k)!r} != {v!r}')
    require(d.get('normalize_advantage') is True, 'Unexpected advantage normalization.')
    require(d.get('_n_updates') == status['ppo_epoch_counter'], 'Source update count/status mismatch.')
    require(source_spec['run_id'] in str(d.get('tensorboard_log','')).replace('\\','/').split('/'),
            'Source checkpoint run provenance mismatch.')
    return {k:d.get(k) for k in expected} | {'_n_updates': d['_n_updates']}


def inspect_source(root, key):
    c = CASES[key]; run = root/'runs'/c['run_id']; snap = root/'source_snapshot'
    required = ['run_spec.json','training_settings.json','status.json','checkpoints.json',
                'software.json','training_updates.csv','training_episodes.csv']
    for name in required:
        p = run/name
        require(p.is_file() and not p.is_symlink(), 'Missing source file: '+str(p))
    spec, settings, status = (read_json(run/n) for n in required[:3])
    p = read_json(root/'protocol.json')
    require(spec.get('run_id') == c['run_id'] and spec.get('algorithm') == 'ppo' and spec.get('seed') == 42,
            key+': wrong source run.')
    require(spec.get('beta') == c['beta'], key+': wrong beta.')
    require(spec.get('nrg1_protocol_sha256') == sha(root/'protocol.json'), 'Source run/protocol hash mismatch.')
    require({k:v for k,v in spec.items() if k!='nrg1_protocol_sha256'} == p['request']['spec']
            and settings == p['request']['settings'], 'Source frozen spec/settings mismatch.')
    env = spec['environment']; rd = spec['reward_definition']
    for k,v in dict(mask_mode=c['mask'], tradeoff_beta=c['beta'], detour_budget_max=6,
                    step_budget_factor=5.0, grid_size=50, coverage_reward_scale=0.0,
                    objective_cost_scale=0.0, success_terminal_reward=0.0,
                    failure_terminal_penalty=0.0).items():
        require(env.get(k) == v, key+': unexpected environment.'+k)
    require(rd.get('first_visit') == 'delta_new_active_count / active_cell_count'
            and rd.get('cost_multiplier') == 11.0 and rd.get('success_terminal') == 2.0
            and rd.get('failure_terminal') == 0.0, 'Not the actual FULL COST11 replacement reward.')
    require(settings['total_timesteps'] == 5_000_000 and settings['num_envs'] == 8
            and settings['ppo']['gamma'] == 1.0 and settings['ppo']['gae_lambda'] == .95,
            'Expected original 5M/8-env/gamma=1/lambda=.95 settings.')
    require(settings['ppo']['learning_rate_floor'] == 5e-5, 'Source LR floor differs from the reviewed run.')
    require(status.get('state') == 'completed' and status.get('requested_timesteps') == 5_000_000,
            'Only completed original 5M sources may initialize this experiment.')
    require(sha(root/'manifest.csv') == p['manifest_sha256'], 'Source manifest changed.')
    manifest = rows(root/'manifest.csv')
    require(Counter((r['source'],r['split']) for r in manifest) ==
            {('synthetic','train'):900, ('synthetic','validation'):180,
             ('synthetic','test'):720, ('osm','test'):500}, 'Reviewed split counts changed.')
    for field in ('map_id','path','sha256'):
        require(len({r[field] for r in manifest}) == len(manifest), 'Duplicate/cross-split '+field)
    protected = {str(root/'protocol.json'):sha(root/'protocol.json'),
                 str(root/'manifest.csv'):sha(root/'manifest.csv')}
    for name in required + [f'map_order_rank{i}.csv' for i in range(8)]:
        f = run/name
        require(f.is_file() and not f.is_symlink(), 'Missing source file: '+str(f))
        protected[str(f)] = sha(f)
    updates = rows(run/'training_updates.csv')
    require(updates and int(float(updates[-1]['timesteps'])) == status['actual_timesteps'],
            'Source final update log/status mismatch.')
    require(math.isclose(float(updates[-1]['learning_rate']), settings['ppo']['learning_rate_floor'],
                         rel_tol=0, abs_tol=1e-12), 'Source final LR has not reached its floor.')
    hashes = {}
    for name in RUNTIME_FILES:
        f = snap/name; expected = p['snapshot_hash'].get(name)
        require(f.is_file() and not f.is_symlink() and expected and sha(f) == expected,
                'Frozen runtime missing/modified: '+str(f))
        hashes[name] = expected; protected[str(f)] = expected
    require(hashes[TRAINER+'.py'] == TRAINER_SHA256, 'Trainer version differs; no blind source patch will be applied.')
    regs = read_json(run/'checkpoints.json'); finals = [r for r in regs if r.get('final') is True]
    require(len(finals) == 1, 'No unique registered final checkpoint.')
    final = finals[0]; model = run/'last_model.zip'
    require(model.is_file() and not model.is_symlink() and sha(model) == final['sha256'],
            'Missing/changed original last_model.zip: '+str(model))
    require(int(final['num_timesteps']) == status['actual_timesteps'] == 5_005_312,
            'Expected reviewed final step 5,005,312.')
    metadata = check_zip_metadata(model, final, spec, settings, status)
    protected[str(model)] = final['sha256']
    return dict(key=key, label=c['label'], source_root=str(root), source_run=str(run),
                source_snapshot=str(snap), spec=spec, settings=settings, status=status,
                source_model=str(model), source_model_sha256=final['sha256'],
                source_model_metadata=metadata, runtime_hashes=hashes, protected=protected,
                software=read_json(run/'software.json'), manifest_sha256=p['manifest_sha256'])


def verify_protected(job):
    for p,h in job['protected'].items():
        require(Path(p).is_file() and sha(p) == h, 'Original input changed: '+p)


def gpu_inventory(indices):
    require(shutil.which('nvidia-smi'), 'nvidia-smi is unavailable. Run on the original GPU server.')
    r = subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name,memory.free',
                        '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
    inv = {}
    for row in csv.reader(r.stdout.splitlines()):
        require(len(row) >= 4, 'Unexpected nvidia-smi response.')
        inv[int(row[0].strip())] = dict(index=int(row[0].strip()), uuid=row[1].strip(), name=row[2].strip())
    require(len(set(indices)) == 3 and all(i in inv for i in indices),
            f'Three distinct physical GPUs are required. Requested={indices}; available={sorted(inv)}')
    return [inv[i] for i in indices]


def extension_config(job, target, cp_interval):
    base = Path(job['case_dir']); source_step = job['status']['actual_timesteps']
    e = dict(version=VERSION, source_run_id=job['spec']['run_id'],
             source_model=str(base/'source_5m'/'last_model.zip'),
             source_model_sha256=job['source_model_sha256'], source_step=source_step,
             source_episodes=job['status']['episodes'], source_epoch_counter=job['status']['ppo_epoch_counter'],
             target_timesteps=target, lr_rule='constant_source_final_floor',
             learning_rate=job['settings']['ppo']['learning_rate_floor'],
             initial_boundary='policy+optimizer restored; episodes and global RNG restarted',
             sampler_rule='verify-and-advance-original-balanced-order-from-each-rank-log',
             partial_source_episodes='discard up to 8 unfinished episodes; not counted as failures',
             later_recovery='native optimizer-boundary full environment/sampler/RNG/optimizer recovery',
             test_access='source experiments have already been evaluated; continuation is a separate diagnostic')
    settings = deepcopy(job['settings'])
    settings.update(total_timesteps=target, device='cuda:0', checkpoint_every_timesteps=cp_interval,
                    retain_completed_resume=True)
    # Record the ACTUAL continuation LR. Original schedule parameters remain in source_5m/.
    settings['ppo']['learning_rate_initial'] = e['learning_rate']
    settings['ppo']['learning_rate_floor'] = e['learning_rate']
    settings['continuation'] = e
    keep = ('algorithm','mask','reward_case','beta','coverage_scale','objective_scale','mapping',
            'initialization','seed','detour_budget_max','budget_observation','mission_time_observation',
            'environment','remaining_step_observation','legacy_scalar_reward_coefficients_active',
            'reward_definition','external_stop')
    spec = {k:deepcopy(job['spec'][k]) for k in keep if k in job['spec']}
    suffix = f'__continue_to_{target}'
    spec.update(run_id=job['spec']['run_id']+suffix,
                configuration_id=job['spec']['configuration_id']+suffix,
                stage='cost11_extended_training', continuation=e,
                fresh_weights=False, source_weights_used=True,
                source_protocol_sha256=job['spec']['nrg1_protocol_sha256'])
    return spec, settings


def patch_train_sb3(source):
    """Patch ONLY continuation loading/counters/LR guards; keep native audits and save logic."""
    tree = ast.parse(source)
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train_sb3']
    require(len(funcs) == 1, 'Exactly one train_sb3 definition is required.')
    n = funcs[0]; old = '\n'.join(source.splitlines()[n.lineno-1:n.end_lineno])+'\n'
    text = old
    patches = [
        ("checkpoints = deepcopy(state['checkpoints']) if state else []",
         "checkpoints = deepcopy(state['checkpoints']) if state else _ext_baseline(settings)"),
        ("episode_count = state['episode_count'] if state else 0",
         "episode_count = state['episode_count'] if state else settings['continuation']['source_episodes']"),
        ("next_cp = state['next_cp'] if state else cp_interval",
         "next_cp = state['next_cp'] if state else (settings['continuation']['source_step'] // cp_interval + 1) * cp_interval"),
        ("next_resume = state['next_resume'] if state else resume_interval",
         "next_resume = state['next_resume'] if state else (settings['continuation']['source_step'] // resume_interval + 1) * resume_interval"),
        ("            super().train()\n", "            _ext_assert(self, spec, settings)\n            super().train()\n            _ext_after_update(self, settings)\n"),
        ("                model = SaveAfterUpdate(**common)",
         "                model = _ext_load_5m(SaveAfterUpdate, env, spec, settings, out, records)"),
        ("            write_json(out / 'model_architecture.json'",
         "            _ext_assert(model, spec, settings)\n            write_json(out / 'model_architecture.json'"),
        ("reset_num_timesteps=not bool(state)", "reset_num_timesteps=False"),
    ]
    for a,b in patches:
        require(text.count(a) == 1, 'Trainer patch target changed: '+a)
        text = text.replace(a,b,1)
    ast.parse(text)
    require('model = SaveAfterUpdate(**common)' not in text, 'Fresh-policy branch was not removed.')
    diff = ''.join(difflib.unified_diff(old.splitlines(True),text.splitlines(True),
                                      fromfile='original/train_sb3',tofile='continuation/train_sb3'))
    return text, diff


def prepare_case(job, target, cp_interval):
    base = Path(job['case_dir'])
    spec, settings = extension_config(job,target,cp_interval)
    source_text = (Path(job['source_snapshot'])/(TRAINER+'.py')).read_text(encoding='utf-8')
    patched, diff = patch_train_sb3(source_text)
    protocol = dict(version=VERSION, source_model_sha256=job['source_model_sha256'],
                    source_protocol_sha256=job['spec']['nrg1_protocol_sha256'],
                    source_manifest_sha256=job['manifest_sha256'], snapshot_hash=job['runtime_hashes'],
                    request=dict(spec=spec,settings=settings,gpu=job['gpu']),
                    launcher_sha256=job['launcher_sha256'], patched_function_sha256=hashlib.sha256(patched.encode()).hexdigest())
    if base.exists():
        require(not base.is_symlink() and (base/'continuation_protocol.json').is_file(),
                'Unrecognized existing output case: '+str(base))
        freeze(base/'continuation_protocol.json',protocol)
    else:
        with tempfile.TemporaryDirectory(prefix='.'+job['key']+'_prepare_',dir=base.parent) as tmp:
            tmp = Path(tmp); (tmp/'source_snapshot').mkdir(); (tmp/'source_5m').mkdir()
            for name,h in job['runtime_hashes'].items():
                shutil.copy2(Path(job['source_snapshot'])/name,tmp/'source_snapshot'/name)
                require(sha(tmp/'source_snapshot'/name)==h,'Snapshot copy failed: '+name)
            for name in ['run_spec.json','training_settings.json','status.json','checkpoints.json',
                         'software.json','training_updates.csv','training_episodes.csv','last_model.zip'] + \
                        [f'map_order_rank{i}.csv' for i in range(job['settings']['num_envs'])]:
                shutil.copy2(Path(job['source_run'])/name,tmp/'source_5m'/name)
            shutil.copy2(Path(job['source_root'])/'manifest.csv',tmp/'manifest.csv')
            shutil.copy2(Path(job['source_root'])/'protocol.json',tmp/'source_5m'/'original_protocol.json')
            atomic_text(tmp/'patches'/'train_sb3_continuation.py',patched)
            atomic_text(tmp/'patches'/'source_patch.diff',diff)
            write_json(tmp/'continuation_protocol.json',protocol)
            write_json(tmp/'config.json',dict(environment=spec['environment'],training=settings,
                                              reward_definition=spec['reward_definition'],continuation=settings['continuation']))
            os.replace(tmp,base)
    for name,h in job['runtime_hashes'].items():
        require(sha(base/'source_snapshot'/name)==h,'Extension snapshot modified: '+name)
    for name in ['run_spec.json','training_settings.json','status.json','checkpoints.json',
                 'software.json','training_updates.csv','training_episodes.csv','last_model.zip'] + \
                [f'map_order_rank{i}.csv' for i in range(job['settings']['num_envs'])]:
        expected = job['protected'][str(Path(job['source_run'])/name)]
        require(sha(base/'source_5m'/name)==expected,'Copied source artifact changed: '+name)
    require(sha(base/'source_5m'/'last_model.zip')==job['source_model_sha256'],'Copied 5M model changed.')
    require(sha(base/'manifest.csv')==job['manifest_sha256'],'Copied manifest changed.')
    require((base/'patches'/'train_sb3_continuation.py').read_text()==patched,'Continuation patch changed.')
    spec['continuation_protocol_sha256'] = sha(base/'continuation_protocol.json')
    return spec, settings


def import_runtime(snapshot, hashes):
    snapshot = Path(snapshot).resolve(); sys.dont_write_bytecode = True
    sys.path.insert(0,str(snapshot)); importlib.invalidate_caches()
    modules = []
    for name in [TRAINER,'full_v2_utils',CORE,FEATURES,LOGGER,'resume_ch4']:
        if name in sys.modules:
            require(Path(sys.modules[name].__file__).resolve().parent == snapshot,
                    'Each model must use a separate worker process: '+name)
        m = importlib.import_module(name)
        require(Path(m.__file__).resolve().parent==snapshot,'Imported non-frozen module: '+name)
        modules.append(m)
    for m in list(sys.modules.values()):
        f = getattr(m,'__file__',None)
        if f and Path(f).suffix=='.py' and Path(f).resolve().parent==snapshot:
            require(Path(f).name in hashes and sha(f)==hashes[Path(f).name],
                    'Unverified imported local dependency: '+str(f))
    trainer, utils, core, features, logger, recovery = modules
    require(logger.C3NT_VERSION==core.VERSION and core.COST_MULTIPLIER==11.0
            and core.NORMALIZATION_STEP_FACTOR==5.0,'Wrong original COST11 core/logger.')
    logger.install(trainer,utils)
    return SimpleNamespace(trainer=trainer,utils=utils,core=core,features=features,recovery=recovery)


def runtime_check(job):
    import torch
    for name in ('torch','numpy','gymnasium','stable-baselines3','sb3-contrib'):
        actual = importlib.metadata.version(name)
        require(actual == job['software'][name],
                f'Original {name}={job["software"][name]}, current={actual}. Use the original training environment; no automatic upgrades.')
    require(sys.version_info >= (3,10), 'Python >=3.10 is required.')
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,
            'The assigned physical GPU must be the only visible CUDA device.')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')==job['gpu']['uuid'], 'GPU UUID/worker mismatch.')
    require(torch.version.cuda==job['software']['cuda_version'], 'Original CUDA runtime changed.')
    if job['settings'].get('tensorboard'):
        require(importlib.util.find_spec('tensorboard') is not None, 'Original TensorBoard dependency is missing.')
    torch.set_num_threads(int(job['settings'].get('torch_threads',4)))
    return dict(device='cuda:0',physical_gpu=job['gpu'],torch=str(torch.__version__),
                cuda_runtime=torch.version.cuda,python=sys.version,versions={n:importlib.metadata.version(n)
                for n in ('numpy','gymnasium','stable-baselines3','sb3-contrib')})


def training_records(rt, job):
    """Resolve metadata in the ORIGINAL manifest location; only train arrays are loaded."""
    import numpy as np
    rr = rt.utils.load_manifest(Path(job['source_root'])/'manifest.csv',50,verify_files=False)
    rr = [r for r in rr if r.split=='train' and r.source=='synthetic']
    require(len(rr)==900,'Exactly the original 900 synthetic training maps are required.')
    require(rt.trainer.run_identity(job['spec'],job['settings'],rr)==job['status']['identity'],
            'Original trainer identity does not match train record order/spec/settings.')
    from coverage_core import validate_priority_map
    for i,rec in enumerate(rr,1):
        require(Path(rec.path).is_file(),'Original training map is missing: '+rec.path)
        a = validate_priority_map(np.load(rec.path,allow_pickle=False),50,min_active=2)
        require(rt.utils.array_hash(a)==rec.sha256,'Training map hash changed: '+rec.map_id)
        if i % 300 == 0: print(f'[MAP CHECK] {job["key"]}: {i}/{len(rr)} train arrays verified',flush=True)
    return rr


def rebuild_orders(rt, records, source_dir, original_settings, seed):
    """Replay sampler draws only. Do not replay actions, infer masks, or consume global RNG."""
    states, evidence = [], []
    for rank in range(original_settings['num_envs']):
        p = Path(source_dir)/f'map_order_rank{rank}.csv'; log = rows(p)
        require(log,'Empty sampler log: '+str(p))
        order = rt.utils.BalancedMapOrder(records,original_settings['map_sequence_seed']+seed*1009+rank)
        for i,r in enumerate(log,1):
            require(int(r['rank'])==rank and int(r['episode'])==i and int(r['sequence_position'])==i,
                    'Sampler log is not a contiguous original sequence: '+str(p))
            rec = records[order.next_index()]
            require(rec.map_id==r['map_id'] and rec.pattern==r['pattern'] and str(rec.tier)==str(r['tier'])
                    and rec.source==r['source'],'Original balanced sampler/log mismatch: '+str(p))
        states.append(dict(order=order.state_dict(),episode=len(log)))
        evidence.append(dict(rank=rank,verified_draws=len(log),last_map_id=log[-1]['map_id'],
                             next_episode=len(log)+1,source_log_sha256=sha(p)))
    return states,evidence


def tensor_tree_digest(obj):
    """Exact portable digest of numeric tensors/scalars (not pickle byte layout)."""
    import torch
    h = hashlib.sha256()
    def rec(x):
        if isinstance(x,torch.Tensor):
            y=x.detach().cpu().contiguous()
            h.update(('T'+str(y.dtype)+str(tuple(y.shape))).encode()); h.update(y.numpy().tobytes())
        elif isinstance(x,dict):
            h.update(b'D')
            for k in sorted(x,key=lambda v:(type(v).__name__,repr(v))): rec(k);rec(x[k])
        elif isinstance(x,(list,tuple)):
            h.update(('L'+str(len(x))).encode())
            for y in x: rec(y)
        else:
            h.update((type(x).__name__+':'+repr(x)+';').encode())
    rec(obj); return h.hexdigest()


def assert_model(model, settings, spec, source=False):
    # Explicitly reject a swapped spec/settings pair rather than hide the error
    # with defaults. The original trainer's argument order differs from this API.
    require(isinstance(settings, dict) and isinstance(settings.get('ppo'), dict)
            and isinstance(settings.get('continuation'), dict),
            'assert_model expected training settings with ppo/continuation; '
            'the spec/settings call contract is wrong or metadata is incomplete.')
    require(isinstance(spec, dict) and 'seed' in spec and 'environment' in spec,
            'assert_model expected a run spec, not training settings.')
    import numpy as np
    p=settings['ppo']; e=settings['continuation']
    for k in ('n_steps','batch_size','n_epochs','gamma','gae_lambda','ent_coef','vf_coef','max_grad_norm','target_kl'):
        require(getattr(model,k)==p[k], 'Model hyperparameter changed: '+k)
    require(model.n_envs==settings['num_envs'] and model.seed==spec['seed'], 'Model n_envs/seed mismatch.')
    require(model.normalize_advantage is True and model.clip_range_vf is None,'PPO normalization/clipping changed.')
    for fraction in (0.0,0.5,1.0):
        require(math.isclose(float(model.clip_range(fraction)),p['clip_range'],abs_tol=1e-12), 'clip_range changed.')
    require(e['source_step'] <= model.num_timesteps <= e['target_timesteps']+model.n_steps*model.n_envs,
            'Continuation global step lies outside the planned range.')
    require(set(model.observation_space.spaces)=={'map','heading','detour_budget'} and
            model.observation_space['map'].shape==(6,50,50),'Wrong observation space.')
    require(np.all(model.observation_space['detour_budget'].low==0) and
            np.all(model.observation_space['detour_budget'].high==1),'Budget observation changed.')
    require(getattr(model,'cost11_continuation_source_sha256',None)==e['source_model_sha256'],
            'Missing/wrong continuation provenance in model.')
    for progress in (1.0,.75,.5,0.0,-.001):
        require(math.isclose(float(model.lr_schedule(progress)),e['learning_rate'],rel_tol=0,abs_tol=1e-12),
                'LR schedule restarted/reannealed instead of preserving the source floor.')
    for group in model.policy.optimizer.param_groups:
        require(math.isclose(float(group['lr']),e['learning_rate'],rel_tol=0,abs_tol=1e-12),
                'Optimizer LR does not match continuation LR.')
    if source:
        require(model.num_timesteps==e['source_step'] and model._n_updates==e['source_epoch_counter'],
                'Source step/update count not restored.')


def assert_from_trainer(model, spec, settings):
    """Bridge the AUTHENTIC generated trainer's (model, spec, settings) call.

    Keep the generated train_sb3 bytes and the existing run identity unchanged.
    assert_model itself uses (model, settings, spec); keywords prevent swapping.
    Both the post-load and pre-optimizer call sites use this one bridge.
    """
    return assert_model(model=model, settings=settings, spec=spec)


def preflight_generated_guards(rt, model, spec, settings):
    """Exercise BOTH actual generated guard call expressions, without updates.

    Only _ext_assert calls extracted from the hash-verified original trainer are
    executed. No arbitrary source or user-supplied expressions are evaluated.
    """
    source = Path(rt.trainer.__file__).read_text(encoding='utf-8')
    patched, _ = patch_train_sb3(source)
    calls = [n for n in ast.walk(ast.parse(patched)) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == '_ext_assert']
    require(len(calls) == 2, 'Expected exactly two generated assertion call sites.')
    names = set()
    for call in calls:
        require(not call.keywords and len(call.args) == 3 and
                all(isinstance(a, ast.Name) for a in call.args),
                'Unexpected generated assertion syntax.')
        first, second, third = [a.id for a in call.args]
        require(first in {'self', 'model'} and second == 'spec' and third == 'settings',
                'Generated assertion argument contract differs from the verified trainer.')
        names.add(first)
        values = {'self': model, 'model': model, 'spec': spec, 'settings': settings}
        assert_from_trainer(*[values[a.id] for a in call.args])
    require(names == {'self', 'model'}, 'Missing post-load or pre-update assertion.')
    return dict(generated_guard_sites_checked=len(calls),
                bridge='(model,spec,settings) -> keyword arguments to assert_model',
                optimizer_updates_performed=0)


def load_5m(cls, env, spec, settings, out, records, rt, job, orders):
    """Load policy AND optimizer; keep source tensor values and accumulated Adam moments."""
    import torch
    from stable_baselines3.common.save_util import load_from_zip_file
    source=Path(settings['continuation']['source_model']);e=settings['continuation']
    require(sha(source)==e['source_model_sha256'],'Source model copy hash mismatch.')
    # Read tensor states independently, before editing LR/provenance. No policy-only warm start.
    _,params,_=load_from_zip_file(source,device='cpu',load_data=False)
    require(params and 'policy' in params and 'policy.optimizer' in params,
            'Full policy and optimizer are required.')
    expected=tensor_tree_digest(params)
    model=cls.load(source,env=env,device=settings['device'],force_reset=True)
    require(tensor_tree_digest(model.get_parameters())==expected,'Policy/optimizer state was not restored exactly.')
    optim=model.policy.optimizer.state_dict()
    require(optim['state'] and all('step' in s and float(s['step'])>0 for s in optim['state'].values()),
            'Optimizer moments/step counters are missing; refusing a reset optimizer.')
    # Both saved learning_rate and constructed lr_schedule must be replaced. On later loads
    # SB3 rebuilds lr_schedule from learning_rate, so changing lr_schedule alone is insufficient.
    model.learning_rate=float(e['learning_rate']);model._setup_lr_schedule()
    for group in model.policy.optimizer.param_groups: group['lr']=float(e['learning_rate'])
    require(tensor_tree_digest(model.get_parameters())==expected,
            'Source floor differs from optimizer LR or tensor states changed during initialization.')
    model.tensorboard_log=str(Path(out)/'tensorboard') if settings.get('tensorboard') else None
    model.cost11_continuation_source_sha256=e['source_model_sha256']
    model.cost11_continuation_source_step=e['source_step']
    model.cost11_continuation_target=e['target_timesteps']
    model.cost11_continuation_lr_rule=e['lr_rule']
    model._last_obs=None;model._last_episode_starts=None
    # Only log/audit accumulators are reset; policy/optimizer remain untouched.
    for name in ('_c3nt_pending_rewards','_cost5_diag_raw','_cost5_diag_bootstrap'):
        if hasattr(model,name):
            require(len(getattr(model,name))==0,'Source was not saved after a complete rollout/update: '+name)
    for rank,adapter in enumerate(env.envs):
        adapter.stream.order.load_state_dict(orders[rank]['order'])
        adapter.stream.episode=orders[rank]['episode']
        adapter.stream.total_steps=e['source_step']//settings['num_envs']
    assert_model(model,settings,spec,source=True)
    rt.features.audit_cnn3_model(model);rt.core.audit_model(model)
    write_json(Path(out)/'continuation_load_audit.json',dict(
        source_path=str(source),source_sha256=e['source_model_sha256'],
        source_step=model.num_timesteps,source_update_counter=model._n_updates,
        policy_and_optimizer_tensor_digest=expected,exact_tensor_restore=True,
        optimizer_parameter_states=len(optim['state']),
        optimizer_step_min=min(float(s['step']) for s in optim['state'].values()),
        optimizer_step_max=max(float(s['step']) for s in optim['state'].values()),
        learning_rate=e['learning_rate'],sampler_positions=[x['episode'] for x in orders],
        full_source_environment_rng_restored=False,subsequent_full_recovery_enabled=True))
    print(f'[LOAD 5M] {job["key"]}: step={model.num_timesteps:,}; policy+Adam restored; LR={e["learning_rate"]:g}',flush=True)
    print('[BOUNDARY] New episodes/RNG at source boundary; verified balanced sampler continues after original draws.',flush=True)
    return model


def install_continuation(rt, job, spec, settings, records, orders):
    source=Path(rt.trainer.__file__).read_text(encoding='utf-8');patched,_=patch_train_sb3(source)
    def baseline(s):
        e=s['continuation']
        return [dict(path=e['source_model'],num_timesteps=e['source_step'],
                     sha256=e['source_model_sha256'],final=False,source_baseline=True)]
    def load(cls,env,sp,st,out,rr):
        return load_5m(cls,env,sp,st,out,rr,rt,job,orders)
    def after(model,st):
        e=st['continuation']
        for k in ('train/learning_rate','train/approx_kl','train/clip_fraction','train/value_loss','train/entropy_loss'):
            value=model.logger.name_to_value.get(k)
            require(value is not None and math.isfinite(float(value)),'Nonfinite/missing update metric: '+k)
        require(math.isclose(float(model.logger.name_to_value['train/learning_rate']),e['learning_rate'],abs_tol=1e-12),
                'Actual optimizer update used an unexpected LR.')
    rt.trainer.__dict__.update(_ext_baseline=baseline,_ext_load_5m=load,
                               _ext_assert=assert_from_trainer,_ext_after_update=after)
    exec(compile(patched,str(Path(job['case_dir'])/'patches'/'train_sb3_continuation.py'),'exec'),rt.trainer.__dict__)


def preflight_model(rt,job,records,target=20_000_000,cp_interval=1_000_000):
    import numpy as np
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.save_util import load_from_zip_file
    from gymnasium import spaces
    model=MaskablePPO.load(job['source_model'],device='cuda:0')
    _,params,_=load_from_zip_file(job['source_model'],device='cpu',load_data=False)
    require(tensor_tree_digest(model.get_parameters())==tensor_tree_digest(params),'Preflight policy/optimizer mismatch.')
    require(model.num_timesteps==job['status']['actual_timesteps'],'Preflight source step mismatch.')
    require(model.policy.optimizer.state_dict()['state'],'Source optimizer has no state.')
    rt.features.audit_cnn3_model(model);rt.core.audit_model(model)
    cfg=rt.trainer.ExperimentEnvConfig(**job['spec']['environment']); core=rt.trainer.make_core(cfg)
    require(isinstance(core,rt.core.C3NoClockMixin),'Original replacement-reward adapter is not active.')
    a=np.load(records[0].path,allow_pickle=False)
    obs,info=core.reset(a,seed=42,map_name=records[0].map_id)
    require(set(obs)==set(model.observation_space.spaces),'Source model/core observation mismatch.')
    require(core.config.mask_mode==CASES[job['key']]['mask'],'Wrong source mask.')
    mask=core.action_masks()
    act,_=model.predict(obs,deterministic=True,action_masks=mask)
    nxt,reward,term,trunc,info=core.step(int(act))
    require(bool(info['nrg_reward_replaced']) and math.isfinite(float(reward)), 'Core/model smoke check failed.')
    # Prepare continuation-only attributes on this in-memory preflight model.
    # No source model is saved or modified, and no optimizer step is taken.
    sp, st = extension_config(job, target, cp_interval)
    e = st['continuation']
    before = tensor_tree_digest(model.get_parameters())
    model.learning_rate = float(e['learning_rate'])
    model._setup_lr_schedule()
    for group in model.policy.optimizer.param_groups:
        group['lr'] = float(e['learning_rate'])
    model.cost11_continuation_source_sha256 = e['source_model_sha256']
    guards = preflight_generated_guards(rt, model, sp, st)
    assert_model(model=model, settings=st, spec=sp, source=True)
    require(tensor_tree_digest(model.get_parameters()) == before,
            'Preflight unexpectedly changed policy/optimizer tensor states.')
    return dict(source_step=model.num_timesteps,optimizer_states=len(model.policy.optimizer.state_dict()['state']),
                core_version=rt.core.VERSION,mask_mode=cfg.mask_mode,beta=cfg.tradeoff_beta,
                one_action_smoke_passed=True,optimizer_updates_performed=0,
                continuation_guards=guards)


def _launcher_acceptance(plan):
    actual = sha(Path(__file__).resolve())
    prior = plan.get('launcher_sha256')
    require(prior in {actual, KNOWN_V1_LAUNCHER_SHA256},
            'This release accepts only its own plan or the exact known v1 launcher. '
            'No unknown code revision or hash is silently accepted.')
    require(plan.get('version') == VERSION and
            all(j.get('launcher_sha256') == prior for j in plan['jobs'].values()),
            'Inconsistent frozen launcher revision in plan/jobs.')
    return dict(bugfix=BUGFIX_VERSION, plan_sha256=sha(Path(plan['out'])/'continuation_plan.json'),
                plan_launcher_sha256=prior, active_launcher_sha256=actual,
                compatibility='known-v1-assertion-bridge-only' if prior != actual else 'native-fix1',
                frozen_plan_rewritten=False, per_case_protocol_rewritten=False,
                training_settings_changed=False, generated_train_function_changed=False)


def bind_continuation_plan(out, proposed):
    """Authorize the one known bugfix WITHOUT changing an existing run identity.

    Existing source, target, checkpoint interval, GPU UUID, runtime hashes, and
    per-run settings must all match. Only the top/job launcher hashes may differ.
    The old plan and continuation protocols remain byte-for-byte untouched;
    active_launcher.json records the real executable hash used by every worker.
    Caller must hold the existing parent.lock.
    """
    out = Path(out)
    path = out/'continuation_plan.json'
    if path.exists():
        require(not path.is_symlink(), 'Symlink continuation plan.')
        saved = read_json(path)
        if saved != proposed:
            require(saved.get('launcher_sha256') == KNOWN_V1_LAUNCHER_SHA256,
                    'Frozen plan changed beyond the one known v1 launcher fix.')
            compatible = deepcopy(proposed)
            compatible['launcher_sha256'] = KNOWN_V1_LAUNCHER_SHA256
            for job in compatible['jobs'].values():
                job['launcher_sha256'] = KNOWN_V1_LAUNCHER_SHA256
            require(saved == compatible,
                    'Source/config/target/GPU changed: fix1 will not rewrite the frozen experiment. '
                    'Keep the original root, output path, GPUs and training target.')
        plan = saved
    else:
        freeze(path, proposed)
        plan = proposed
    acceptance = _launcher_acceptance(plan)
    executable = out/'code_revisions'/(acceptance['active_launcher_sha256']+'.py')
    if executable.exists():
        require(not executable.is_symlink() and sha(executable)==acceptance['active_launcher_sha256'],
                'Saved active launcher was modified.')
    else:
        executable.parent.mkdir(parents=True, exist_ok=True)
        atomic_text(executable, Path(__file__).read_text(encoding='utf-8'))
        require(sha(executable)==acceptance['active_launcher_sha256'],
                'Active launcher copy/hash mismatch.')
    freeze(out/'active_launcher.json', acceptance)
    print('[ASSERT BRIDGE FIX] post-load and pre-update argument order corrected.',flush=True)
    if acceptance['compatibility']=='known-v1-assertion-bridge-only':
        print('[COMPATIBLE RETRY] Existing 20M plan/protocols retained unchanged. '
              'New executable hash recorded in active_launcher.json; no run deleted.',flush=True)
    return plan


def verify_active_launcher(plan):
    path = Path(plan['out'])/'active_launcher.json'
    require(path.is_file() and not path.is_symlink(), 'Missing active launcher acceptance.')
    require(read_json(path) == _launcher_acceptance(plan),
            'Active launcher or frozen plan changed after parent preflight.')


def audit_known_startup_retry(job, spec):
    """Document v1's exact pre-learning crash; never clear trained/recovery state."""
    run = Path(job['case_dir'])/'runs'/spec['run_id']
    status_file = run/'status.json'
    if not status_file.is_file():
        return
    status = read_json(status_file)
    if status.get('state')!='failed' or status.get('error')!="KeyError('ppo')":
        return  # Normal native full-state recovery remains untouched.
    if (run/'resume'/'latest.json').exists():
        return  # An actual committed state must be handled by the original loader.
    require('_ext_assert(model, spec, settings)' in status.get('traceback',''),
            'KeyError context differs; refusing to claim no extension training occurred.')
    work = [run/n for n in ('training_updates.csv','training_episodes.csv','checkpoints.json',
                            'last_model.zip','last_model.pt') if (run/n).exists()]
    work += list((run/'checkpoints').glob('*')) if (run/'checkpoints').exists() else []
    require(not work,
            'The known startup failure has unexpected learning artifacts. Keep all files; '
            'do not reset this run automatically: '+str(work))
    audit = run/'continuation_load_audit.json'
    require(audit.is_file() and read_json(audit).get('source_step')==job['status']['actual_timesteps']
            and read_json(audit).get('source_sha256')==job['source_model_sha256'],
            'Cannot verify the original 5M load at the failed startup.')
    backup = Path(job['case_dir'])/'repair_audit'/'v1_assert_ppo'
    for p in (status_file, audit):
        q = backup/p.name
        if q.exists():
            require(sha(q)==sha(p), 'Prior startup-failure audit differs.')
        else:
            q.parent.mkdir(parents=True,exist_ok=True)
            require(not q.is_symlink(), 'Symlink repair-audit destination.')
            shutil.copy2(p,q)
    print(f'[RETRY VERIFIED] {job["key"]}: v1 failed before model.learn(); '
          f'reload original {job["status"]["actual_timesteps"]:,}-step model. '
          'No trained state or source files removed.',flush=True)


def worker(plan_path,key,phase):
    plan=read_json(plan_path);job=plan['jobs'][key]
    verify_protected(job)
    verify_active_launcher(plan)
    versions=runtime_check(job)
    case=Path(job['case_dir'])
    rt=import_runtime(Path(job['source_snapshot']) if phase=='check' else case/'source_snapshot',job['runtime_hashes'])
    require(rt.core.VERSION==CORE_VERSIONS[CASES[key]['mask']],'Wrong frozen core version for '+key)
    rr=training_records(rt,job)
    orders,evidence=rebuild_orders(rt,rr,Path(job['source_run']) if phase=='check' else case/'source_5m',
                                   job['settings'],job['spec']['seed'])
    require(sum(x['episode'] for x in orders)==job['status']['episodes']+job['settings']['num_envs'],
            'Original sampler draws/completed-episode count disagree.')
    if phase=='check':
        report=preflight_model(rt,job,rr,plan['target_timesteps'],plan['checkpoint_every_timesteps'])
        write_json(Path(plan['out'])/'checks'/(key+'.json'),dict(status='PASS',runtime=versions,
                   model=report,sampler=evidence,plan_sha256=sha(plan_path)))
        verify_protected(job)
        print(f'[CHECK PASS] {key}; physical GPU {job["gpu"]["index"]}',flush=True)
        return
    spec,settings=extension_config(job,plan['target_timesteps'],plan['checkpoint_every_timesteps'])
    spec['continuation_protocol_sha256']=sha(case/'continuation_protocol.json')
    write_json(case/'sampler_continuation.json',dict(rule=settings['continuation']['sampler_rule'],ranks=evidence))
    write_json(case/'runtime_at_start.json',versions)
    install_continuation(rt,job,spec,settings,rr,orders)
    try:
        audit_known_startup_retry(job, spec)
        result=rt.trainer.train_run(spec,settings,rr,case/'runs'/spec['run_id'])
    except rt.recovery.TrainingPaused as e:
        print(f'[PAUSED SAFELY] {key} at {e.step:,}; rerun the same command.',flush=True)
        raise SystemExit(130)
    finally:
        verify_protected(job)
    require(result.get('state')=='completed','Training did not finish.')
    print(f'[TRAIN COMPLETE] {key}: actual_step={result["actual_timesteps"]:,}; validation/test NOT run.',flush=True)


def launch_workers(plan_path,phase):
    plan=read_json(plan_path);processes=[];interrupted=[False]
    previous={}
    def handle(sig,frame):
        if interrupted[0]:
            for p,_,_ in processes:
                if p.poll() is None: p.kill()
            raise KeyboardInterrupt('Second interrupt: use last committed recovery.')
        interrupted[0]=True
        print('\n[STOP] Safe stop requested for all workers; waiting for current rollout/update and recovery save.',flush=True)
        for p,_,_ in processes:
            if p.poll() is None: p.send_signal(signal.SIGTERM)
    for sig in (signal.SIGINT,signal.SIGTERM):
        previous[sig]=signal.getsignal(sig);signal.signal(sig,handle)
    try:
        for key,job in plan['jobs'].items():
            if interrupted[0]: break
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=job['gpu']['uuid']
            env['CUDA_DEVICE_ORDER']='PCI_BUS_ID';env['PYTHONUNBUFFERED']='1';env['PYTHONDONTWRITEBYTECODE']='1'
            env['OMP_NUM_THREADS']=str(job['settings'].get('torch_threads',4))
            env['MKL_NUM_THREADS']=env['OMP_NUM_THREADS']
            logpath=Path(plan['out'])/'logs'/f'{key}_{phase}.log';logpath.parent.mkdir(parents=True,exist_ok=True)
            require(not logpath.is_symlink(),'Symlink log output.')
            log=logpath.open('a',encoding='utf-8',buffering=1)
            log.write('\n=== '+time.strftime('%Y-%m-%d %H:%M:%S')+' '+phase+' ===\n');log.flush()
            p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--_worker',key,
                                '--_plan',str(plan_path),'--phase',phase],env=env,
                               stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            processes.append((p,log,(key,logpath)))
            print(f'[LAUNCH {phase.upper()}] GPU {job["gpu"]["index"]}: {job["label"]}\n  log: {logpath}',flush=True)
        failed=[]
        for p,log,(key,logpath) in processes:
            rc=p.wait();log.close()
            print(f'[{phase.upper()} {"DONE" if rc==0 else "STOPPED"}] {key}; exit={rc}',flush=True)
            if rc:
                failed.append((key,rc))
                tail=logpath.read_text(encoding='utf-8',errors='replace').splitlines()[-18:]
                print('\n'.join(tail),flush=True)
        return 130 if interrupted[0] else (1 if failed else 0)
    finally:
        # On an unexpected parent error, never leave a silent background training job.
        for p,log,_ in processes:
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)
        for p,log,_ in processes:
            if p.poll() is None: p.wait()
            if not log.closed: log.close()
        for sig,h in previous.items(): signal.signal(sig,h)


def report(out):
    out=Path(out).resolve();plan=read_json(out/'continuation_plan.json')
    summary=[];checkpoints=[]
    for key,job in plan['jobs'].items():
        spec,settings=extension_config(job,plan['target_timesteps'],plan['checkpoint_every_timesteps'])
        run=Path(job['case_dir'])/'runs'/spec['run_id'];status=read_json(run/'status.json') if (run/'status.json').is_file() else {}
        update_rows=rows(run/'training_updates.csv') if (run/'training_updates.csv').is_file() else []
        last=update_rows[-1] if update_rows else {}
        step=int(float(last.get('timesteps',job['status']['actual_timesteps'])))
        row=dict(model=key,state=status.get('state','not_started'),source_step=job['status']['actual_timesteps'],
                 latest_logged_step=step,requested_total=plan['target_timesteps'],
                 extension_updates=len(update_rows),learning_rate=last.get('learning_rate'),
                 approx_kl=last.get('approx_kl'),entropy_loss=last.get('entropy_loss'),
                 explained_variance=last.get('explained_variance'),physical_gpu=job['gpu']['index'],
                 run_dir=str(run),validation_executed=False,test_executed=False)
        summary.append(row)
        if (run/'checkpoints.json').is_file():
            for cp in read_json(run/'checkpoints.json'):
                checkpoints.append(dict(model=key,step=cp['num_timesteps'],path=cp['path'],
                                        sha256=cp['sha256'],final=cp.get('final',False),
                                        source_baseline=cp.get('source_baseline',False),
                                        exists=Path(cp['path']).is_file()))
    write_csv(out/'training_status_summary.csv',summary)
    write_csv(out/'checkpoints_for_validation.csv',checkpoints)
    print('\n=== Continuation status (training logs, NOT validation) ===',flush=True)
    for row in summary:
        print(f'{row["model"]:16s} {row["state"]:12s} step={row["latest_logged_step"]:,}/{plan["target_timesteps"]:,}',flush=True)
    allowed={'.json','.csv','.py','.diff','.txt','.md'}
    zip_path=out/'UPLOAD_cost11_20m_training_logs.zip'
    temp=zip_path.with_suffix('.writing.zip')
    require(not zip_path.is_symlink(),'Symlink upload ZIP output.')
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for directory,dirs,files in os.walk(out,followlinks=False):
            dirs[:]=sorted(d for d in dirs if d not in {'resume','checkpoints','tensorboard','logs','__pycache__','.locks','.run_locks'}
                           and not (Path(directory)/d).is_symlink())
            for name in sorted(files):
                p=Path(directory)/name
                if p.suffix.lower() in allowed and p.is_file() and not p.is_symlink():
                    z.write(p,p.relative_to(out).as_posix())
    os.replace(temp,zip_path)
    print(f'[UPLOAD TRAINING LOGS] {zip_path}',flush=True)


def arguments(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root',default='.',help='Suite parent containing the original three experiments.')
    p.add_argument('--out',default=DEFAULT_OUT,help='New extension root. Never use an original 5M root.')
    p.add_argument('--phase',choices=['check','train','report'],default='train')
    p.add_argument('--gpus',nargs=3,type=int,default=[1,2,3],metavar=('RELAXED05','RELAXED0','STRICT05'))
    p.add_argument('--target-steps',type=int,default=20_000_000,help='TOTAL lifetime environment interactions, NOT additional.')
    p.add_argument('--checkpoint-every',type=int,default=1_000_000,help='Model saves at global boundaries (one rollout overshoot).')
    for name in CASES: p.add_argument('--'+name.replace('_','-')+'-root',default=None)
    p.add_argument('--_worker',choices=list(CASES),help=argparse.SUPPRESS)
    p.add_argument('--_plan',help=argparse.SUPPRESS)
    return p.parse_args(argv)


def main(argv=None):
    args=arguments(argv)
    require(os.name=='posix','This launcher is for the original Linux GPU server.')
    if args._worker:
        worker(Path(args._plan),args._worker,args.phase);return 0
    require(not Path(args.out).expanduser().is_symlink(), 'Output root must not be a symlink.')
    out=Path(args.out).expanduser().resolve();base=Path(args.root).expanduser().resolve()
    if args.phase=='report':
        with lock(out/'.locks'/'parent.lock'): report(out)
        return 0
    require(base.is_dir(),'Source suite does not exist: '+str(base))
    require(not out.is_symlink() and out!=base and not within(base,out), 'Unsafe output/source overlap.')
    require(args.target_steps>5_005_312 and args.checkpoint_every>0,'Target must exceed 5M and checkpoint interval must be positive.')
    require(args.checkpoint_every>=8192,'Checkpoint interval must not be smaller than one original rollout.')
    jobs={};gpulist=gpu_inventory(args.gpus);script_hash=sha(Path(__file__).resolve())
    # Finish all source checks BEFORE creating any training run or launching any learning.
    for (key,c),gpu in zip(CASES.items(),gpulist):
        source=locate(base,key,getattr(args,key+'_root'),out)
        require(not within(out,source) and not within(source,out),'Output overlaps an original source: '+str(source))
        j=inspect_source(source,key);j.update(case_dir=str(out/key),gpu=gpu,launcher_sha256=script_hash)
        jobs[key]=j
    plan=dict(version=VERSION,out=str(out),source_suite=str(base),target_timesteps=args.target_steps,
              checkpoint_every_timesteps=args.checkpoint_every,launcher_sha256=script_hash,jobs=jobs,
              mode='checkpoint_continuation_not_fresh_0_to_20m',learning_rate_rule='keep_original_final_floor',
              validation_during_training=False,automatic_validation=False,automatic_test=False)
    require(not out.exists() or (out/'continuation_plan.json').is_file() or not any(out.iterdir()),
            'Output is nonempty and is not this continuation experiment: '+str(out))
    out.mkdir(parents=True,exist_ok=True)
    with lock(out/'.locks'/'parent.lock'):
        plan=bind_continuation_plan(out,plan)
        jobs=plan['jobs']
        for key,j in jobs.items():
            print(f'[SOURCE] {key}: {j["source_model"]}',flush=True)
        print(f'[TOTAL TARGET] {args.target_steps:,}; source=5,005,312; added request={args.target_steps-5_005_312:,}',flush=True)
        print('[LR] source floor 5e-5 held constant. gamma=1; FULL COST11; original mask/CNN/optimizer retained.',flush=True)
        # All three GPU preflights must succeed before ANY training worker is launched.
        rc=launch_workers(out/'continuation_plan.json','check')
        if rc: return rc
        if args.phase=='check':
            print('[ALL CHECKS PASSED] No optimizer updates performed. Run without --phase check to train.',flush=True)
            return 0
        estimated=sum(Path(j['source_model']).stat().st_size *
                      (math.ceil((args.target_steps-j['status']['actual_timesteps'])/args.checkpoint_every)+7)
                      for j in jobs.values())
        require(shutil.disk_usage(out).free > estimated,
                f'Insufficient disk for planned model checkpoints/recovery (estimated >{estimated/2**30:.2f} GiB).')
        for j in jobs.values(): prepare_case(j,args.target_steps,args.checkpoint_every)
        rc=launch_workers(out/'continuation_plan.json','train')
        report(out)
        if rc==0: print('[DONE] All three models reached the total target. Original 5M files unchanged. No validation/test was run.',flush=True)
        return rc


if __name__=='__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('[INTERRUPTED] Re-run the same command to use the last committed recovery.',file=sys.stderr)
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
