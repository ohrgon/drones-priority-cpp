#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NRG1 CNN3 + no clock + external-cap truncation (combined change).

Add beside the existing suite together with nrg1_c3nt_features.py.
check: real environment/network/save-load audit only, no learn()/optimizer step.
train: fresh/resumed 5M -> final-model validation.
all:   fresh/resumed 5M -> validation -> original test+validation full evaluation.
validation/test: evaluate a completed C3NT run only. No teacher. Cap failure penalty removed.
Uses the existing completed NRG1 source_snapshot; never the current 4-way source.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import csv
import difflib
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

import numpy as np

VERSION = 'nrg1_cnn3_noclock_externalcap_v1_20260925'
NRG_VERSION = 'relaxed_b6_nrg1_20260924_v1'
ORIGINAL_RUN = 'ppo_relaxedB6_NRG1_b0.5_g1_l0.95_budgetobs_timeleft_seed42'
RUN_ID = 'ppo_relaxedB6_NRG1_C3NT_b0.5_g1_l0.95_budgetobs_seed42'
SOURCE_TRAINER = 'trainers_ch4_bounded_b6_nrg1'
PAD_TRAINER = 'trainers_ch4_bounded_b6_nrg1_c3nt'
PACKAGE = ('run_nrg1_c3nt.py', 'nrg1_c3nt_features.py', 'nrg1_c3nt_core.py')
LW = 'priority_weighted_first_visit_latency_sec'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for b in iter(lambda: handle.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def safe(value):
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.ndarray): return safe(value.tolist())
    if isinstance(value, np.generic): return safe(value.item())
    if isinstance(value, dict): return {str(k): safe(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.writing')
    with temp.open('w', encoding='utf-8') as handle:
        json.dump(safe(data), handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    os.replace(temp, path)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    temp = path.with_name(path.name + '.writing')
    with temp.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows: writer.writerow(safe(row))
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temp, path)


def canonical_split(value):
    value = str(value).strip().lower()
    return 'validation' if value in {'val','valid','validation'} else value


def path_from_script(value):
    value = Path(value).expanduser()
    return (value if value.is_absolute() else Path(__file__).resolve().parent/value).resolve()


@contextmanager
def output_lock(path):
    import fcntl
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as handle:
        try: fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Another process is using {path.parent}') from exc
        try: yield
        finally: fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def checked_child(root, name):
    root = Path(root).resolve(); value = root/name
    require(not Path(name).is_absolute() and root in value.resolve().parents,
            f'Unsafe snapshot member: {name}')
    require(value.is_file() and not value.is_symlink(), f'Snapshot file missing/symlink: {value}')
    return value


def checkpoint_data(path):
    # No cloudpickle/unpickling needed to validate scalar metadata.
    with zipfile.ZipFile(path) as archive:
        require(archive.getinfo('data').file_size < 32_000_000, 'Oversized checkpoint metadata.')
        data = json.loads(archive.read('data'))
        version = archive.read('_stable_baselines3_version').decode().strip()
    return data, version


def verify_manifest(path):
    rows = read_csv(path)
    require(rows, 'Empty original manifest.')
    require({'map_id','path','split','source','sha256'}.issubset(rows[0]), 'Wrong manifest schema.')
    for row in rows:
        split = canonical_split(row['split'])
        require(split in {'train','validation','test'}, 'Unknown split: '+split)
        require(row['source'] in {'synthetic','osm'}, 'Unknown source: '+row['source'])
        require(row['source'] != 'osm' or split == 'test', 'OSM must remain test-only.')
        require(Path(row['path']).is_absolute(), 'Frozen NRG1 manifest must contain absolute map paths.')
    train = [r for r in rows if canonical_split(r['split']) == 'train']
    validation = [r for r in rows if canonical_split(r['split']) == 'validation']
    heldout = [r for r in rows if canonical_split(r['split']) != 'train']
    require(train and len(validation) == 180, 'Expected original train + validation 180; no new split generated.')
    for key in ('map_id','path','sha256'):
        # Preserve training records exactly, including any intentional duplicate
        # training content. The original runner only forbids held-out duplicates
        # and train/held-out overlap; do not invent a new split/deduplication.
        require(len({r[key] for r in heldout}) == len(heldout), f'Duplicate held-out identity: {key}')
        require(not {r[key] for r in train} & {r[key] for r in heldout}, f'Train/held-out leakage: {key}')
    return dict(total=len(rows), counts=dict(Counter(r['source']+'/'+canonical_split(r['split']) for r in rows)))


def source_context(root, run_dir=None):
    """Use the actual completed NRG1 experiment, not old 4-way or GUIDED1."""
    root = Path(root).resolve()
    for name in ('protocol.json','manifest.csv'):
        require((root/name).is_file(), f'Completed NRG1 input missing: {root/name}')
    protocol = read(root/'protocol.json')
    require(sha(root/'manifest.csv') == protocol.get('manifest_sha256'), 'Original frozen manifest hash changed.')
    source_hash = protocol.get('snapshot_hash')
    require(isinstance(source_hash,dict) and source_hash, 'Original snapshot hashes missing.')
    snapshot = root/'source_snapshot'
    for name, expected in source_hash.items():
        require(sha(checked_child(snapshot,name)) == expected, 'Original snapshot changed: '+name)
    required_sources = (SOURCE_TRAINER+'.py','run_relaxed_b6_nrg1.py','nrg1_core.py',
                        'coverage_core.py','core_ch4.py','core_ch4_bounded_b6.py',
                        'evaluation_ch4.py','full_v2_utils.py','resume_ch4.py')
    require(all(name in source_hash for name in required_sources), 'Required NRG1 files absent from frozen source list.')
    run = Path(run_dir).resolve() if run_dir else root/'runs'/ORIGINAL_RUN
    require(run.is_dir(), f'Original NRG1 run not found: {run}\nUse --run-dir for its actual folder.')
    for name in ('run_spec.json','training_settings.json','status.json','checkpoints.json','last_model.zip'):
        require((run/name).is_file(), f'Original NRG1 artifact missing: {run/name}')
    spec = read(run/'run_spec.json'); settings = read(run/'training_settings.json'); status = read(run/'status.json')
    require(spec.get('run_id') == ORIGINAL_RUN and spec.get('reward_case') == 'NRG1',
            'Source must be ORIGINAL NRG1, not 4-way/C3NT/GUIDED1.')
    require(spec.get('algorithm') == 'ppo' and spec.get('seed') == 42, 'Expected PPO seed42.')
    require(spec.get('remaining_step_observation') is True, 'Source has no remaining-step observation.')
    require(spec.get('budget_observation') is True and spec.get('mission_time_observation') is False,
            'Original observation flags changed.')
    require(spec.get('legacy_scalar_reward_coefficients_active') is False, 'Legacy reward unexpectedly active.')
    env = spec['environment']
    for k,v in dict(grid_size=50,mask_mode='relaxed',detour_budget_max=6,step_budget_factor=5.0,
                    tradeoff_beta=0.5,coverage_reward_scale=0.0,objective_cost_scale=0.0,
                    success_terminal_reward=0.0,failure_terminal_penalty=0.0,paper_mode='none').items():
        require(env.get(k) == v, f'Unexpected original environment.{k}: {env.get(k)!r}')
    rd = spec['reward_definition']
    require(rd.get('version') == NRG_VERSION and rd.get('success_terminal') == 2.0
            and rd.get('failure_terminal') == -2.0, 'Unexpected source reward definition.')
    require(settings.get('total_timesteps') == 5_000_000, 'Original target must be 5M.')
    require(settings['ppo']['gamma'] == 1.0 and settings['ppo']['gae_lambda'] == 0.95,
            'Expected NRG1 gamma1/lambda.95.')
    require(status.get('state') == 'completed' and status.get('actual_timesteps',0) >= 5_000_000,
            'Source NRG1 training is not completed.')
    require(spec.get('nrg1_protocol_sha256') == sha(root/'protocol.json'), 'Source run/protocol mismatch.')
    requested = protocol['request']
    clean = {k:v for k,v in spec.items() if k != 'nrg1_protocol_sha256'}
    require(clean == requested['spec'] and settings == requested['settings'],
            'Saved source spec/settings differ from frozen protocol.')
    model = run/'last_model.zip'; model_sha = sha(model)
    registry = read(run/'checkpoints.json')
    require(any(r.get('final') and Path(r['path']).name == 'last_model.zip' and r['sha256'] == model_sha
                for r in registry), 'Original final checkpoint not in its registry.')
    data, sb3 = checkpoint_data(model)
    require(sb3 == '2.7.0', f'This runner targets the recorded source SB3 2.7.0, found {sb3}.')
    require(data.get('gamma') == 1.0 and data.get('gae_lambda') == 0.95, 'Source model settings mismatch.')
    require(data.get('num_timesteps') == status['actual_timesteps'], 'Source model/status steps mismatch.')
    kwargs = data.get('policy_kwargs',{})
    require('CombinedExtractor' in str(kwargs.get('features_extractor_class'))
            and 'Pad2' not in str(kwargs) and 'CNN3' not in str(kwargs), 'Source is not the original CombinedExtractor model.')
    counts = verify_manifest(root/'manifest.csv')
    return dict(root=root, run=run, snapshot=snapshot, protocol=protocol, spec=spec,
                settings=settings, counts=counts, source_model_sha256=model_sha, sb3_version=sb3)


def make_plan(context, device=None, smoke=False):
    spec = deepcopy(context['spec']); spec.pop('nrg1_protocol_sha256', None)
    spec.update(run_id=RUN_ID, configuration_id=RUN_ID.rsplit('_seed', 1)[0],
                stage='cnn3_noclock_external_cap_combined_change',
                reward_case='NRG1_C3NT', remaining_step_observation=False)
    spec['encoder_ablation'] = dict(name='CNN3', source_run=ORIGINAL_RUN,
        specification='Conv3x3/s2/p1: 6->32->64->64; flatten3136->256',
        old_spatial=[50,11,4,2], new_spatial=[50,25,13,7],
        map_feature_dim=256, merged_feature_dim=261,
        teacher=False, warm_start=False, cnn_only_ablation=False)
    spec['reward_definition'] = deepcopy(context['spec']['reward_definition'])
    spec['reward_definition'].update(version=VERSION,
        cost='-((1-beta)+beta*W_pre/W_total)*dt/(ceil(5*N)*time_turn_sec)',
        success_terminal=2.0, failure_terminal=0.0,
        normalization='ceil(5*N)*time_turn: independent of external execution cap',
        external_cap='terminated=False, truncated=True; SB3 adds gamma*V(last_state)',
        metrics='Capped unfinished paths remain evaluation failures; Lw=null',
        prefix_returns='No success/failure separation assertion on truncated prefix returns')
    spec['external_stop'] = dict(cap_steps='ceil(environment.step_budget_factor*N)',
        safety_step_factor=5.0, normalization_step_factor=5.0,
        clock_observed=False, reset_after_cap=True, timeout_penalty=0.0)
    spec['diagnostic_smoke_only'] = bool(smoke)
    settings = deepcopy(context['settings'])
    if device is not None: settings['device'] = device
    if smoke: settings['total_timesteps'] = int(settings['num_envs'] * settings['ppo']['n_steps'])
    require(spec['environment'] == context['spec']['environment'], 'Native physics/B6/cap changed.')
    require(settings['ppo'] == context['settings']['ppo'], 'PPO hyperparameter mutation.')
    evaluation = deepcopy(context['protocol']['request']['evaluation'])
    return spec, settings, evaluation


def transform_trainer(source):
    """Patch the verified NRG1 trainer; retain its durable training/resume logic."""
    tree = ast.parse(source)
    training = next((n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train_sb3'),None)
    require(training is not None, 'Unsupported trainer: train_sb3 absent.')
    edits=0;core_edits=0
    for node in ast.walk(tree):
        if isinstance(node,ast.ImportFrom) and node.module=='nrg1_core':
            node.module='nrg1_c3nt_core';core_edits+=1
        if isinstance(node,ast.ImportFrom) and node.module=='stable_baselines3.common.torch_layers':
            require(len(node.names)==1 and node.names[0].name=='CombinedExtractor',
                    'Unexpected torch_layers import, no guessed replacement.')
            node.module='nrg1_c3nt_features'
            node.names=[ast.alias(name='CNN3CombinedExtractor',asname='CombinedExtractor')]
            edits+=1
    require(edits==1 and core_edits==2,
            f'Expected one extractor and two NRG core imports, got {edits}/{core_edits}.')
    logs=[n for n in ast.walk(training) if isinstance(n,ast.ClassDef) and n.name=='LogCallback']
    require(len(logs)==1, 'Missing/ambiguous LogCallback.')
    start=next((n for n in logs[0].body if isinstance(n,ast.FunctionDef) and n.name=='_on_training_start'),None)
    require(start is not None, 'Missing training-start audit hook.')
    require(not any(isinstance(n,ast.FunctionDef) and n.name=='_on_rollout_end' for n in logs[0].body),
            'Existing rollout-end implementation differs; review before patching.')
    at=0
    while at<len(start.body) and isinstance(start.body[at],(ast.Global,ast.Nonlocal)):at+=1
    start.body[at:at]=ast.parse(
        'cnn3_write_model_audit(self.model, out / "network_observation_audit.json")').body
    logs[0].body += ast.parse("""
def _on_rollout_end(self):
    c3nt_audit_rollout_end(self.locals, self.model)
""").body
    tree.body += ast.parse("""
from nrg1_c3nt_features import write_cnn3_model_audit as cnn3_write_model_audit
from nrg1_c3nt_core import audit_rollout_end as c3nt_audit_rollout_end
""").body
    ast.fix_missing_locations(tree)
    result=ast.unparse(tree)+'\n';compile(result,PAD_TRAINER+'.py','exec')
    return result


def transform_evaluator(source):
    old="require(model.gamma==1.0 and 'remaining_step_ratio' in model.observation_space.spaces,'Wrong final model.')"
    require(source.count(old)==1, 'Frozen NRG1 evaluator signature differs; no unverified patch.')
    result=source.replace(old,
        "from nrg1_c3nt_features import audit_cnn3_model\n    audit_cnn3_model(model)")
    result=result.replace('Relaxed B6 NRG1 —', 'Relaxed B6 CNN3 / NoClock / ExternalCap —')
    compile(result,'c3nt_frozen_evaluation.py','exec')
    return result


def prepare_snapshot(context, destination):
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    hashes = context['protocol']['snapshot_hash']
    for name in hashes:
        source = checked_child(context['snapshot'],name)
        target = destination/name; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
    for name in PACKAGE:
        source = Path(__file__).resolve().parent/name
        require(source.is_file(), f'Package file missing: {source}')
        require(not (destination/name).exists(), 'Original snapshot already has '+name)
        shutil.copy2(source,destination/name)
    source = (destination/(SOURCE_TRAINER+'.py')).read_text(encoding='utf-8-sig')
    revised = transform_trainer(source)
    (destination/(PAD_TRAINER+'.py')).write_text(revised, encoding='utf-8')
    (destination/'C3NT_trainer.diff').write_text(''.join(difflib.unified_diff(
        source.splitlines(True),revised.splitlines(True),fromfile=SOURCE_TRAINER+'.py',
        tofile=PAD_TRAINER+'.py')),encoding='utf-8')
    old_eval=(destination/'run_relaxed_b6_nrg1.py').read_text(encoding='utf-8-sig')
    new_eval=transform_evaluator(old_eval)
    (destination/'c3nt_frozen_evaluation.py').write_text(new_eval, encoding='utf-8')
    (destination/'C3NT_evaluator.diff').write_text(''.join(difflib.unified_diff(
        old_eval.splitlines(True), new_eval.splitlines(True),
        fromfile='original NRG1 evaluator', tofile='C3NT evaluator')), encoding='utf-8')
    return {str(p.relative_to(destination)):sha(p) for p in sorted(destination.rglob('*')) if p.is_file()}


def runtime(snapshot):
    snapshot = Path(snapshot).resolve(); sys.dont_write_bytecode=True
    for name in (PAD_TRAINER,'nrg1_c3nt_core','nrg1_c3nt_features','full_v2_utils','evaluation_ch4'):
        if name in sys.modules:
            require(Path(sys.modules[name].__file__).resolve().parent == snapshot,
                    f'Module {name} came from a different folder; use a fresh process.')
    sys.path.insert(0,str(snapshot)); importlib.invalidate_caches()
    name = '_c3nt_frozen_nrg1_runner'
    imp = importlib.util.spec_from_file_location(name,snapshot/'c3nt_frozen_evaluation.py')
    entry = importlib.util.module_from_spec(imp);sys.modules[name]=entry;imp.loader.exec_module(entry)
    trainer = importlib.import_module(PAD_TRAINER)
    utils = importlib.import_module('full_v2_utils')
    evaluation = importlib.import_module('evaluation_ch4')
    nrg = importlib.import_module('nrg1_c3nt_core')
    features = importlib.import_module('nrg1_c3nt_features')
    for module in (trainer,utils,evaluation,nrg,features):
        require(Path(module.__file__).resolve().parent == snapshot, 'Runtime escaped frozen snapshot.')
    require(nrg.VERSION == VERSION, 'Wrong C3NT reward adapter.')
    # The original evaluator references its module-level RUN_ID. No source file
    # is overwritten; it is redirected only to the separate C3NT output run.
    entry.RUN_ID = RUN_ID
    entry.VERSION = VERSION
    require(hasattr(entry,'evaluate_phase'), 'Frozen evaluator interface not available.')
    return trainer,utils,evaluation,nrg,features,entry


def software_check():
    import importlib.metadata as metadata
    versions = {}
    for name in ('torch','numpy','gymnasium','stable-baselines3','sb3-contrib'):
        try: versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError as exc:
            raise RuntimeError(f'{name} is missing. Run in the original dqn_env.') from exc
    for name in ('stable-baselines3','sb3-contrib'):
        require(versions[name] == '2.7.0', f'{name}={versions[name]} differs from original 2.7.0; stop.')
    return versions


def preflight(rt, spec, settings, records, temp_root):
    """Original server core + real SB3 checks; collect diagnostic rollouts but never optimize."""
    import torch
    import gymnasium as gym
    from gymnasium import spaces
    from dataclasses import replace
    from types import MethodType
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.preprocessing import preprocess_obs
    from stable_baselines3.common.vec_env import DummyVecEnv
    trainer,utils,ev,nrg,features,entry=rt
    torch.set_num_threads(int(settings.get('torch_threads',4)))
    versions=software_check()
    cfg=trainer.ExperimentEnvConfig(**spec['environment'])
    require(cfg.grid_size==50 and cfg.step_budget_factor==5 and cfg.detour_budget_max==6,
            'Unexpected C3NT physics/mask/cap.')
    small=np.zeros((50,50),np.uint8);small[1,1]=1;small[1,2]=5
    distant=np.zeros_like(small);distant[1,1]=1;distant[48,48]=5
    at_cap=np.zeros_like(small);at_cap[1,1]=1;at_cap[1,11]=5
    singleton=np.zeros_like(small);singleton[2,2]=5
    dense=np.zeros_like(small);dense[1:6,1:6]=1
    environment_cases=[]
    cases=[('success',small,(1,1)),('success_at_cap',at_cap,(1,1)),
           ('safety_cap',distant,(1,1)),('singleton',singleton,(2,2)),
           ('budget_walk',dense,(3,3))]
    for name,arr,start in cases:
        core=trainer.make_core(cfg)
        obs,_=core.reset(arr,start=start,heading=1,seed=42,map_name='c3nt_check_'+name)
        require(set(obs)==features.OBS_KEYS,'Clock or missing key in core observation.')
        require(core.max_steps==math.ceil(5*core.target_cells_count),'Stop cap changed.')
        rng=np.random.default_rng(42)
        terminated=truncated=False
        for _ in range(max(1,core.max_steps)):
            action=0 if core.complete_at_reset else 1 if name in {'success','success_at_cap'} else int(rng.choice(np.flatnonzero(core.action_masks())))
            obs,reward,terminated,truncated,info=core.step(action)
            require(set(obs)==features.OBS_KEYS,'Clock leaked after step.')
            require(math.isclose(reward,sum(info['step_reward_'+k] for k in nrg.REWARD_KEYS),abs_tol=1e-9),
                    'Environment reward delivery mismatch.')
            require(not(terminated and truncated),'Both flags true.')
            if terminated or truncated:break
        core.assert_consistent()
        if name in {'success','success_at_cap','singleton'}:
            require(terminated and not truncated and bool(info['is_success']), 'Actual completion mislabelled.')
            require(info['step_reward_terminal']==2.0, 'Missing one-time completion reward.')
        if name=='success_at_cap':require(core.steps==core.max_steps,'Completion-at-cap test not at cap.')
        if name=='safety_cap':
            require(truncated and not terminated and not info['is_success'],'External interruption mislabelled.')
            require(info['step_reward_terminal']==0 and info['reward_sum_terminal']==0,
                    'Safety cap still receives a failure terminal penalty.')
        with_error=False
        try: core.step(0)
        except RuntimeError:with_error=True
        require(with_error,'Core permits movement after collection stopped without reset.')
        environment_cases.append(dict(case=name,steps=int(core.steps),terminated=bool(terminated),
            truncated=bool(truncated),success=int(info['is_success']),
            environment_return=float(core.total_return),terminal_reward=float(info['step_reward_terminal']),
            identity_error=float(info['nrg_return_identity_error'])))

    # A larger TEST-ONLY external cap must not change normalization, observations
    # or per-step reward along an identical prefix. Training cap remains exactly 5N.
    short=trainer.make_core(cfg);long=trainer.make_core(replace(cfg,step_budget_factor=20.0))
    x,_=short.reset(distant,start=(1,1),heading=1,seed=42)
    y,_=long.reset(distant,start=(1,1),heading=1,seed=42)
    require(short._nrg_cap==long._nrg_cap, 'Execution cap changes reward scale.')
    for j in range(short.max_steps):
        for key in x:require(np.array_equal(x[key],y[key]),'Execution cap leaks into observation.')
        x,r,ds,ts,info=short.step(1);y,s,dl,tl,_=long.step(1)
        require(math.isclose(r,s,abs_tol=1e-10),'Execution cap changes prefix reward.')
    require(ts and not ds and not dl and not tl,'Expected external-cap split did not occur.')
    for key in x:require(np.array_equal(x[key],y[key]),'Last-state observation is artificial terminal encoding.')

    training=[r for r in records if canonical_split(r.split)=='train']
    require(training,'No training maps.')
    for rec in training[:3]:
        arr=np.load(rec.path,allow_pickle=False)
        require(utils.array_hash(arr)==rec.sha256,'Training map hash mismatch: '+rec.map_id)
        c=trainer.make_core(cfg);o,_=c.reset(arr,seed=42,map_name=rec.map_id)
        require(set(o)==features.OBS_KEYS,'Original training map interface mismatch.')

    class ProbeEnv(gym.Env):
        metadata={}
        def __init__(self,kind):
            super().__init__();self.kind=kind;self.core=trainer.make_core(cfg)
            self.action_space=spaces.Discrete(4)
            self.observation_space=spaces.Dict({
                'map':spaces.Box(0,255,(6,50,50),dtype=np.uint8),
                'heading':spaces.Discrete(4),
                'detour_budget':spaces.Box(0,1,(1,),dtype=np.float32)})
        def reset(self,*,seed=None,options=None):
            super().reset(seed=seed)
            a=distant if self.kind=='cap' else small
            return self.core.reset(a,start=(1,1),heading=1,seed=42,map_name='c3nt_probe_'+self.kind)
        def step(self,action):return self.core.step(int(action))
        def action_masks(self):return self.core.action_masks()
    vector_env=DummyVecEnv([lambda:ProbeEnv('cap'),lambda:ProbeEnv('success')])
    obs,_=vector_env.envs[0].reset(seed=42)
    p=settings['ppo']
    # Small buffers for this check ONLY; source training_settings are unmodified.
    model=MaskablePPO('MultiInputPolicy',vector_env,device=settings['device'],seed=42,
        learning_rate=p['learning_rate_initial'],n_steps=16,batch_size=16,
        n_epochs=p['n_epochs'],gamma=p['gamma'],gae_lambda=p['gae_lambda'],ent_coef=p['ent_coef'],
        vf_coef=p['vf_coef'],max_grad_norm=p['max_grad_norm'],clip_range=p['clip_range'],
        target_kl=p['target_kl'],verbose=0,
        policy_kwargs={'features_extractor_class':features.CNN3CombinedExtractor,
            'features_extractor_kwargs':{'cnn_output_dim':256},'share_features_extractor':True,
            'net_arch':{'pi':[256,256],'vf':[256,256]},'activation_fn':torch.nn.ReLU})
    report=features.audit_cnn3_model(model)
    report['model_summary']=str(model.policy)
    report['initial_weight_sensitivity']=features.multi_input_sensitivity_audit(model.policy.features_extractor.extractors['map'])
    model.policy.set_training_mode(False)
    hook_values=[]
    def hook(module,inputs):hook_values.append(inputs[0].detach().cpu().clone())
    handle=model.policy.features_extractor.extractors['map'].cnn[0].register_forward_pre_hook(hook)
    with torch.no_grad():
        tensor,_=model.policy.obs_to_tensor(obs)
        processed=preprocess_obs(tensor,model.observation_space,normalize_images=model.policy.normalize_images)
        expected=torch.as_tensor(obs['map'],device=model.device,dtype=torch.float32)[None]/255
        require(torch.equal(processed['map'],expected),'SB3 scale is not exactly /255.')
        require(processed['heading'].shape==(1,4),'Heading not one-hot.')
        mask=vector_env.envs[0].action_masks()[None]
        probs=model.policy.get_distribution(tensor,action_masks=mask).distribution.probs
        feats=model.policy.extract_features(tensor)
        require(feats.shape==(1,261),'Combined feature is not 261.')
        require(bool(torch.all(probs[0][~torch.as_tensor(mask[0],device=model.device)]<1e-6)),
                'Invalid action probability.')
    handle.remove()
    require(hook_values and torch.equal(hook_values[0],expected.cpu()),
            'CNN first layer got wrong scale, including potential double normalization.')
    report['scale_test']=dict(dtype=str(hook_values[0].dtype),min=float(hook_values[0].min()),
                             max=float(hook_values[0].max()),exact_sb3_div255=True,
                             cnn_does_not_renormalize=True)
    temporary=Path(temp_root)/'c3nt_probe_model.zip';model.save(temporary)
    restored=MaskablePPO.load(temporary,device=settings['device'])
    features.audit_cnn3_model(restored)
    with torch.no_grad():
        other=restored.policy.get_distribution(tensor,action_masks=mask).distribution.probs
        require(torch.allclose(probs,other,atol=1e-7,rtol=1e-6),'CNN3 save/load changed policy distribution.')
    require(model.num_timesteps==0 and restored.num_timesteps==0,'Unexpected learning before bootstrap test.')
    nrg.audit_model(model)

    # Run the ACTUAL MaskablePPO collector with a deliberately known value
    # function. A terminal position differs from the reset position, so a wrong
    # bootstrap from next reset can be detected. No train()/optimizer step called.
    original_predict=model.policy.predict_values
    original_forward=model.policy.forward
    def known_value(self,observations):
        m=observations['map'];m=m.unsqueeze(0) if m.ndim==3 else m
        flat=m[:,4].reshape(m.shape[0],-1).argmax(dim=1)
        row=flat//50;col=flat%50
        return (7.0+row.float()*0.03+col.float()*0.07).reshape(-1,1)
    model.policy.predict_values=MethodType(known_value,model.policy)
    def scripted_forward(self, observations, deterministic=False, action_masks=None):
        # Test-only actions force a distinct last-state position at the cap.
        # They are legal under the actual mask; no optimizer uses these samples.
        mask_tensor=torch.as_tensor(action_masks,device=model.device,dtype=torch.bool)
        actions=torch.where(mask_tensor[:,1],torch.ones(mask_tensor.shape[0],device=model.device,dtype=torch.long),
                            mask_tensor.long().argmax(dim=1))
        dist=self.get_distribution(observations,action_masks=action_masks)
        return actions, self.predict_values(observations), dist.log_prob(actions)
    model.policy.forward=MethodType(scripted_forward,model.policy)
    class CollectorAudit(BaseCallback):
        def __init__(self):super().__init__();self.truncations=0;self.completions=0;self.distinct_terminal_vs_reset=0
        def _on_step(self):
            nrg.audit_delivery(self.locals,self.model)
            for j,info in enumerate(self.locals['infos']):
                if info.get('c3nt_truncated'):
                    self.truncations+=1
                    last=info['terminal_observation']
                    reset={k:v[j] for k,v in self.locals['new_obs'].items()}
                    require(set(last)==features.OBS_KEYS,'Clock appears in terminal observation.')
                    r,c=info['c3nt_last_position']
                    require(last['map'][4,r,c]==255 and int(last['heading'])==info['c3nt_last_heading'],
                            'Terminal observation does not describe the actual last movement.')
                    self.distinct_terminal_vs_reset+=int(not np.array_equal(last['map'],reset['map']))
                self.completions+=int(bool(info.get('c3nt_terminated')))
            return True
        def _on_rollout_end(self):nrg.audit_rollout_end(self.locals,self.model)
    weights_before={k:v.detach().clone() for k,v in model.policy.state_dict().items()}
    optimizer_before=deepcopy(model.policy.optimizer.state_dict())
    cb=CollectorAudit()
    _,callback=model._setup_learn(32,callback=cb,reset_num_timesteps=True,progress_bar=False)
    callback.on_training_start({}, {})
    model.collect_rollouts(vector_env,callback,model.rollout_buffer,n_rollout_steps=16)
    callback.on_training_end()
    model.policy.predict_values=original_predict
    model.policy.forward=original_forward
    require(cb.truncations>0 and cb.completions>0 and cb.distinct_terminal_vs_reset>0,
            'Collector test failed to exercise both completion and distinct timeout states.')
    require(model._n_updates==0 and all(torch.equal(v,model.policy.state_dict()[k]) for k,v in weights_before.items()),
            'Check unexpectedly changed network weights.')
    require(not model.policy.optimizer.state_dict()['state'] and not optimizer_before['state'],
            'Optimizer state was unexpectedly updated.')
    report.update(software=versions,environment_cases=environment_cases,
        cap_independent_normalizer_and_observation='PASS',
        sb3_save_load='PASS',diagnostic_collector_transitions=int(model.num_timesteps),
        original_training_settings_unmodified=True,optimizer_updates_executed=0,
        bootstrap_test=dict(**model._c3nt_last_rollout_audit,
                           completed_transitions=cb.completions,
                           timeout_state_differs_from_reset=cb.distinct_terminal_vs_reset,
                           known_value_function_used_for_collector_test=True,
                           scripted_legal_actions_in_collector_test=True),
        return_note='Logs contain raw environment returns; critic continuation is only in rollout targets. '
                    'Unfinished capped paths remain failures, not positive-return successes.')
    vector_env.close();del model,restored
    if torch.cuda.is_available():torch.cuda.empty_cache()
    print('[CHECK PASS] CNN3 50->25->13->7, 3136->256, merged261; no clock; '
          '2500/2500 connected; real SB3 truncation bootstrap from LAST state; no optimizer update.',flush=True)
    return report


def float_value(x):
    try: value=float(x)
    except (ValueError,TypeError): return None
    return value if math.isfinite(value) else None


def paired_validation(source_root,out):
    old=source_root/'validation_final'/'per_map.csv';new=out/'validation_final'/'per_map.csv'
    if not old.is_file() or not new.is_file(): return
    a=read_csv(old);b=read_csv(new)
    ma={r['map_id']:r for r in a};mb={r['map_id']:r for r in b}
    require(len(ma)==len(a)==len(b)==len(mb)==180 and set(ma)==set(mb), 'Validation map identities differ.')
    paired=[]
    for mid in sorted(ma):
        x,y=ma[mid],mb[mid]
        for key in ('map_sha256','initial_pair_id','start_r','start_c','start_heading','max_steps'):
            require(key in x and key in y and x[key]==y[key], 'Paired validation mismatch: '+key+' '+mid)
        sx=bool(int(float(x['is_success'])));sy=bool(int(float(y['is_success'])))
        tx,ty=float_value(x['mission_time_sec']),float_value(y['mission_time_sec'])
        lx,ly=float_value(x[LW]),float_value(y[LW])
        paired.append(dict(map_id=mid,tier=y['tier'],pattern=y['pattern'],original_success=int(sx),
            c3nt_success=int(sy),original_T=tx,c3nt_T=ty,original_Lw=lx,c3nt_Lw=ly,
            delta_T_on_both_success=ty-tx if sx and sy else None,
            delta_Lw_on_both_success=ly-lx if sx and sy else None))
    write_csv(out/'paired_validation_vs_original.csv',paired)
    common=[r for r in paired if r['original_success'] and r['c3nt_success']]
    avg=lambda k:float(np.mean([r[k] for r in common])) if common else None
    write(out/'paired_validation_summary.json',dict(n_maps=180,
        original_success=sum(r['original_success'] for r in paired),c3nt_success=sum(r['c3nt_success'] for r in paired),
        lost=sum(r['original_success'] and not r['c3nt_success'] for r in paired),
        gained=sum(not r['original_success'] and r['c3nt_success'] for r in paired),
        paired_success_n=len(common),delta_T_mean=avg('delta_T_on_both_success'),
        delta_Lw_mean=avg('delta_Lw_on_both_success'),
        note='Combined CNN/clock/truncation change minus original NRG1; common success only, not a CNN-only effect or WLG/BCD goal PASS.'))


def make_review(out):
    files=[out/name for name in ('protocol.json','config.json','preflight.json','cases_plan.json',
            'paired_validation_vs_original.csv','paired_validation_summary.json') if (out/name).is_file()]
    run=out/'runs'/RUN_ID
    for name in ('run_spec.json','training_settings.json','status.json','training_episodes.csv',
                 'training_updates.csv','checkpoints.json','network_observation_audit.json','model_architecture.json'):
        if (run/name).is_file(): files.append(run/name)
    for phase in ('validation_final','full_test'):
        for name in ('per_map.csv','summary.json','group_summary.csv','evaluation_complete.json','evaluation_identity.json'):
            if (out/phase/name).is_file(): files.append(out/phase/name)
    files += [out/'source_snapshot'/name for name in PACKAGE if (out/'source_snapshot'/name).is_file()]
    target=out/'c3nt_review.zip';temp=target.with_name('c3nt_review.writing.zip')
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files: archive.write(path,path.relative_to(out))
    os.replace(temp,target)
    print('[REVIEW] '+str(target),flush=True)


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('check','train','validation','test','all'),default='check')
    p.add_argument('--root',type=Path,default=Path('chapter4_relaxedB6_NRG1_seed42'))
    p.add_argument('--run-dir',type=Path,help='Actual original NRG1 run folder, only if moved.')
    p.add_argument('--out',type=Path,default=Path('chapter4_relaxedB6_NRG1_C3NT_seed42'))
    p.add_argument('--check-out', type=Path, default=Path('nrg1_c3nt_check_report'),
                   help='Separate folder for shape/scale/bootstrap report; no training artifacts.')
    p.add_argument('--device',default=None,help='Training GPU, e.g. cuda:0; default original saved device.')
    p.add_argument('--eval-device',default='cpu',help='Validation/test device; keep fixed for cache reuse.')
    p.add_argument('--smoke',action='store_true',help='ONE original PPO rollout for integration only; separate *smoke* output required.')
    return p.parse_args()


def main():
    args=parse_args();root=path_from_script(args.root);out=path_from_script(args.out)
    require(root!=out and root not in out.parents and out not in root.parents,
            'Output must be separate from (not inside/above) original NRG1.')
    require(out!=Path(__file__).resolve().parent,'Output cannot be the suite directory.')
    context=source_context(root,path_from_script(args.run_dir) if args.run_dir else None)
    spec,settings,ecfg=make_plan(context,args.device,args.smoke)
    if args.smoke:
        require('smoke' in out.name.lower() and args.phase in {'check','train','all'},
                '--smoke requires a separate output containing smoke and phase check/train/all.')
    print('[SOURCE NRG1] '+str(context['run']))
    print('[SOURCE SNAPSHOT] '+str(context['snapshot']))
    print('[OUTPUT] '+str(out))
    print('[COMBINED CHANGE] CNN3/s2/p1 x3, projection3136->256; clock REMOVED; safety cap TRUNCATES')
    print('[FIXED] B6 | cap5N | beta=.5 | gamma1 | lambda.95 | train maps/PPO; moving costs unchanged')
    print('[FRESH WEIGHTS] no teacher/BC/warm-start; +2 completion retained, cap -2 removed')
    print(f'[TRAIN] target={settings["total_timesteps"]:,} | device={settings["device"]}')
    print('[SPLITS] '+str(context['counts']),flush=True)
    if args.phase=='check':
        with tempfile.TemporaryDirectory(prefix='nrg1_c3nt_check_') as temp:
            snapshot=Path(temp)/'source_snapshot';prepare_snapshot(context,snapshot)
            rt=runtime(snapshot)
            records=rt[1].load_manifest(root/'manifest.csv',50)
            report_dir=path_from_script(args.check_out)
            require(report_dir not in (root,out,context['snapshot']) and root not in report_dir.parents
                    and out not in report_dir.parents and report_dir not in root.parents
                    and report_dir not in out.parents, 'Check report must be separate from experiment inputs/outputs.')
            require(report_dir!=Path(__file__).resolve().parent, 'Check report cannot overwrite suite.')
            report=preflight(rt,spec,settings,records,temp)
            report_dir.mkdir(parents=True,exist_ok=True)
            with output_lock(report_dir/'.report.lock'):
                write(report_dir/'checks.json',report)
                (report_dir/'model_summary.txt').write_text(report['model_summary'],encoding='utf-8')
                write(report_dir/'requested_changes.json',dict(spec=spec,settings=settings))
                for item in ('C3NT_trainer.diff','C3NT_evaluator.diff'):
                    shutil.copy2(snapshot/item,report_dir/item)
            print('[CHECK REPORT] '+str(report_dir))
        print('[CHECK ONLY] No optimizer update and no training experiment created.')
        return
    out.mkdir(parents=True,exist_ok=True)
    with output_lock(out/'.runner.lock'):
        request=dict(version=VERSION,source_root=str(root),source_run=str(context['run']),
            source_protocol_sha256=sha(root/'protocol.json'),source_manifest_sha256=sha(root/'manifest.csv'),
            source_model_sha256=context['source_model_sha256'],source_snapshot_hash=context['protocol']['snapshot_hash'],
            package_hash={n:sha(Path(__file__).parent/n) for n in PACKAGE},
            spec=spec,settings=settings,evaluation=ecfg,eval_device=args.eval_device)
        pp=out/'protocol.json';snapshot=out/'source_snapshot'
        if pp.is_file():
            protocol=read(pp)
            require(protocol['request']==safe(request),'Frozen C3NT protocol changed. Use same command/code or a NEW --out.')
            for name,h in protocol['snapshot_hash'].items():
                require(sha(checked_child(snapshot,name))==h,'C3NT source snapshot changed: '+name)
            require(sha(out/'manifest.csv')==protocol['manifest_sha256'],'C3NT manifest changed.')
        else:
            require(not any(p.name!='.runner.lock' for p in out.iterdir()),'Nonempty unrecognized output; use a new --out.')
            hashes=prepare_snapshot(context,snapshot)
            shutil.copy2(root/'manifest.csv',out/'manifest.csv')
            protocol=dict(request=request,snapshot_hash=hashes,manifest_sha256=sha(out/'manifest.csv'),counts=context['counts'])
            write(pp,protocol)
            write(out/'config.json',dict(environment=spec['environment'],training=settings,evaluation=ecfg,
                reward_definition=spec['reward_definition'],encoder_ablation=spec['encoder_ablation']))
        spec=deepcopy(spec);spec['nrg1_protocol_sha256']=sha(pp)
        rt=runtime(snapshot);trainer,utils,ev,nrg,features,entry=rt
        versions=software_check();write(out/'software_at_run.json',versions)
        records=utils.load_manifest(out/'manifest.csv',50)
        if args.phase in {'train','all'}:
            with tempfile.TemporaryDirectory(prefix='c3nt_preflight_') as temp:
                write(out/'preflight.json',preflight(rt,spec,settings,records,temp))
            training=[r for r in records if canonical_split(r.split)=='train']
            result=trainer.train_run(spec,settings,training,out/'runs'/RUN_ID)
            require(result.get('state')=='completed','Training incomplete; evaluation not started.')
            print('[TRAIN COMPLETE] '+str(out/'runs'/RUN_ID/'last_model.zip'),flush=True)
        if args.smoke:
            make_review(out);print('[SMOKE COMPLETE] No validation/test; not a 5M experiment.');return
        final=out/'runs'/RUN_ID/'last_model.zip'
        require(final.is_file(),'C3NT final model missing. Run --phase train first.')
        # Validate the saved custom class and image geometry before reusing the
        # cloned evaluator (clock guard replaced). Only trusted checkpoints are loaded.
        from sb3_contrib import MaskablePPO
        import torch
        model=MaskablePPO.load(final,device=args.eval_device)
        features.audit_cnn3_model(model);del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        if args.phase in {'train','all','validation'}:
            entry.evaluate_phase(out,trainer,utils,ev,spec,ecfg,records,'validation',args.eval_device)
            paired_validation(root,out)
        if args.phase in {'all','test'}:
            entry.evaluate_phase(out,trainer,utils,ev,spec,ecfg,records,'test',args.eval_device)
        make_review(out)


if __name__=='__main__':
    try: main()
    except KeyboardInterrupt:
        print('\n[STOP] Rerun the SAME command; PPO uses the original committed optimizer-boundary recovery.')
        raise SystemExit(130)
    except Exception as exc:
        print(f'[STOP] {type(exc).__name__}: {exc}',file=sys.stderr)
        raise
