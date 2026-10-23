"""Authenticated native COST11 runtime and B-only experiment orchestration.

This module does not reimplement PPO, the coverage reward, the CNN, or recovery.
Those components are imported from the user's frozen, hash-verified snapshots.
"""
from __future__ import annotations
import ast, csv, hashlib, importlib, importlib.metadata, json, math, os, shutil
import subprocess, sys, time
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import b124_native_support as U

VERSION = 'cost11_B124_beta05_native40M_cp5M_val1M_v5_completeaudit_20261002'
PACKAGE = Path(__file__).resolve().parent
STAGES = (5_000_000, 20_000_000, 40_000_000)
INTERVAL = 1_000_000  # validation; temporary weights removed after successful evaluation
CHECKPOINT_INTERVAL = 5_000_000  # permanent model-selection checkpoints
BLOCK = 8192
KEYS = ('B1', 'B2', 'B4')
T, LW = U.T, U.LW


def reference():
    r = U.read(PACKAGE / 'b124_reference.json')
    for name, h in r['bundle_files'].items():
        U.require(U.sha(PACKAGE / name) == h, 'Package reference changed: ' + name)
    return r


def relocate(path, root):
    """Use an existing exact path, otherwise relocate the documented suite root."""
    p = Path(path)
    if p.is_file():
        return p.resolve()
    token = '/uav_chapter4_suite_v2/'
    if token in str(p):
        q = root / str(p).split(token, 1)[1]
        if q.is_file():
            return q.resolve()
    raise FileNotFoundError(f'Required frozen source is missing: {path}\nSuite root: {root}')


def original_5m(root, ref):
    runid = ref['strict']['spec']['continuation']['source_run_id']
    base = root / 'chapter4_C3NT_COST11_STRICT_BETA_0_05_1_seed42' / 'beta0p5' / 'runs'
    candidates = [base / runid]
    candidates.extend(p.parent for p in base.glob('*/run_spec.json'))
    found = {}
    for run in candidates:
        if not (run / 'run_spec.json').is_file():
            continue
        sp, st = U.read(run / 'run_spec.json'), U.read(run / 'training_settings.json')
        if sp.get('algorithm') != 'ppo' or sp.get('beta') != .5 or sp.get('seed') != 42:
            continue
        U.validate_spec(sp, st, .5)
        status = U.read(run / 'status.json')
        if not (status.get('state') == 'completed' and st['total_timesteps'] == 5_000_000):
            continue
        U.require(st['ppo'] == ref['initial_5m_settings_evidence']['ppo'],
                  'Original beta=.5 5M PPO settings differ from the TEST8 first-stage reference')
        U.require(sp['environment'] == ref['strict']['spec']['environment'],
                  'Original Strict 5M and 40M physical/reward configuration differ')
        found[str(run.resolve())] = (sp, st)
    U.require(len(found) == 1,
              'Need one original Strict beta=.5 5M run (to verify its LR schedule). Searched: ' + str(base))
    name, (sp, st) = next(iter(found.items()))
    protected = {str(Path(name) / n): U.sha(Path(name) / n)
                 for n in ('run_spec.json', 'training_settings.json', 'status.json')}
    return sp, st, protected


def make_plan(root, out, gpus):
    root, out = root.resolve(), out.resolve()
    ref = reference()
    code_hashes = {p.name: U.sha(p) for p in PACKAGE.glob('*.py')}
    code_hashes['b124_reference.json'] = U.sha(PACKAGE / 'b124_reference.json')
    if (out / 'plan.json').exists():
        plan = U.read(out / 'plan.json')
        U.require(plan['version'] == VERSION and plan['package_hashes'] == code_hashes,
                  'Existing output belongs to different code/reference. Use a new --out.')
        U.require(plan['root'] == str(root), 'Resume root changed')
        return plan
    U.require(len(gpus) == 3 and len(set(gpus)) == 3, 'Supply three distinct GPUs for B=1,2,4')
    sp, st, protected = original_5m(root, ref)
    source_status = U.read(next(path for path in protected if Path(path).name == 'status.json'))
    for key, value in ref['strict']['spec']['environment'].items():
        if key != 'mask_mode':
            U.require(ref['relaxed']['spec']['environment'][key] == value, 'Strict/Relaxed environment mismatch: ' + key)
    rd0 = {k:v for k,v in ref['strict']['spec']['reward_definition'].items() if k != 'version'}
    rd6 = {k:v for k,v in ref['relaxed']['spec']['reward_definition'].items() if k != 'version'}
    U.require(rd0 == rd6, 'Strict/Relaxed replacement reward definitions differ')
    out.mkdir(parents=True, exist_ok=True)
    # Copy only the known, authenticated module graph. No generic engine fallback.
    runtime = {}
    for family in ('strict', 'relaxed'):
        snap = out / '_runtime' / family
        snap.mkdir(parents=True, exist_ok=True)
        hashes = {}
        for name, info in ref[family]['sources'].items():
            src = relocate(info['path'], root)
            U.require(U.sha(src) == info['sha256'], 'Original runtime changed: ' + str(src))
            dst = snap / name
            if not dst.exists():
                shutil.copyfile(src, dst)
            U.require(U.sha(dst) == info['sha256'], 'Copied runtime changed: ' + str(dst))
            hashes[name] = info['sha256']
        U.require(hashes[U.TRAINER + '.py'] == U.TRAINER_SHA, 'Unsupported native trainer revision')
        runtime[family] = dict(path=str(snap), hashes=hashes,
                               version=ref[family]['native_version'])
    manifest_src = relocate(ref['manifest'], root)
    U.require(U.sha(manifest_src) == ref['manifest_sha256'], 'TEST8 manifest has changed')
    mm = U.rows(manifest_src)
    expected = {('synthetic', 'train'): 900, ('synthetic', 'validation'): 180,
                ('synthetic', 'test'): 720, ('osm', 'test'): 500}
    U.require(Counter((r['source'], r['split']) for r in mm) == expected, 'Wrong data partitions')
    for field in ('map_id', 'sha256'):
        U.require(len({r[field] for r in mm}) == len(mm), 'Duplicate/cross-split data: ' + field)
    for row in mm:
        p = Path(row['path'])
        if not p.is_file():
            p = root.parent / row['map_id']
        U.require(p.is_file(), 'Map not available: ' + row['path'])
        row['path'] = str(p.resolve())
    U.csv_write(out / 'manifest.csv', mm)
    protected[str(manifest_src)] = U.sha(manifest_src)
    for name in ('reference_test_baselines.csv', 'reference_validation_baselines.csv'):
        shutil.copyfile(PACKAGE / name, out / name)
    plan = dict(version=VERSION, root=str(root), out=str(out), gpus=gpus,
                Bs=[1, 2, 4], beta=.5, seed=42, stage_targets=list(STAGES),
                target=40_000_000, checkpoint_interval=CHECKPOINT_INTERVAL, validation_interval=INTERVAL, optimizer_block=BLOCK,
                initial_seed=20260907, evaluation_seed=42, runtime=runtime,
                template_spec=sp, template_settings=st, template_training_identity=source_status['identity'],
                software=ref['strict']['software'],
                manifest=str(out / 'manifest.csv'), manifest_sha256=U.sha(out / 'manifest.csv'),
                protected=protected, package_hashes=code_hashes,
                baseline_hashes={n: U.sha(out / n) for n in
                    ('reference_test_baselines.csv', 'reference_validation_baselines.csv')},
                primary_test_checkpoint='fixed first completed PPO update at nominal 40M; not selected on Test',
                raw_only=True, connection_refinement=False, test_scope='synthetic720 + OSM500; validation excluded',
                initial_training='fresh independent weights for each B; same seed and network; no Strict warm start',
                segment_boundaries='0->5M, 5M->20M, 20M->40M; policy+Adam continued at 5/20M; sampler positions preserved; episodes/RNG restarted at those two boundaries',
                existing_test_exposure=ref['test_access_note'])
    U.write(out / 'plan.json', plan)
    return plan


def verify_plan(plan):
    for name, h in plan['package_hashes'].items():
        U.require(U.sha(PACKAGE / name) == h, 'Experiment code changed: ' + name)
    for p, h in plan['protected'].items():
        U.require(U.sha(p) == h, 'Original reference changed: ' + p)
    U.require(U.sha(plan['manifest']) == plan['manifest_sha256'], 'Frozen manifest changed')
    for n, h in plan['baseline_hashes'].items():
        U.require(U.sha(Path(plan['out']) / n) == h, 'Frozen numerical baseline changed: ' + n)


def segment_paths(plan, B, stage):
    base = Path(plan['out']) / f'B{B}' / ('fresh_0_5M', 'continue_5_20M', 'continue_20_40M')[stage]
    run = base / 'runs' / f'ppo_COST11_FULL_B{B}_beta05_seed42_to{STAGES[stage] // INTERVAL}M'
    return base, run


def final_checkpoint(run):
    status = U.read(run / 'status.json')
    U.require(status.get('state') == 'completed', 'Preceding segment is not complete: ' + str(run))
    cps = [r for r in U.read(run / 'checkpoints.json') if r.get('final')]
    U.require(len(cps) == 1, 'Need one registered final checkpoint')
    cp = deepcopy(cps[0]); cp['path'] = str(run / 'last_model.zip')
    U.require(U.sha(cp['path']) == cp['sha256'], 'Final checkpoint changed')
    U.require(int(cp['num_timesteps']) == int(status['actual_timesteps']), 'Final counters mismatch')
    return cp, status


def segment_spec(plan, B, stage):
    base, run = segment_paths(plan, B, stage)
    sp, st = deepcopy(plan['template_spec']), deepcopy(plan['template_settings'])
    for key in ('continuation', 'continuation_protocol_sha256', 'nrg1_protocol_sha256',
                'strict_beta_sweep', 'source_protocol_sha256', 'strict50m'):
        sp.pop(key, None); st.pop(key, None)
    # This is the only environment/observation semantic change relative to Strict.
    sp['environment']['mask_mode'] = 'relaxed'
    sp['environment']['detour_budget_max'] = B
    sp['detour_budget_max'] = B
    sp['mask'] = f'relaxed_b{B}'
    sp['reward_definition']['version'] = plan['runtime']['relaxed']['version']
    sp.update(run_id=run.name, configuration_id=run.name, stage='B_budget_controlled_training',
              fresh_weights=stage == 0, source_weights_used=stage > 0)
    for key in plan['template_spec']['environment']:
        if key not in ('mask_mode', 'detour_budget_max'):
            U.require(sp['environment'][key] == plan['template_spec']['environment'][key], 'Unintended env change: ' + key)
    st.update(total_timesteps=STAGES[stage], checkpoint_every_timesteps=CHECKPOINT_INTERVAL,
              device='cuda:0', retain_completed_resume=True)
    entry = dict(B=B, beta=.5, stage_index=stage, target=STAGES[stage], initial_source='fresh')
    if stage:
        _, previous = segment_paths(plan, B, stage - 1)
        cp, status = final_checkpoint(previous)
        entry.update(source_model=cp['path'], source_sha256=cp['sha256'],
                     source_step=int(cp['num_timesteps']), source_run=str(previous),
                     source_episodes=int(status['episodes']), source_epoch_counter=int(status['ppo_epoch_counter']),
                     initial_source='same_B_previous_segment_policy_and_Adam',
                     boundary_recovery='sampler advanced, episodes/RNG restarted; subsequent recovery is full-state')
        st['ppo']['learning_rate_initial'] = 5e-5
        st['ppo']['learning_rate_floor'] = 5e-5
    sp['b124_protocol'] = deepcopy(entry); st['b124_segment'] = entry
    return sp, st


def runtime(plan, B, family='relaxed', train=False):
    import numpy as np
    import torch
    snap = Path(plan['runtime'][family]['path']).resolve()
    hashes = plan['runtime'][family]['hashes']
    for name, h in hashes.items():
        U.require(U.sha(snap / name) == h, 'Frozen runtime changed: ' + name)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(snap)); importlib.invalidate_caches()
    modules = []
    for name in (U.TRAINER, 'full_v2_utils', U.CORE, U.FEATURES, U.LOGGER, 'resume_ch4', 'core_ch4'):
        mod = importlib.import_module(name)
        U.require(Path(mod.__file__).resolve().parent == snap, 'Wrong imported engine: ' + name)
        modules.append(mod)
    trainer, utils, core, features, logger, recovery, initial = modules
    for mod in list(sys.modules.values()):
        f = getattr(mod, '__file__', None)
        if f and Path(f).suffix == '.py' and Path(f).resolve().parent == snap:
            U.require(Path(f).name in hashes and U.sha(f) == hashes[Path(f).name], 'Unauthenticated engine dependency: ' + f)
    for name in ('torch', 'numpy', 'gymnasium', 'stable-baselines3', 'sb3-contrib'):
        actual, expected = importlib.metadata.version(name), plan['software'].get(name)
        U.require(actual == expected, f'Use original environment: {name} expected {expected}, got {actual}. Do not upgrade automatically.')
    U.require(core.VERSION == plan['runtime'][family]['version'] and core.COST_MULTIPLIER == 11.0
              and core.NORMALIZATION_STEP_FACTOR == 5.0 and logger.C3NT_VERSION == core.VERSION,
              'Wrong COST11 reward/normalization version')
    if family == 'relaxed':
        # The supplied native bounded core already supports B>=0. Parameterize
        # the COMPLETE B6-only adapter audits in reset/step and its histogram;
        # leave the frozen physics, masks, reward formula and snapshots intact.
        from b124_core_guard import install_budget_guard
        guard_record = install_budget_guard(core, B, hashes[U.CORE + '.py'])
        U.freeze(Path(plan['out']) / 'audit' / f'core_guard_relaxed_B{B}.json', guard_record)
        print(f'[CORE GUARD PATCH] relaxed B={B}; helper={guard_record["guard_helper"]}; complete reset/step budget audits 6 -> {B}; histogram={B+1}; source files unchanged', flush=True)
    torch.set_num_threads(int(plan['template_settings'].get('torch_threads', 4)))
    if train:
        U.require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Assigned GPU unavailable')
    logger.install(trainer, utils)
    env = deepcopy(plan['template_spec']['environment'])
    if family == 'relaxed':
        env.update(mask_mode='relaxed', detour_budget_max=B)
    cfg = trainer.ExperimentEnvConfig(**env)
    check_core = trainer.make_core(cfg)
    U.require(isinstance(check_core, core.C3NoClockMixin), 'Native reward replacement not installed')
    U.require(check_core.config == cfg, 'Native constructor silently changed requested B/configuration')
    rr = utils.load_manifest(plan['manifest'], 50, verify_files=False)
    tr = [r for r in rr if r.source == 'synthetic' and r.split == 'train']
    va = sorted([r for r in rr if r.source == 'synthetic' and r.split == 'validation'],
                key=lambda r: (r.source, str(r.tier), r.pattern, r.map_id))
    te = sorted([r for r in rr if r.split == 'test'],
                key=lambda r: (0 if r.source == 'synthetic' else 1, str(r.tier), r.pattern, r.map_id))
    U.require((len(tr), len(va), len(te)) == (900, 180, 1220), 'Changed data partition')
    U.require(trainer.run_identity(plan['template_spec'], plan['template_settings'], tr) ==
              plan['template_training_identity'], 'Original training manifest order/spec/settings no longer match')
    return SimpleNamespace(trainer=trainer, utils=utils, core=core, features=features,
                           recovery=recovery, initial=initial, cfg=cfg, train=tr, val=va, tests=te)


def normalized_budget(core, obs):
    import numpy as np
    B = int(core.config.detour_budget_max) if core.config.mask_mode == 'relaxed' else 0
    expected = float(core.detour_budget) / B if B else 0.
    value = float(np.asarray(obs['detour_budget']).reshape(-1)[0])
    U.require(abs(value - expected) < 1e-6, f'Budget observation is not remaining/B: {value} vs {expected}')
    U.require(set(obs) == {'map', 'heading', 'detour_budget'}, 'Unexpected clock or other observation')
    U.require(np.asarray(obs['map']).shape == (6, 50, 50), 'Wrong spatial observation schema')


def audit_step(core, obs, action):
    """Use the native transition, checking B semantics without modifying it."""
    import numpy as np
    normalized_budget(core, obs)
    mask = np.asarray(core.action_masks(), dtype=bool)
    strict, _ = core.strict_mask_details_no_cache()
    before = int(core.detour_budget)
    U.require(mask.shape == (4,) and mask.any() and mask[action], 'Action violates native mask')
    if core.config.mask_mode == 'relaxed':
        expected_mask = core.feasible_mask() if before > 0 else strict
        U.require(np.array_equal(mask, expected_mask), 'Budget-open/exhausted action mask mismatch')
    next_obs, reward, terminated, truncated, info = core.step(action)
    if core.config.mask_mode == 'relaxed':
        expected = (int(core.config.detour_budget_max) if bool(info.get('is_new_active_cell', 0))
                    else max(0, before - 1) if not strict[action] else before)
        U.require(int(core.detour_budget) == expected, 'Budget decrement/reset rule mismatch')
    normalized_budget(core, next_obs)
    U.require(info.get('nrg_reward_replaced') and math.isfinite(float(reward)), 'Native COST11 reward missing')
    U.require(not (terminated and truncated), 'Invalid termination flags')
    return next_obs, reward, terminated, truncated, info


def trace_preflight(plan, B, family):
    import numpy as np
    rt = runtime(plan, B, family)
    traces, events = [], dict(detour=0, exhausted=0, replenished=0, budget_zero_actions=0)
    # Deliberately nontrivial, bounded 50x50 map. No optimizer or test-map access.
    a = np.zeros((50, 50), dtype=np.uint8)
    a[20, 20] = 1; a[20, 21] = 5; a[20, 24] = 3
    a[23, 20] = 2; a[24, 24] = 4; a[21, 23] = 1
    c = rt.trainer.make_core(rt.cfg)
    obs, _ = c.reset(a, start=(20, 20), heading=0, seed=42, map_name='B_semantic_preflight')
    U.require(c.config == rt.cfg, 'Native reset silently changed the requested configuration')
    expected_initial_budget = B if family == 'relaxed' else 0
    U.require(int(c.detour_budget) == expected_initial_budget, 'Initial budget differs from requested B')
    normalized_budget(c, obs)
    i = 0
    while not c._done:
        if c.complete_at_reset:
            c.step(0); break
        mask = np.asarray(c.action_masks(), dtype=bool)
        strict, _ = c.strict_mask_details_no_cache()
        # Force real detours while budget is open, otherwise use native Strict.
        det = np.flatnonzero(mask & ~np.asarray(strict, dtype=bool))
        choices = det if len(det) and B else np.flatnonzero(mask)
        action = int(choices[i % len(choices)])
        before = int(c.detour_budget)
        normalized_budget(c, obs)
        obs_sha = U.digest({k: v.tolist() if hasattr(v, 'tolist') else v for k, v in obs.items()})
        obs, reward, terminated, truncated, info = audit_step(c, obs, action)
        events['detour'] += int(not strict[action])
        events['exhausted'] += int(before > 0 and int(c.detour_budget) == 0)
        events['replenished'] += int(bool(info.get('is_new_active_cell', 0)))
        events['budget_zero_actions'] += int(before == 0)
        traces.append(dict(mask=mask.tolist(), action=action, reward=round(float(reward), 10),
                           obs_sha=obs_sha, pos=list(c.drone_pos), heading=int(c.heading) if hasattr(c, 'heading') else int(obs['heading']),
                           covered=int(c.covered_cells_count), terminated=bool(terminated), truncated=bool(truncated)))
        i += 1
        U.require(i <= int(c.max_steps) + 1, 'Safety cap did not terminate preflight')
    c.assert_consistent()
    if B:
        U.require(all(events[k] > 0 for k in events), 'Preflight did not exercise all B transitions: ' + str(events))
    result = dict(B=B, family=family, trace=traces, trace_sha256=U.digest(traces), events=events,
                  final_T=U.number(c.info(final=True).get(T)), optimizer_updates=0)
    U.write(Path(plan['out']) / 'audit' / f'preflight_{family}_B{B}.json', result)
    print(f'[PREFLIGHT PASS] {family} B={B}: {events}; optimizer updates=0', flush=True)
    return result


def extract_train(source, continued):
    tree = ast.parse(source)
    fs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train_sb3']
    U.require(len(fs) == 1, 'Native train_sb3 definition missing')
    f = fs[0]
    text = '\n'.join(source.splitlines()[f.lineno - 1:f.end_lineno]) + '\n'
    changes = [
        ('            stop.raise_if_requested(self.model.num_timesteps)\n',
         '            _b124_start(self.model, spec, settings, out, checkpoints, save_resume)\n            stop.raise_if_requested(self.model.num_timesteps)\n'),
        ('            super().train()\n', '            _b124_assert(self, spec, settings)\n            super().train()\n'),
        ('            stop.raise_if_requested(self.num_timesteps)\n',
         '            _b124_boundary(self, spec, settings, out, checkpoints, save_resume)\n            stop.raise_if_requested(self.num_timesteps)\n')]
    if continued:
        changes += [
            ("checkpoints = deepcopy(state['checkpoints']) if state else []", "checkpoints = deepcopy(state['checkpoints']) if state else _b124_baseline(settings)"),
            ("episode_count = state['episode_count'] if state else 0", "episode_count = state['episode_count'] if state else settings['b124_segment']['source_episodes']"),
            ("next_cp = state['next_cp'] if state else cp_interval", "next_cp = state['next_cp'] if state else (settings['b124_segment']['source_step']//cp_interval+1)*cp_interval"),
            ("next_resume = state['next_resume'] if state else resume_interval", "next_resume = state['next_resume'] if state else (settings['b124_segment']['source_step']//resume_interval+1)*resume_interval"),
            ('                model = SaveAfterUpdate(**common)', '                model = _b124_load(SaveAfterUpdate, env, spec, settings, out, records)'),
            ('reset_num_timesteps=not bool(state)', 'reset_num_timesteps=False')]
    else:
        changes += [('                model = SaveAfterUpdate(**common)',
                     '                model = _b124_fresh(SaveAfterUpdate, common, spec, settings, out)')]
    for before, after in changes:
        U.require(text.count(before) == 1, 'Native patch no longer matches exactly: ' + before)
        text = text.replace(before, after, 1)
    compile(text, 'B124_native_train_patch.py', 'exec')
    return text


def run_worker(plan, mode, B, stage=0, checkpoint=None, scope='validation', family='relaxed', gpu=None):
    env = os.environ.copy(); env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['CUDA_VISIBLE_DEVICES'] = str(gpu) if gpu is not None else ''
    env['MPLBACKEND'] = 'Agg'
    cmd = [sys.executable, str(PACKAGE / 'run_cost11_B124_beta05_40M.py'), '--_worker', mode,
           '--_plan', str(Path(plan['out']) / 'plan.json'), '--B', str(B), '--stage', str(stage),
           '--scope', scope, '--family', family]
    if checkpoint:
        cmd += ['--checkpoint', str(checkpoint)]
    p = subprocess.run(cmd, cwd=plan['root'], env=env)
    U.require(p.returncode == 0, f'{mode} failed (B={B}, segment={stage}); existing checkpoints/cache were retained')


def milestones(plan, B, stage):
    _, run = segment_paths(plan, B, stage)
    if not (run / 'checkpoints.json').exists():
        return []
    found = {}
    for item in U.read(run / 'checkpoints.json'):
        if item.get('source_baseline'):
            continue
        step = int(item['num_timesteps']); m = step // INTERVAL
        if not (1 <= m <= 40 and m % 5 == 0 and 0 <= step - m * INTERVAL < BLOCK):
            continue
        if m in found:
            continue
        cp = deepcopy(item)
        cp['path'] = str(run / ('last_model.zip' if item.get('final') else 'checkpoints/' + Path(item['path']).name))
        cp.update(nominal_m=m, stage_index=stage)
        U.require(Path(cp['path']).is_file(), 'Missing registered checkpoint: ' + cp['path'])
        found[m] = cp
    return list(found.values())


def eval_base(plan, B, cp, scope):
    return Path(plan['out']) / scope / f'B{B}' / f'step_{cp["num_timesteps"]}'


def eval_done(plan, B, cp, scope):
    base = eval_base(plan, B, cp, scope)
    if not (base / 'complete.json').exists():
        return False
    d = U.read(base / 'complete.json')
    U.require(d['checkpoint_sha256'] == cp['sha256'] and d['scope'] == scope,
              'Completed evaluation provenance changed')
    for name, h in d['files'].items():
        U.require(U.sha(base / name) == h, 'Completed evaluation file changed: ' + str(base / name))
    return True


def pending(plan, B, stage):
    _, run = segment_paths(plan, B, stage)
    pp = run / '.validation_pending' / 'pending.json'
    if pp.is_file():
        cp = U.read(pp)
        U.require(cp.get('recovery_committed'), 'Pending validation has no commit receipt; re-run --phase train to reconcile native recovery')
        if not eval_done(plan, B, cp, 'validation'):
            U.require(U.sha(cp['path']) == cp['sha256'], 'Pending validation snapshot changed')
            run_worker(plan, 'eval', B, stage, cp['path'])
        if cp.get('temporary'):
            Path(cp['path']).unlink(missing_ok=True)
        pp.unlink()
    for cp in milestones(plan, B, stage):
        if not eval_done(plan, B, cp, 'validation'):
            run_worker(plan, 'eval', B, stage, cp['path'])


def assert_ppo(model, sp, st):
    for name in ('n_steps', 'batch_size', 'n_epochs', 'gamma', 'gae_lambda', 'ent_coef',
                 'vf_coef', 'max_grad_norm', 'target_kl'):
        U.require(getattr(model, name) == st['ppo'][name], 'PPO setting changed: ' + name)
    U.require(model.n_envs == 8 and model.seed == 42, 'PPO seed/environment count changed')
    for progress in (1., .5, 0., -.01):
        expected = max(st['ppo']['learning_rate_floor'], st['ppo']['learning_rate_initial'] * progress)
        U.require(U.near(model.lr_schedule(progress), expected), 'PPO LR schedule changed')


def train_segment(plan, B, stage):
    verify_plan(plan)
    rt = runtime(plan, B, train=True)
    import numpy as np
    from stable_baselines3.common.save_util import load_from_zip_file
    sp, st = segment_spec(plan, B, stage)
    base, run = segment_paths(plan, B, stage)
    base.mkdir(parents=True, exist_ok=True)
    U.freeze(base / 'protocol.json', dict(version=VERSION, spec=sp, settings=st,
              runtime=plan['runtime']['relaxed'], manifest_sha256=plan['manifest_sha256']))
    for rec in rt.train:
        U.require(rt.utils.array_hash(np.load(rec.path, allow_pickle=False)) == rec.sha256,
                  'Training map changed: ' + rec.map_id)
    orders = None
    if stage:
        source_run = Path(st['b124_segment']['source_run'])
        source_settings = U.read(source_run / 'training_settings.json')
        orders = U.rebuild_orders(rt, dict(settings=source_settings, source_run=str(source_run)))
    source = (Path(plan['runtime']['relaxed']['path']) / (U.TRAINER + '.py')).read_text()
    patch = extract_train(source, bool(stage))
    pp = base / 'native_train_sb3_patch.py'
    if pp.exists():
        U.require(pp.read_text() == patch, 'Saved native patch changed')
    else:
        U.write_text(pp, patch)

    def fresh(cls, common, spec, settings, out):
        model = cls(**common)
        assert_ppo(model, spec, settings)
        rt.features.audit_cnn3_model(model); rt.core.audit_model(model)
        d = U.tensor_digest(model.get_parameters())
        with U.lock(Path(plan['out']) / 'audit' / '.initial_policy.lock', blocking=True):
            U.freeze(Path(plan['out']) / 'audit' / 'shared_initial_policy_seed42.json',
                     dict(seed=42, tensor_sha256=d, source_weights_used=False))
        U.write(base / 'initialization_audit.json', dict(B=B, seed=42, tensor_sha256=d,
                 source_weights_used=False, optimizer_steps=0))
        print(f'[FRESH] B={B}; same seed42 policy digest={d[:16]}; no Strict weight transfer', flush=True)
        return model

    def baseline(settings):
        e = settings['b124_segment']
        return [dict(path=e['source_model'], num_timesteps=e['source_step'], sha256=e['source_sha256'],
                     source_baseline=True, final=False)]

    def load(cls, env, spec, settings, out, records):
        e = settings['b124_segment']
        U.require(U.sha(e['source_model']) == e['source_sha256'], 'Continuation source changed')
        _, parameters, _ = load_from_zip_file(e['source_model'], device='cpu', load_data=False)
        U.require(parameters and 'policy' in parameters and 'policy.optimizer' in parameters,
                  'Native policy/Adam parameters missing')
        expected = U.tensor_digest(parameters)
        model = cls.load(e['source_model'], env=env, device='cuda:0', force_reset=True)
        U.require(U.tensor_digest(model.get_parameters()) == expected, 'Policy/Adam restoration is not exact')
        U.require(model.num_timesteps == e['source_step'] and model._n_updates == e['source_epoch_counter'],
                  'Continuation source counters mismatch')
        U.require(model.policy.optimizer.state_dict()['state'], 'Adam moments are missing')
        model.learning_rate = 5e-5; model._setup_lr_schedule()
        for group in model.policy.optimizer.param_groups:
            group['lr'] = 5e-5
        U.require(U.tensor_digest(model.get_parameters()) == expected, 'Source optimizer was not at the expected LR floor')
        model.tensorboard_log = str(Path(out) / 'tensorboard') if settings.get('tensorboard') else None
        model._last_obs = None; model._last_episode_starts = None
        for i, adapter in enumerate(env.envs):
            adapter.stream.order.load_state_dict(orders[i]['order'])
            adapter.stream.episode = orders[i]['episode']
            adapter.stream.total_steps = e['source_step'] // settings['num_envs']
        assert_ppo(model, spec, settings)
        rt.features.audit_cnn3_model(model); rt.core.audit_model(model)
        U.write(base / 'continuation_audit.json', dict(B=B, source_step=e['source_step'],
                 source_sha256=e['source_sha256'], tensor_digest=expected, policy_Adam_exact=True,
                 sampler_positions=[x['episode'] for x in orders], source_episodes_discarded=True,
                 boundary_rng_restarted=True, subsequent_resume='native full-state'))
        print(f'[CONTINUE] B={B}: {e["source_step"]:,} -> TOTAL {settings["total_timesteps"]:,}; LR=5e-5', flush=True)
        return model

    def boundary(model, spec, settings, out, checkpoints, save_resume):
        step = int(model.num_timesteps)
        m = step // INTERVAL
        # Native updates are 8192 interactions. Evaluate only at first completed
        # update crossing a global 1M boundary, not at every following update.
        if m < 1 or m > 40 or not (0 <= step - m * INTERVAL < BLOCK):
            return
        completed = Path(plan['out']) / 'validation' / f'B{B}' / f'step_{step}' / 'complete.json'
        parameters_digest = U.tensor_digest(model.get_parameters())
        if completed.is_file():
            record = U.read(completed)
            U.require(record.get('parameter_sha256') == parameters_digest, 'Restored model differs from the already evaluated optimizer boundary')
            U.require(eval_done(plan, B, dict(num_timesteps=step, sha256=record['checkpoint_sha256']), 'validation'), 'Invalid completed validation')
            return
        cp = next((deepcopy(x) for x in checkpoints if not x.get('source_baseline')
                   and not x.get('final') and int(x['num_timesteps']) == step), None)
        pp = Path(out) / '.validation_pending' / 'pending.json'
        if cp is None:
            U.require(m % 5 != 0, 'Missing permanent 5M checkpoint at validation boundary')
            destination = pp.parent / f'step_{step}.zip'
            destination.parent.mkdir(parents=True, exist_ok=True)
            temp = destination.with_name(destination.stem + '.writing.zip')
            model.save(temp)
            rt.recovery.sync_file(temp)
            os.replace(temp, destination)
            cp = dict(path=str(destination), sha256=U.sha(destination), num_timesteps=step,
                      temporary=True, final=False)
        else:
            cp.update(temporary=False, path=str(Path(out) / 'checkpoints' / Path(cp['path']).name))
        cp.update(nominal_m=m, stage_index=stage, parameter_sha256=parameters_digest, recovery_committed=False)
        U.write(pp, cp)
        # Commit training state at the same optimizer boundary BEFORE evaluation.
        # A failed/interrupted CPU evaluation retains this snapshot and its map cache.
        save_resume(model, 'before_1M_validation')
        cp['recovery_committed'] = True
        U.write(pp, cp)
        print(f'[VAL BOUNDARY] B={B}: {m}M / actual {step:,}; '
              f'permanent_checkpoint={not cp["temporary"]}; CPU validation180', flush=True)
        pending(plan, B, stage)

    def on_start(model, spec, settings, out, checkpoints, save_resume):
        pp = Path(out) / '.validation_pending' / 'pending.json'
        if pp.is_file():
            cp = U.read(pp)
            if int(cp['num_timesteps']) > int(model.num_timesteps):
                # A process died between temporary snapshot creation and native
                # recovery commit. No evaluation may have run without its receipt.
                U.require(not cp.get('recovery_committed') and not eval_done(plan, B, cp, 'validation'),
                          'Evaluation is ahead of the restored native recovery state')
                if cp.get('temporary'):
                    Path(cp['path']).unlink(missing_ok=True)
                pp.unlink()
            else:
                if not cp.get('recovery_committed'):
                    U.require(int(cp['num_timesteps']) == int(model.num_timesteps) and
                              cp['parameter_sha256'] == U.tensor_digest(model.get_parameters()),
                              'Orphan snapshot does not match restored optimizer state')
                    cp['recovery_committed'] = True; U.write(pp, cp)
                pending(plan, B, stage)
        # Covers interruption after commit but before a pending receipt was written,
        # and does not recreate temporary weights for an already completed result.
        boundary(model, spec, settings, out, checkpoints, save_resume)

    rt.trainer.__dict__.update(_b124_assert=assert_ppo, _b124_fresh=fresh, _b124_load=load,
                               _b124_baseline=baseline, _b124_boundary=boundary, _b124_start=on_start)
    exec(compile(patch, str(pp), 'exec'), rt.trainer.__dict__)
    # on_start reconciles pending validation with the actually restored full state.
    rt.trainer.train_run(sp, st, rt.train, run)
    pending(plan, B, stage)
    cp, _ = final_checkpoint(run)
    U.require(STAGES[stage] <= int(cp['num_timesteps']) < STAGES[stage] + BLOCK, 'Wrong total training budget')
    verify_plan(plan)
    return cp
