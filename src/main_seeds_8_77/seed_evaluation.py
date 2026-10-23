"""Final 40M evaluation for the new Strict/Relaxed seeds.

Call evaluate_job in a NEW worker process for each method, seed and split.
The original test8.initial_condition and test8.evaluate_map perform initialization,
masked PPO action selection, native stepping, metrics and trace instrumentation.
Only the seed-aware checkpoint loader, provenance and resumable output are new.
No optimizer call, checkpoint selection or training-data rollout occurs here.
"""
from __future__ import annotations

from collections import Counter
import importlib
import importlib.metadata
import os
from pathlib import Path
import sys
import time

from ch5_vendor import test8_original as V


VERSION = 'main_seed8_77_final40m_native_raw_refined_v1'
TARGET = 40_000_000
PACKAGES = ('numpy', 'torch', 'gymnasium', 'stable-baselines3', 'sb3-contrib')


def _checkpoint(job, result):
    V.require(result.get('state') == 'completed' or
              result.get('status', {}).get('state') == 'completed',
              'Evaluation requires completed training_result.json.')
    V.require(int(result.get('seed', job['seed'])) == int(job['seed']) and
              result.get('method', job['method']) == job['method'],
              'Training result belongs to a different method or seed.')
    path = Path(result.get('final_model') or result['model_path']).resolve()
    V.require(path.is_relative_to(Path(job['job_dir']).resolve()),
              'Use this new seed job\'s final checkpoint, not reference seed42 weights.')
    V.require(path.is_file(), 'Final checkpoint missing: ' + str(path))
    expected = result.get('final_model_sha256')
    V.require(expected is not None and V.sha(path) == expected,
              'Final checkpoint hash differs from the completed training receipt.')
    settings = result.get('final_settings', job['settings'])
    step = result.get('actual_timesteps')
    if step is None:
        status = result.get('status', {})
        step = status.get('actual_timesteps', status.get('num_timesteps'))
    V.require(step is not None, 'Training receipt must include actual_timesteps.')
    step = int(step)
    rollout = int(settings['num_envs']) * int(settings['ppo']['n_steps'])
    V.require(TARGET <= step < TARGET + rollout,
              'Final checkpoint is outside the native nominal-40M rollout boundary.')
    return dict(path=str(path), sha256=expected, num_timesteps=step)


def _native_job(job, result, checkpoint):
    method, seed = job['method'], int(job['seed'])
    V.require(method in ('strict', 'relaxed') and seed in (8, 77),
              'This package contains only the two authorized main methods and seeds 8/77.')
    spec = result.get('final_spec', job['spec'])
    settings = result.get('final_settings', job['settings'])
    env = spec['environment']
    expected_mode = 'progress' if method == 'strict' else 'relaxed'
    native_version = V.STRICT_CORE_VERSION if method == 'strict' else V.RELAXED_CORE_VERSION
    V.require(spec['seed'] == seed and spec['algorithm'] == 'ppo', 'Wrong final policy specification.')
    V.require(env == job['spec']['environment'], 'Final environment differs from the frozen new-seed job.')
    V.require(env['mask_mode'] == expected_mode and V.close(env['tradeoff_beta'], .5) and
              int(env['detour_budget_max']) == 6 and env['initialization'] == 'priority_mass',
              'Expected original main FULL COST11 beta0.5 environment and budget observation.')
    V.require(job.get('native_version', native_version) == native_version, 'Wrong native core version.')
    V.require(job['evaluation']['seed'] == V.EVAL_SEED and
              job['evaluation']['initial_seed'] == V.INITIAL_SEED,
              'Training seed changes must not change evaluation seed or initial conditions.')
    V.require(int(job.get('effective_B', 0 if method == 'strict' else 6)) ==
              (0 if method == 'strict' else 6), 'Wrong effective mask budget.')
    snapshot = Path(job['snapshot']).resolve()
    hashes = job['runtime_hashes']
    for name, expected in hashes.items():
        V.require(Path(name).name == name and V.sha(snapshot / name) == expected,
                  'Frozen native source missing or changed: ' + name)
    required = set(('coverage_core.py', 'core_ch4.py', 'core_ch4_bounded_b6.py',
                    'full_v2_utils.py', V.CORE + '.py', V.FEATURES + '.py'))
    V.require(required <= set(hashes), 'Native environment/CNN source hashes are incomplete.')
    return dict(key=f'{method}_seed{seed}', label=f'{method.title()} seed {seed} 40M',
                kind='ppo', family=method, runtime_kind=method, beta=.5,
                B=0 if method == 'strict' else 6, nominal_m=40, seed=seed,
                native_version=native_version, snapshot=str(snapshot),
                sources={n: dict(sha256=h) for n, h in hashes.items()},
                spec=spec, settings=settings, checkpoint=checkpoint,
                software=job['software'])


def _load_runtime(native):
    # ch5_common uses the authenticated original adapter.wrap_factory directly.
    # It does not install a trainer logger or import a replacement generic reward.
    import ch5_common as C
    for name in PACKAGES:
        actual = importlib.metadata.version(name)
        V.require(native['software'].get(name) == actual,
                  f'Use the original environment: {name} expected '
                  f'{native["software"].get(name)}, actual {actual}. No automatic upgrades.')
    snapshot = Path(native['snapshot'])
    names = [Path(n).stem for n in C.NATIVE_FILES] + [V.TRAINER, V.FEATURES, V.LOGGER]
    for name in names:
        V.require(name not in sys.modules,
                  'Evaluation needs a new worker process; native module already loaded: ' + name)
    rt = C.load_native(native)
    sys.path.insert(0, str(snapshot))
    importlib.invalidate_caches()
    try:
        feature = importlib.import_module(V.FEATURES)
        V.require(Path(feature.__file__).resolve() == snapshot / (V.FEATURES + '.py'),
                  'CNN feature module escaped the frozen snapshot.')
        for module in list(sys.modules.values()):
            file = getattr(module, '__file__', None)
            if file and Path(file).suffix == '.py' and Path(file).resolve().parent == snapshot:
                name = Path(file).name
                V.require(name in native['sources'] and V.sha(file) == native['sources'][name]['sha256'],
                          'Unverified native import: ' + str(file))
    finally:
        sys.path.remove(str(snapshot))
    rt.modules[V.FEATURES] = feature
    V.require(V.close(rt.core.NORMALIZATION_STEP_FACTOR, 5), 'Wrong native COST11 normalization.')
    return rt


def _load_model(rt, native):
    import torch
    from sb3_contrib import MaskablePPO
    cp = native['checkpoint']
    V.require(V.sha(cp['path']) == cp['sha256'], 'Final weights changed before inference.')
    torch.set_num_threads(int(native['settings'].get('torch_threads', 4)))
    # Keep the authenticated snapshot importable while SB3 restores serialized
    # policy kwargs/callables, as in the original native checkpoint loader.
    snapshot = Path(native['snapshot'])
    sys.path.insert(0, str(snapshot))
    try:
        model = MaskablePPO.load(cp['path'], device='cpu')
        for module in list(sys.modules.values()):
            file = getattr(module, '__file__', None)
            if file and Path(file).suffix == '.py' and Path(file).resolve().parent == snapshot:
                name = Path(file).name
                V.require(name in native['sources'] and V.sha(file) == native['sources'][name]['sha256'],
                          'Checkpoint imported unverified native code: ' + str(file))
    finally:
        sys.path.remove(str(snapshot))
    V.require(int(model.num_timesteps) == cp['num_timesteps'], 'Loaded policy has the wrong training step.')
    V.require(int(model.seed) == native['seed'] and
              int(model.n_envs) == int(native['settings']['num_envs']), 'Loaded policy has the wrong seed/env count.')
    for name in ('n_steps', 'batch_size', 'n_epochs', 'gamma', 'gae_lambda',
                 'ent_coef', 'vf_coef', 'max_grad_norm'):
        V.require(V.close(getattr(model, name), native['settings']['ppo'][name]),
                  'Loaded PPO setting differs: ' + name)
    rt.core.audit_model(model)
    rt.modules[V.FEATURES].audit_cnn3_model(model)
    model.policy.set_training_mode(False)
    return model


def _save_npz(path, arrays):
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    V.require(not path.is_symlink(), 'Symlink output refused: ' + str(path))
    temporary = path.with_name(path.name + '.writing.' + str(os.getpid()))
    try:
        with temporary.open('wb') as file:
            np.savez_compressed(file, **arrays)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _receipt(directory, name, identity, required):
    path = directory / name
    if not path.exists():
        return None
    receipt = V.read(path)
    V.require(receipt['identity'] == identity and set(required) <= set(receipt.get('files', {})),
              'Committed evaluation cache has a different identity or missing files: ' + str(path))
    for filename, expected in receipt['files'].items():
        V.require(Path(filename).name == filename and V.sha(directory / filename) == expected,
                  'Committed evaluation output changed: ' + str(directory / filename))
    return receipt


def _commit(directory, filename, identity, rec, files):
    V.write(directory / filename, dict(identity=identity, map_id=rec['map_id'],
            map_sha256=rec['sha256'], files={name: V.sha(directory / name) for name in files}))


def _check_row(row, native, rec, ic):
    for name, value in [('map_id', rec['map_id']), ('map_sha256', rec['sha256']),
                        ('source', rec['source']), ('split', rec['split']),
                        ('initial_pair_id', ic['pair_id']), ('method', native['key']),
                        ('checkpoint_sha256', native['checkpoint']['sha256'])]:
        V.require(row.get(name) == value, 'Cached result identity changed: ' + name)
    V.require(int(row['train_seed']) == native['seed'] and
              int(row['checkpoint_step']) == native['checkpoint']['num_timesteps'],
              'Cached result seed/checkpoint step changed.')
    V.require([int(row['start_r']), int(row['start_c'])] == ic['start'] and
              int(row['start_heading']) == int(ic['heading']), 'Cached initial state changed.')


def _process_map(rt, native, rec, base, identity, get_model):
    import numpy as np
    import ch5_common as C
    directory = base / 'maps' / V.map_key(rec)
    # Recheck the actual selected input map even when a completed result is reused.
    array = np.load(rec['path'], allow_pickle=False)
    V.require(rt.utils.array_hash(array) == rec['sha256'], 'Evaluation map changed: ' + rec['map_id'])
    ic = V.initial_condition(rt, rec, array)
    raw_files = ('raw_metrics.json', 'raw_trace.npz', 'raw_provenance.json')
    final_files = raw_files + ('raw_complete.json', 'refined_metrics.json',
                              'refined_trace.npz', 'refinement_audit.json', 'segments.csv')
    if _receipt(directory, 'complete.json', identity, final_files) is not None:
        raw = V.read(directory / 'raw_metrics.json')
        refined = V.read(directory / 'refined_metrics.json')
        audit = V.read(directory / 'refinement_audit.json')
        for row in (raw, refined):
            _check_row(row, native, rec, ic)
        V.require(V.truth(raw['is_success']) == V.truth(refined['is_success']), 'Cached success changed.')
        return raw, refined, audit, 'completed_refinement_reused'
    if _receipt(directory, 'raw_complete.json', identity, raw_files) is not None:
        raw = V.read(directory / 'raw_metrics.json')
        with np.load(directory / 'raw_trace.npz', allow_pickle=False) as stored:
            trace = {name: stored[name].copy() for name in stored.files}
        origin = 'saved_raw_trace_reused'
    else:
        # Unchanged original routine: all mission metrics come from core.info(final=True).
        raw, trace = V.evaluate_map(rt, native, get_model(), rec, array, ic)
        raw.update(train_seed=native['seed'], model=native['key'], deterministic=True)
        _save_npz(directory / 'raw_trace.npz', trace)
        V.write(directory / 'raw_metrics.json', raw)
        V.write(directory / 'raw_provenance.json', dict(
            origin='own_final40m_checkpoint_native_cpu_masked_argmax',
            train_seed=native['seed'], checkpoint=native['checkpoint'],
            metrics_source='unchanged_native_core.info(final=True)',
            initial_condition_function='ch5_vendor.test8_original.initial_condition',
            rollout_function='ch5_vendor.test8_original.evaluate_map',
            optimizer_updates=0, refinement_applied_in_this_stage=False))
        _commit(directory, 'raw_complete.json', identity, rec, raw_files)
        origin = 'own_final40m_native_raw_inferred'
    _check_row(raw, native, rec, ic)
    V.require(rt.utils.array_hash(trace['priority_map']) == rec['sha256'], 'Cached trace map changed.')
    refined, arrays, audit, segments = C.common_refine(rt, raw, trace)
    refined.update(train_seed=native['seed'], model=native['key'], deterministic=True)
    audit.update(train_seed=native['seed'], method=native['key'], map_id=rec['map_id'],
                 map_sha256=rec['sha256'], source=rec['source'], split=rec['split'],
                 initial_pair_id=ic['pair_id'], checkpoint_step=native['checkpoint']['num_timesteps'],
                 checkpoint_sha256=native['checkpoint']['sha256'], raw_provenance=origin)
    _save_npz(directory / 'refined_trace.npz', arrays)
    V.write(directory / 'refined_metrics.json', refined)
    V.write(directory / 'refinement_audit.json', audit)
    V.csv_write(directory / 'segments.csv', segments)
    _commit(directory, 'complete.json', identity, rec, final_files)
    return raw, refined, audit, origin


def _summary(rows, native, split, stage):
    groups = []
    for scope in ['all'] + sorted({r['source'] for r in rows}):
        selected = rows if scope == 'all' else [r for r in rows if r['source'] == scope]
        success = [r for r in selected if V.truth(r['is_success'])]
        groups.append(dict(source=scope, n_maps=len(selected), n_success=len(success),
            success_pct=100 * len(success) / len(selected) if selected else None,
            time_mean_population='own_successful_maps', time_mean_n=len(success),
            **{V.T: V.mean(r[V.T] for r in success), V.LW: V.mean(r[V.LW] for r in success)},
            coverage_pct_all=V.mean(r['coverage_pct'] for r in selected),
            mission_time_mean_all_prefixes_sec=V.mean(r[V.T] for r in selected)))
    return dict(method=native['family'], train_seed=native['seed'], split=split, stage=stage,
                checkpoint=native['checkpoint'], groups=groups,
                note='Times in the main fields use successful maps only; failed-path Lw stays undefined. '
                     'Validation and test are separate. Across-seed paired analyses should join map IDs '
                     'and initial_pair_id, and declare a common-success population when failures differ.')


def evaluate_job(job, training_result, split='test'):
    """Evaluate this seed's own final model on Val180 OR Test1220, with resumable traces.

    Required job fields: job_dir, seed, method, snapshot, runtime_hashes, spec,
    settings, manifest, manifest_sha256, software and evaluation(seed, initial_seed).
    training_result is the committed train_job return value. Run in a fresh process.
    """
    V.require(split in ('validation', 'test'), 'Only final validation or test evaluation is supported.')
    cp = _checkpoint(job, training_result)
    native = _native_job(job, training_result, cp)
    manifest = Path(job['manifest']).resolve()
    V.require(V.sha(manifest) == job['manifest_sha256'], 'Frozen dataset manifest changed.')
    rt = _load_runtime(native)
    records_all = rt.utils.load_manifest(manifest, 50, verify_files=False)
    V.require(Counter((r.source, r.split) for r in records_all) == V.EXPECTED_COUNTS,
              'Expected fixed Synthetic train900/validation180/test720 and OSM test500.')
    selected = [r for r in records_all if r.split == split]
    expected = 180 if split == 'validation' else 1220
    V.require(len(selected) == expected and
              (split != 'validation' or all(r.source == 'synthetic' for r in selected)),
              'Wrong held-out evaluation membership.')
    records = [dict(map_id=r.map_id, path=r.path, source=r.source, split=r.split,
                    pattern=r.pattern, tier=r.tier, sha256=r.sha256) for r in selected]
    here = Path(__file__).resolve().parent
    code_names = ('seed_evaluation.py', 'ch5_common.py', 'ch5_vendor/test8_original.py',
                  'ch5_vendor/segment_graph_original.py')
    params = dict(version=VERSION, train_seed=native['seed'], method=native['family'],
                  checkpoint=cp, final_spec=native['spec'], final_settings=native['settings'],
                  runtime_hashes=job['runtime_hashes'], software=job['software'],
                  manifest_sha256=job['manifest_sha256'], split=split,
                  membership=[(r['map_id'], r['sha256'], r['source'], r['split']) for r in records],
                  evaluation=dict(seed=V.EVAL_SEED, initial_seed=V.INITIAL_SEED,
                                  deterministic=True, device='cpu'),
                  code_sha256={name: V.sha(here / name) for name in code_names})
    identity = V.digest(params)
    base = Path(job['job_dir']).resolve() / 'evaluation' / split
    raw_rows, refined_rows, audits = [], [], []
    origins = Counter()
    model = None
    started = time.time()

    def get_model():
        nonlocal model
        if model is None:
            model = _load_model(rt, native)
        return model

    with V.lock(base / '.evaluation.lock'):
        V.freeze(base / 'identity.json', dict(sha256=identity, parameters=params))
        try:
            for i, rec in enumerate(records, 1):
                raw, refined, audit, origin = _process_map(rt, native, rec, base, identity, get_model)
                raw_rows.append(raw)
                refined_rows.append(refined)
                audits.append(audit)
                origins[origin] += 1
                progress = dict(method=native['family'], train_seed=native['seed'], split=split,
                    completed=i, expected=expected, success=sum(V.truth(r['is_success']) for r in raw_rows),
                    changed_paths=sum(bool(a['path_changed']) for a in audits),
                    cache_origins=dict(origins), last_map=rec['map_id'], elapsed_sec=time.time()-started)
                V.write(base / 'progress.json', progress)
                if i == 1 or i % 25 == 0 or i == expected:
                    V.csv_write(base / 'raw_per_map.csv', raw_rows)
                    V.csv_write(base / 'refined_per_map.csv', refined_rows)
                    V.csv_write(base / 'refinement_audit_per_map.csv', audits)
                    print(f'[EVAL] {native["key"]} {split} {i}/{expected}; '
                          f'success={progress["success"]}; changed={progress["changed_paths"]}; '
                          f'elapsed={progress["elapsed_sec"]:.1f}s', flush=True)
            V.write(base / 'summary.json', dict(
                raw=_summary(raw_rows, native, split, 'raw'),
                refined=_summary(refined_rows, native, split, 'refined')))
            files = ('raw_per_map.csv', 'refined_per_map.csv', 'refinement_audit_per_map.csv', 'summary.json')
            result = dict(state='completed', identity=identity, method=native['family'],
                seed=native['seed'], split=split, n_maps=len(raw_rows), checkpoint=cp,
                output_dir=str(base), files={name: V.sha(base / name) for name in files},
                optimizer_updates=0, cache_origins=dict(origins))
            V.write(base / 'complete.json', result)
            return result
        finally:
            model = None
