"""Per-kappa adapter for the uploaded COST1 runner; no new PPO/refinement engine.

The source snapshot is private to each new case. Only the two time-cost terms,
return-identity accounting, and version/diagnostic labels change in the core.
All upstream files shipped under upstream/ remain byte-identical.
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import json
import math
import os
import re
from collections import Counter
import shutil
import sys
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
UP = HERE / 'upstream'
sys.dont_write_bytecode = True
sys.path.insert(0, str(UP))
import run_kappa1_fresh5m as F
E = F.E

VERSION = 'table514_kappa5_15_20_fresh5m_v1_20261007'
ADDED = (5, 15, 20)
EXISTING = (1, 3, 7, 9, 10, 11, 12, 13, 14)
ALL_COSTS = tuple(sorted(EXISTING + ADDED))
ORIGIN = 'fresh_training_from_scratch_added_kappa_not_historical_recovery'
TARGET = 5_000_000
FINAL_STEP = 5_005_312
CORE_NAME = 'nrg1_c3nt_core.py'
ORIGINAL_CORE_HASH = E.NATIVE_HASH
ORIGINAL_NATIVE_VERSION = E.NATIVE_VERSION
ORIGINAL_RUN_ID = E.RUN_ID
ORIGINAL_NORMALIZED = E.normalized
ORIGINAL_PROCESS = inspect.getsource(F.process)
ORIGINAL_TRAIN = inspect.getsource(F.train)
ORIGINAL_EVALUATE = inspect.getsource(F.evaluate)
_ACTIVE_KAPPA = None


def need(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def native_version(kappa: int) -> str:
    return f'table514_cnn3_noclock_externalcap_cost{kappa}_v1_20261007'


def run_id(kappa: int) -> str:
    return f'ppo_relaxedB6_C3NT_COST{kappa}_FULL_H5_b0.5_seed42_fresh5M_table514'


def formula(kappa: int) -> str:
    return f'-{kappa}*((1-beta)+beta*W_pre/W_total)*dt/(ceil(5*N)*time_turn_sec)'


def verify_package() -> dict[str, str]:
    """Verify both the extension and the original uploaded package."""
    expected = E.read(HERE / 'PACKAGE_SHA256.json')
    for name, digest in expected.items():
        p = E.member(HERE, name)
        need(p.is_file() and E.sha(p) == digest, 'Package file changed/missing: ' + name)
    old = E.read(UP / 'PACKAGE_SHA256.json')
    for name, digest in old.items():
        p = E.member(UP, name)
        need(p.is_file() and E.sha(p) == digest, 'Uploaded original changed: ' + name)
    need(E.sha(UP / 'native' / CORE_NAME) == ORIGINAL_CORE_HASH,
         'The uploaded COST1 core differs from the expected source.')
    return expected


def make_request(kappa: int, device: str) -> dict[str, Any]:
    need(kappa in ADDED, 'Only kappa=5,15,20 may be newly trained.')
    spec = E.read(UP / 'reference/fresh_spec.json')
    settings = E.read(UP / 'reference/training_settings.json')
    evaluation = E.read(UP / 'reference/evaluation.json')
    settings['device'] = device
    spec.update(run_id=run_id(kappa), configuration_id=run_id(kappa).rsplit('_seed', 1)[0],
                reward_case=f'FULL_C3NT_COST{kappa}', stage='table514_added_kappa_fresh5m',
                experiment_origin=ORIGIN,
                experiment_purpose='Add kappa=5,15,20; reuse nine completed conditions without inference')
    spec['reward_definition'].update(version=native_version(kappa), cost_multiplier=float(kappa),
                                     cost=formula(kappa), source_reward_version=ORIGINAL_NATIVE_VERSION)
    need(settings['total_timesteps'] == TARGET and settings['num_envs'] == 8,
         'Original training settings changed.')
    need(settings['ppo']['n_steps'] == 1024, 'Unexpected original PPO rollout size.')
    return dict(version=native_version(kappa), launcher_version=VERSION, kappa=kappa,
                experiment_origin=ORIGIN, spec=spec, settings=settings, evaluation=evaluation,
                eval_device='cpu', warm_start=False, old_checkpoint_required=False,
                test_enabled=False, osm_enabled=False,
                existing_kappas_training=False, existing_kappas_inference=False,
                existing_kappas_refinement=False,
                settings_source='Uploaded CH5_KAPPA1_FRESH5M_v1; only kappa and case identity change',
                reference_settings_sha256=E.sha(UP / 'reference/training_settings.json'),
                original_manifest_array_membership='upstream/reference/train_validation_manifest.csv',
                historical_success_count_is_not_an_acceptance_target=True)


def _replace_assignment_string(source: str, field: str, before: str, after: str) -> str:
    tree = ast.parse(source)
    hits = 0
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == field for t in n.targets):
            need(isinstance(n.value, ast.Constant) and n.value.value == before,
                 'Unexpected source assignment: ' + field)
            n.value = ast.Constant(after)
            hits += 1
    need(hits == 1, 'Expected one source assignment: ' + field)
    ast.fix_missing_locations(tree)
    result = ast.unparse(tree) + '\n'
    compile(result, field + '_generated.py', 'exec')
    return result


def scaled_core(kappa: int) -> tuple[str, dict[str, Any]]:
    """An AST-limited cost scaling. Coverage, terminal, masks and physics stay intact."""
    source = (UP / 'native' / CORE_NAME).read_text(encoding='utf-8')
    tree = ast.parse(source)
    cost_fields = {'latency', 'mission_time'}
    hits = dict(version=0, multiplier=0, reward_terms=0, aggregate_cost=0, effective_o=0, info_label=0)
    # Retain a structural witness for all methods that must NOT change.
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'C3NoClockMixin')
    protected_methods = {n.name: ast.dump(n, include_attributes=False) for n in cls.body
                         if isinstance(n, ast.FunctionDef) and n.name != 'info'}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'VERSION' for t in node.targets):
            need(isinstance(node.value, ast.Constant) and node.value.value == ORIGINAL_NATIVE_VERSION,
                 'Source VERSION changed')
            node.value = ast.Constant(native_version(kappa)); hits['version'] += 1
        if isinstance(node, ast.FunctionDef) and node.name == 'reward_parts':
            returns = [n for n in ast.walk(node) if isinstance(n, ast.Return) and isinstance(n.value, ast.Dict)]
            need(len(returns) == 1, 'Source reward dictionary changed')
            d = returns[0].value
            for i, key in enumerate(d.keys):
                if isinstance(key, ast.Constant) and key.value in cost_fields:
                    d.values[i] = ast.BinOp(ast.Name('COST_MULTIPLIER', ast.Load()), ast.Mult(), d.values[i])
                    hits['reward_terms'] += 1
    for n in ast.walk(cls):
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'cost' for t in n.targets):
            need(ast.unparse(n.value) == '((1 - beta) * self.elapsed_time_sec + beta * wait) / self._nrg_cap',
                 'Source return identity changed')
            n.value = ast.BinOp(ast.Name('COST_MULTIPLIER', ast.Load()), ast.Mult(), n.value)
            hits['aggregate_cost'] += 1
        if isinstance(n, ast.Call) and ast.unparse(n.func) == 'result.update':
            keys = {kw.arg for kw in n.keywords}
            if 'nrg_version' in keys:
                for kw in n.keywords:
                    if kw.arg == 'nrg_effective_legacy_o':
                        need(ast.unparse(kw.value) == 'self.time_straight_sec / self._nrg_cap',
                             'Source legacy-o diagnostic changed')
                        kw.value = ast.BinOp(ast.Name('COST_MULTIPLIER', ast.Load()), ast.Mult(), kw.value)
                        hits['effective_o'] += 1
                n.keywords.append(ast.keyword('c3nt_cost_multiplier', ast.Name('COST_MULTIPLIER', ast.Load())))
                hits['info_label'] += 1
    pos = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'VERSION' for t in n.targets))
    tree.body.insert(pos + 1, ast.parse(f'COST_MULTIPLIER = {float(kappa)!r}').body[0])
    hits['multiplier'] += 1
    need(hits == dict(version=1, multiplier=1, reward_terms=2, aggregate_cost=1, effective_o=1, info_label=1),
         'Unexpected cost patch counts: ' + str(hits))
    for n in cls.body:
        if isinstance(n, ast.FunctionDef) and n.name in protected_methods:
            need(ast.dump(n, include_attributes=False) == protected_methods[n.name],
                 'Core movement/mask/observation/step logic modified: ' + n.name)
    ast.fix_missing_locations(tree)
    text = ast.unparse(tree) + '\n'
    compile(text, CORE_NAME, 'exec')
    return text, dict(kappa=kappa, patch_counts=hits, base_core_sha256=ORIGINAL_CORE_HASH,
                      generated_core_sha256=hashlib.sha256(text.encode()).hexdigest(),
                      physics_masks_observations_step_ast_unchanged=True,
                      coverage_coefficient=1, completion_reward=2, failure_terminal_reward=0)


def install_snapshot(out: Path, kappa: int) -> dict[str, str]:
    snap = out / 'source_snapshot'; snap.mkdir(exist_ok=True)
    generated, audit = scaled_core(kappa)
    hashes = {}
    for source in sorted((UP / 'native').glob('*.py')):
        name = source.name; data = source.read_bytes()
        if name == CORE_NAME:
            data = generated.encode('utf-8')
        elif name == 'run_nrg1_c3nt_logfix.py':
            text = _replace_assignment_string(data.decode(), 'C3NT_VERSION', ORIGINAL_NATIVE_VERSION, native_version(kappa))
            text = _replace_assignment_string(text, 'RUN_ID', ORIGINAL_RUN_ID, run_id(kappa))
            data = text.encode()
        elif name == 'run_nrg1_c3nt.py':
            text = _replace_assignment_string(data.decode(), 'VERSION', ORIGINAL_NATIVE_VERSION, native_version(kappa))
            text = _replace_assignment_string(text, 'RUN_ID', ORIGINAL_RUN_ID, run_id(kappa))
            data = text.encode()
        dest = snap / name
        if dest.exists():
            need(dest.read_bytes() == data, 'Private snapshot changed: ' + str(dest))
        else:
            dest.write_bytes(data)
        hashes[name] = E.sha(dest)
    protected = ('coverage_core.py', 'core_ch4.py', 'core_ch4_bounded_b6.py', 'nrg1_c3nt_features.py',
                 'trainers_ch4_bounded_b6_nrg1_c3nt.py', 'resume_ch4.py', 'full_v2_utils.py')
    for name in protected:
        need(hashes[name] == E.sha(UP / 'native' / name), 'Protected uploaded code changed: ' + name)
    audit['byte_identical_native_files'] = list(protected)
    audit['refinement_engine_sha256'] = E.sha(UP / 'engine/ch5_common.py')
    audit['segment_graph_sha256'] = E.sha(UP / 'engine/ch5_vendor/segment_graph_original.py')
    E.write(out / 'audit/kappa_source_patch.json', audit)
    return hashes


def source_context(out: Path, explicit: Path | None = None) -> dict[str, Any]:
    """Validate a NEW final case, not an original coefficient-one run."""
    base = Path(explicit if explicit is not None else out).resolve()
    pr = E.read(base / 'protocol.json'); req = pr['request']; kappa = int(req['kappa'])
    need(kappa in ADDED and req == make_request(kappa, req['settings']['device']), 'Wrong new-case protocol')
    sp, st = req['spec'], req['settings']; run = base / 'runs' / run_id(kappa)
    status = E.read(run / 'status.json'); actual = E.read(run / 'run_spec.json')
    need(status.get('state') == 'completed', 'Training has not completed: ' + str(run))
    need(int(status['actual_timesteps']) == FINAL_STEP, 'Wrong final 5M optimizer boundary')
    need(E.read(run / 'training_settings.json') == st, 'Training settings mismatch')
    need(actual.get('nrg1_protocol_sha256') == E.sha(base / 'protocol.json'), 'Run/protocol hash mismatch')
    need({k: v for k, v in actual.items() if k != 'nrg1_protocol_sha256'} == sp, 'Run/spec mismatch')
    cp = run / 'last_model.zip'; h = E.sha(cp)
    finals = [r for r in E.read(run / 'checkpoints.json') if r.get('final')]
    need(len(finals) == 1 and int(finals[0]['num_timesteps']) == FINAL_STEP and finals[0]['sha256'] == h,
         'Final model registry mismatch')
    with zipfile.ZipFile(cp) as z:
        need(z.getinfo('data').file_size < 32_000_000, 'Oversized SB3 metadata')
        data = json.loads(z.read('data'))
        need(data['num_timesteps'] == FINAL_STEP and data['seed'] == 42 and data['n_envs'] == st['num_envs'],
             'Final checkpoint counters/seed differ')
        for key in ('n_steps', 'batch_size', 'n_epochs', 'gamma', 'gae_lambda', 'ent_coef', 'vf_coef',
                    'max_grad_norm', 'target_kl'):
            need(data[key] == st['ppo'][key], 'Final checkpoint PPO mismatch: ' + key)
        need(z.read('_stable_baselines3_version').decode().strip() == '2.7.0', 'Not original SB3 2.7.0')
    hashes = pr['snapshot_hash']
    expected_core = scaled_core(kappa)[1]['generated_core_sha256']
    need(hashes[CORE_NAME] == expected_core, 'Wrong cost-specific core')
    for name, digest in hashes.items():
        need(E.sha(base / 'source_snapshot' / name) == digest, 'Snapshot hash mismatch: ' + name)
    need(E.sha(base / 'manifest.csv') == pr['manifest_sha256'], 'Manifest changed')
    rows = E.rows(base / 'manifest.csv')
    need(Counter(r['split'] for r in rows) == {'train': 900, 'validation': 180}, 'Wrong 900/180 membership')
    va = [r for r in rows if r['split'] == 'validation']
    va.sort(key=lambda r: (r['source'], str(r['tier']), r['pattern'], r['map_id']))
    return dict(root=str(out), base=str(base), run=str(run), spec=actual, settings=st, kappa=kappa,
                checkpoint_sha256=h, checkpoint_step=FINAL_STEP, checkpoint_path=str(cp),
                snapshot=str(base / 'source_snapshot'), hashes=hashes, records=va,
                evaluation=req['evaluation'], native_version=native_version(kappa))


def _adapt_function(source: str, kappa: int, *, process: bool = False) -> None:
    """Change case labels and the single summary cost tag, NOT training operations."""
    tree = ast.parse(source); cost_hits = 0
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            n.value = re.sub(r'\bCOST1\b', f'COST{kappa}', n.value).replace('kappa1_fresh_check_', f'kappa{kappa}_fresh_check_')
        if process and isinstance(n, ast.keyword) and n.arg == 'cost':
            need(isinstance(n.value, ast.Constant) and n.value.value == 1, 'Process summary cost changed upstream')
            n.value = ast.Constant(kappa); cost_hits += 1
        if isinstance(n, ast.keyword) and n.arg == 'historical_kappa1_reproduced':
            n.arg = 'historical_result_reproduced'
        if isinstance(n, ast.keyword) and n.arg == 'old_cost1_success_not_enforced':
            n.arg = 'historical_success_not_enforced'
    if process:
        need(cost_hits == 1, 'Expected one per-map summary cost label')
    ast.fix_missing_locations(tree)
    exec(compile(tree, str(HERE / 'kappa_extension.py') + ':adapted', 'exec'), F.__dict__)


def configure(kappa: int) -> None:
    """Configure ONE case per process to prevent module/policy cross-contamination."""
    global _ACTIVE_KAPPA
    need(kappa in ADDED, 'Unsupported newly trained kappa')
    need(_ACTIVE_KAPPA in (None, kappa), 'Different kappa requires a fresh worker process')
    _ACTIVE_KAPPA = kappa
    E.NATIVE_VERSION = native_version(kappa); E.RUN_ID = run_id(kappa); E.MODEL = None
    E.NATIVE_HASH = scaled_core(kappa)[1]['generated_core_sha256']
    E.source_context = source_context
    def normalized(row, rec, ic, job):
        r = ORIGINAL_NORMALIZED(row, rec, ic, job)
        r.update(method=f'relaxed_kappa{kappa}', label=f'Relaxed COST{kappa} beta0.5 5M',
                 kappa=kappa, run_id=run_id(kappa), c3nt_cost_multiplier=float(kappa))
        return r
    E.normalized = normalized
    F.VERSION = VERSION; F.RUN_ID = run_id(kappa); F.EXPERIMENT_ORIGIN = ORIGIN
    F.make_request = lambda device: make_request(kappa, device)
    F.install_snapshot = lambda out: install_snapshot(out, kappa)
    F.report = case_report
    F.worker_init = refinement_worker_init
    _adapt_function(ORIGINAL_PROCESS, kappa, process=True)
    _adapt_function(ORIGINAL_TRAIN, kappa)
    _adapt_function(ORIGINAL_EVALUATE, kappa)


def check_rows(rr: list[dict[str, Any]], expected_costs: tuple[int, ...], *, strict_new: bool = False) -> None:
    """Reject partial/duplicate maps, mixed membership, and fabricated failure Lw."""
    need(len(rr) == 180 * len(expected_costs), 'Need 180 complete records per kappa, including failures')
    other = F.refs()[0]
    reference = {r['map_id']: r for r in other if int(float(r['cost'])) == 3}
    seen = set()
    for r in rr:
        c = int(float(r['cost'])); key = (c, r['map_id'])
        need(c in expected_costs and key not in seen, 'Unexpected/duplicate kappa/map: ' + str(key)); seen.add(key)
        base = reference.get(r['map_id'])
        need(base is not None and r['map_sha256'] == base['map_sha256'] and r['initial_pair_id'] == base['initial_pair_id'],
             'Validation map/hash/initial pair differs: ' + r['map_id'])
        need(r['source'] == 'synthetic' and (r.get('split') or 'validation') == 'validation', 'Test/OSM records are forbidden')
        for flag in ('actual_source_replay_verified', 'native_witness_replay_verified', 'witness_respects_original_execution_cap'):
            need(E.flag(r[flag]), 'Unverified source/refinement record: ' + flag)
        ok = E.flag(r['is_success']); t, tr = E.num(r[E.T]), E.num(r['witness_time_sec'])
        need(t is not None and tr is not None and 0 <= tr <= t + 2e-6, 'Invalid Raw/Refined elapsed time')
        lw, lwr = E.num(r['original_Lw_sec']), E.num(r['witness_Lw_sec'])
        cov = E.num(r['coverage_pct'])
        need(cov is not None and 0 <= cov <= 100 + 1e-6, 'Invalid coverage')
        need(int(float(r['witness_steps'])) <= int(float(r['steps'])), 'Refinement expanded steps')
        if ok:
            need(lw is not None and lwr is not None and 0 <= lwr <= lw + 2e-6, 'Successful path lacks valid Lw')
            need(abs(cov - 100) <= 1e-6, 'Success without complete coverage')
        else:
            need(lw is None and lwr is None, 'Failed path must have undefined complete-mission Lw')
            # Old eight-cost diagnostics can contain optimized reached prefixes.
            # Preserve those archival values, but NEVER use them as completion times.
            # The uploaded new-run common_refine preserves all failed actions.
            if c == 1 or c in ADDED:
                E.near(t, tr, 'New-run failed prefix must remain unchanged')
        if c == 1 or strict_new:
            need(int(float(r['checkpoint_step'])) == FINAL_STEP, 'Existing/new run is not final 5M')
            h = r['checkpoint_sha256']
            need(isinstance(h, str) and len(h) == 64 and all(x in '0123456789abcdef' for x in h), 'Invalid checkpoint hash')
    need(all(sum(c == cost for c, mid in seen) == 180 for cost in expected_costs), 'Missing kappa records')


def table(rr: list[dict[str, Any]], costs: tuple[int, ...]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    check_rows(rr, costs)
    grouped = {c: {r['map_id']: r for r in rr if int(float(r['cost'])) == c} for c in costs}
    common = set(next(iter(grouped.values())))
    for group in grouped.values():
        common &= {m for m, r in group.items() if E.flag(r['is_success'])}
    if common:
        result, members = E.paired_table(rr, required=costs)
    else:
        result = [dict(kappa=c, N_validation=180, success_n=sum(E.flag(r['is_success']) for r in g.values()),
                       common_n=0, Raw_T_sec=None, Raw_Lw_sec=None, Refined_T_sec=None, Refined_Lw_sec=None)
                  for c, g in grouped.items()]
        members = []
    for row in result:
        c = row['kappa']; row.update(training_M=5, success_pct=100 * row['success_n'] / 180,
                                     origin='fresh_added_run' if c in ADDED else 'existing_saved_result')
        for metric in ('T', 'Lw'):
            a, b = row[f'Raw_{metric}_sec'], row[f'Refined_{metric}_sec']
            row[f'{metric}_reduction_pct'] = None if a is None or a == 0 else 100 * (a - b) / a
    return result, members


def case_report(out: Path, job: dict | None = None, ident: str | None = None) -> dict:
    """Per-case cache report; never invokes any old kappa or writes a nine-cost table."""
    out = Path(out); out.joinpath('report').mkdir(exist_ok=True)
    kappa = int(E.read(out / 'protocol.json')['request']['kappa']); own = []
    if job is not None:
        for rec in job['records']:
            d = F.mapdir(out, rec)
            if F.committed(d, 'complete.json', ident):
                row = E.read(d / 'summary.json')
                need(int(row['cost']) == kappa and row['map_id'] == rec['map_id'], 'Wrong cached kappa/map')
                own.append(row)
    own.sort(key=lambda r: r['map_id'])
    if len(own) == 180:
        check_rows(own, (kappa,), strict_new=True)
    status = dict(version=VERSION, kappa=kappa, status='COMPLETE' if len(own) == 180 else 'PARTIAL',
                  evaluated_maps=len(own), validation_maps=180, test_rollouts=0, osm_rollouts=0,
                  checkpoint_step=job['checkpoint_step'] if job else None,
                  checkpoint_sha256=job['checkpoint_sha256'] if job else None, identity=ident,
                  experiment_origin=ORIGIN, existing_costs_retrained=False)
    E.csvwrite(out / 'report/per_map.csv', own, fields=None if own else ['cost', 'map_id'])
    E.write(out / 'report/audit.json', status)
    if len(own) == 180:
        tab, members = table(own, (kappa,))
        E.csvwrite(out / 'report/own_success_summary.csv', tab)
        E.write(out / 'report/complete.json', dict(version=VERSION, kappa=kappa, identity=ident,
                files={n: E.sha(out / 'report' / n) for n in ('per_map.csv', 'audit.json', 'own_success_summary.csv')}))
    print(f'[REPORT COST{kappa}] {status["status"]}; {len(own)}/180 Raw/Refined maps.', flush=True)
    return status


def refinement_worker_init(job: dict) -> None:
    # Spawned CPU workers import the original runner afresh; configure BEFORE
    # resolving the task function and loading the cost-specific native module.
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    configure(int(job['kappa']))
    F.JOB = job
    F.RT = E.load_native(job)
    E.MODEL = None
