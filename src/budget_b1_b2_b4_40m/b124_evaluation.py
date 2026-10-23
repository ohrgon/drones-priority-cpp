"""Deterministic native-core evaluation with map/initial-condition cache checks.
CPU subprocesses only; no modification of a training worker's RNG or model.
"""
from __future__ import annotations
import gc, hashlib, math
from pathlib import Path
import b124_engine as E
import b124_native_support as U


def checkpoint_record(plan, B, stage, path):
    wanted = Path(path).resolve()
    _, run = E.segment_paths(plan, B, stage)
    pending = run / '.validation_pending' / 'pending.json'
    candidates = [U.read(pending)] if pending.is_file() else []
    candidates += E.milestones(plan, B, stage)
    for cp in candidates:
        if Path(cp['path']).resolve() == wanted:
            U.require(U.sha(wanted) == cp['sha256'], 'Checkpoint bytes changed')
            return cp
    raise RuntimeError('Unregistered checkpoint refused: ' + str(path))


def validate_row(row, version, B):
    U.require(row.get('nrg_version') == version, 'Wrong evaluated native core')
    U.require(U.near(row.get('c3nt_cost_multiplier'), 11) and
              U.near(row.get('c3nt_normalization_step_factor'), 5), 'Wrong COST11 normalization')
    U.require(int(row['max_steps']) == math.ceil(5 * int(row['active_cells'])), 'Wrong execution horizon')
    U.require(int(row['invalid_action_count']) == 0, 'Invalid movement in evaluated path')
    error = U.number(row.get('nrg_return_identity_error'))
    U.require(error is not None and abs(error) < 1e-6, 'Native reward identity failed')
    t, lw = U.number(row.get(U.T)), U.number(row.get(U.LW))
    U.require(t is not None and t >= 0, 'Invalid mission duration')
    if U.truth(row['is_success']):
        U.require(U.near(row['coverage_pct'], 100) and lw is not None and 0 <= lw <= t + 1e-7,
                  'Completed-path metric mismatch')
    else:
        U.require(row.get('termination_reason') == 'external_step_cap' and
                  row.get('c3nt_stop_kind') == 'safety_cap' and U.truth(row['c3nt_truncated']) and
                  int(row['steps']) == int(row['max_steps']) and lw is None,
                  'Failed path must retain undefined Lw and native timeout flags')
    counts = [int(row[f'allowed_actions_k{k}_count']) for k in range(1, 5)]
    decisions = int(row['decision_steps'])
    U.require(sum(counts) == decisions and sum((k+1)*v for k, v in enumerate(counts)) == int(row['allowed_actions_sum']),
              'Action-count accounting failed')
    U.require(int(row['effective_B']) == B, 'Evaluation used incorrect B')


def evaluate_one(rt, B, model, rec, array, ic):
    import numpy as np
    import torch
    core = rt.trainer.make_core(rt.cfg)
    obs, _ = core.reset(array, start=tuple(ic['start']), heading=ic['heading'],
                        seed=42, map_name=rec.map_id)
    ent, norm, counts = [], [], [0, 0, 0, 0]
    strict_steps = steps = budget_open = 0
    if core.complete_at_reset:
        core.step(0)
    else:
        while not core._done:
            mask = np.asarray(core.action_masks(), dtype=bool)
            U.require(mask.shape == (4,) and mask.any(), 'Empty action mask')
            strict, _ = core.strict_mask_details_no_cache()
            with torch.no_grad():
                ob, _ = model.policy.obs_to_tensor(obs)
                distribution = model.policy.get_distribution(ob, action_masks=mask.reshape(1, -1))
                probabilities = distribution.distribution.probs[0].detach().cpu().numpy().astype(float)
            U.require(np.isfinite(probabilities).all() and abs(probabilities.sum()-1) < 1e-5 and
                      np.all(probabilities[~mask] < 1e-6), 'Invalid masked probabilities')
            action = int(np.argmax(probabilities))
            K = int(mask.sum()); counts[K-1] += 1
            positive = probabilities[probabilities > 0]
            entropy = -float(np.sum(positive * np.log(positive)))
            ent.append(entropy)
            if K > 1:
                norm.append(entropy / math.log(K))
            strict_steps += int(strict[action])
            budget_open += int(core.detour_budget > 0)
            obs, _, terminated, truncated, _ = E.audit_step(core, obs, action)
            steps += 1
            U.require(steps <= int(core.max_steps)+1, 'Safety cap failed')
    core.assert_consistent()
    row = core.info(final=True)
    row.update(map_id=rec.map_id, map_sha256=rec.sha256, source=rec.source, split=rec.split,
               pattern=rec.pattern, tier=rec.tier, beta=.5, effective_B=B, deterministic=True,
               initial_pair_id=ic['pair_id'], start_r=ic['start'][0], start_c=ic['start'][1],
               start_heading=ic['heading'], decision_steps=steps,
               masked_entropy_mean=U.mean(ent), multiaction_normalized_entropy_mean=U.mean(norm),
               allowed_actions_sum=sum((k+1)*v for k,v in enumerate(counts)),
               allowed_actions_mean=sum((k+1)*v for k,v in enumerate(counts))/steps if steps else None,
               chosen_strict_ratio=strict_steps/steps if steps else None,
               budget_open_ratio=budget_open/steps if steps else None,
               raw_only=True, connection_refinement=False)
    for k, v in enumerate(counts, 1):
        row[f'allowed_actions_k{k}_count'] = v
    row = U.safe(row)
    validate_row(row, rt.core.VERSION, B)
    return row


def evaluate(plan, B, stage, path, scope):
    U.require(scope in ('validation', 'test'), 'Unknown evaluation scope')
    E.verify_plan(plan)
    cp = checkpoint_record(plan, B, stage, path)
    if scope == 'test':
        U.require(cp['nominal_m'] == 40 and stage == 2, 'Test only uses fixed nominal40M model')
    base = E.eval_base(plan, B, cp, scope)
    with U.lock(base / '.eval.lock'):
        if E.eval_done(plan, B, cp, scope):
            print(f'[EVAL SKIP] B={B} {scope} {cp["nominal_m"]}M', flush=True)
            return
        rt = E.runtime(plan, B)
        import numpy as np
        from sb3_contrib import MaskablePPO
        records = rt.val if scope == 'validation' else rt.tests
        target = 180 if scope == 'validation' else 1220
        U.require(len(records) == target, 'Incorrect evaluation cardinality')
        parameters = dict(version=E.VERSION, B=B, beta=.5, checkpoint_sha256=cp['sha256'],
                          checkpoint_step=int(cp['num_timesteps']), environment=U.safe(vars(rt.cfg)),
                          runtime_hashes=plan['runtime']['relaxed']['hashes'], scope=scope,
                          initial_seed=plan['initial_seed'], evaluation_seed=42, deterministic=True,
                          records=[(r.map_id, r.sha256) for r in records])
        identity = U.digest(parameters)
        U.freeze(base/'identity.json', dict(sha256=identity, parameters=parameters))
        # Equality checks tie every new result to the documented B0/BCD/WLG input.
        refname = 'reference_validation_baselines.csv' if scope == 'validation' else 'reference_test_baselines.csv'
        refs = {r['map_id']: r for r in U.rows(Path(plan['out'])/refname) if r['method'] == 'bcd'}
        U.require(set(refs) == {r.map_id for r in records}, 'Baseline/map partition mismatch')
        rows = []; model = None
        parameter_digest = cp.get('parameter_sha256')
        try:
            for i, rec in enumerate(records, 1):
                array = np.load(rec.path, allow_pickle=False)
                U.require(rt.utils.array_hash(array) == rec.sha256, 'Map content changed: ' + rec.map_id)
                ic = U.initial_condition(rt, rec, array)
                old = refs[rec.map_id]
                U.require(rec.sha256 == old['map_sha256'] and ic['pair_id'] == old['initial_pair_id'],
                          'New baseline comparison would use a different map/initial condition: ' + rec.map_id)
                destination = base/'maps'/(hashlib.sha256(rec.map_id.encode()).hexdigest()+'.json')
                if destination.is_file():
                    cache = U.read(destination); row = cache['row']
                    U.require(cache['identity'] == identity and cache['row_sha256'] == U.digest(row),
                              'Changed evaluation cache')
                    U.require(row['map_id'] == rec.map_id and row['map_sha256'] == rec.sha256 and
                              row['initial_pair_id'] == ic['pair_id'], 'Cache/input mismatch')
                    validate_row(row, rt.core.VERSION, B)
                else:
                    if model is None:
                        U.require(U.sha(cp['path']) == cp['sha256'], 'Checkpoint changed before inference')
                        model = MaskablePPO.load(cp['path'], device='cpu')
                        U.require(model.num_timesteps == int(cp['num_timesteps']), 'Loaded step mismatch')
                        observed_digest = U.tensor_digest(model.get_parameters())
                        U.require(parameter_digest is None or parameter_digest == observed_digest, 'Snapshot parameter digest changed')
                        parameter_digest = observed_digest
                        rt.features.audit_cnn3_model(model); rt.core.audit_model(model)
                        model.policy.set_training_mode(False)
                    row = evaluate_one(rt, B, model, rec, array, ic)
                    row.update(method=f'ppo_B{B}_beta05_40M_schedule', model=f'B{B}',
                               nominal_m=cp['nominal_m'], checkpoint_step=int(cp['num_timesteps']),
                               checkpoint_sha256=cp['sha256'])
                    U.write(destination, dict(identity=identity, row_sha256=U.digest(row), row=row))
                rows.append(row)
                if i == 1 or i % 25 == 0 or i == target:
                    U.csv_write(base/'per_map.csv', rows)
                    print(f'[{scope.upper()}] B={B} {cp["nominal_m"]}M {i}/{target} '
                          f'success={sum(U.truth(r["is_success"]) for r in rows)}', flush=True)
        finally:
            del model; gc.collect()
        if parameter_digest is None:
            model = MaskablePPO.load(cp['path'], device='cpu')
            parameter_digest = U.tensor_digest(model.get_parameters())
            del model; gc.collect()
        from b124_report import summarize
        U.write(base/'summary.json', dict(B=B, scope=scope, nominal_m=cp['nominal_m'], **summarize(rows, target)))
        U.write(base/'complete.json', dict(checkpoint_sha256=cp['sha256'], checkpoint_step=cp['num_timesteps'],
                  nominal_m=cp['nominal_m'], scope=scope, identity=identity, n_maps=target,
                  parameter_sha256=parameter_digest,
                  files={name: U.sha(base/name) for name in ('summary.json', 'per_map.csv', 'identity.json')}))
    from b124_report import build_report
    build_report(plan, bundle=False)
