"""Common audited fixed-heading connection refinement, using the uploaded A* unchanged.

NumPy and standard library only. B changes admissibility, not the algorithm.
The stored first-visit order and first-arrival heading stay fixed. Every output
connector uses no more steps than its input connector. This is the constructive
(nonexpanding-step) refinement, NOT an unconstrained/global ordering optimum.
Failed episodes are preserved byte-for-byte at the action-array level.
"""
from __future__ import annotations
import importlib
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
from ch5_vendor import test8_original as V
from ch5_vendor.segment_graph_original import SegmentGraph

NATIVE_FILES = ('coverage_core.py', 'core_ch4.py', 'core_ch4_bounded_b6.py',
                'full_v2_utils.py', 'c3nt_cost_sweep_generated_core.py')
T, LW = V.T, V.LW
ATOL = 2e-6
META = ('method', 'label', 'family', 'beta', 'effective_B', 'nominal_m',
        'checkpoint_step', 'checkpoint_sha256', 'map_id', 'map_sha256', 'source',
        'split', 'pattern', 'tier', 'initial_pair_id', 'start_r', 'start_c',
        'start_heading', 'evaluation_seed', 'initial_seed', 'native_reward_beta')


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def near(a, b, label, atol=ATOL):
    aa, bb = V.num(a), V.num(b)
    if aa is None and bb is None:
        return
    need(aa is not None and bb is not None and math.isclose(aa, bb, rel_tol=1e-9, abs_tol=atol),
         f'{label}: {a!r} != {b!r}')


def load_native(job):
    """Load only original NumPy environment modules, never trainer or weights."""
    snapshot = Path(job['snapshot']).resolve()
    for name in NATIVE_FILES:
        p = snapshot / name
        need(p.is_file() and V.sha(p) == job['sources'][name]['sha256'],
             'Native snapshot missing/changed: ' + str(p))
    # Workers normally handle one method. Explicit unloading also makes isolated
    # tests safe without letting a Strict module leak into a Relaxed case.
    for name in (Path(f).stem for f in NATIVE_FILES):
        sys.modules.pop(name, None)
    sys.path.insert(0, str(snapshot)); importlib.invalidate_caches()
    modules = {}
    try:
        for f in NATIVE_FILES:
            name = Path(f).stem
            m = importlib.import_module(name)
            need(Path(m.__file__).resolve().parent == snapshot, 'Wrong module: ' + name)
            modules[name] = m
    finally:
        sys.path.remove(str(snapshot))
    adapter = modules['c3nt_cost_sweep_generated_core']
    bounded = modules['core_ch4_bounded_b6']
    need(adapter.VERSION == job['native_version'], 'Native COST11 version differs')
    near(adapter.COST_MULTIPLIER, 11, 'COST11')
    cfg = bounded.ExperimentEnvConfig(**job['spec']['environment'])
    factory = adapter.wrap_factory(bounded.make_core)
    need(tuple(modules['coverage_core'].MOVES) == V.MOVE_TUPLES, 'Action convention differs')
    return SimpleNamespace(cfg=cfg, factory=factory, core=adapter, bounded=bounded,
                           initial=modules['core_ch4'], utils=modules['full_v2_utils'],
                           modules=modules, moves=V.MOVE_TUPLES)


def events(priority, positions, actions, start_heading):
    """(step,row,col,arrival_heading) including a visited start at time zero."""
    covered = np.zeros(np.asarray(priority).shape, bool)
    out = []
    for step, pos in enumerate(positions):
        q = tuple(map(int, pos))
        if priority[q] > 0 and not covered[q]:
            covered[q] = True
            out.append((step, q[0], q[1], int(start_heading if step == 0 else actions[step-1])))
    return out


def replay(rt, row, trace, *, check_stored=True):
    """Native replay of every primitive move, with full mask/budget/metric audit."""
    priority = np.asarray(trace['priority_map'])
    need(rt.utils.array_hash(priority) == row['map_sha256'], 'Trace map hash mismatch')
    rawact = np.asarray(trace['actions'])
    need(rawact.ndim == 1 and np.isfinite(rawact).all() and
         np.equal(rawact, np.rint(rawact)).all() and np.all((rawact >= 0) & (rawact < 4)),
         'Invalid primitive action array')
    actions = rawact.astype(np.int8)
    n = len(actions)
    start = (int(row['start_r']), int(row['start_c']))
    core = rt.factory(rt.cfg)
    core.reset(priority, start=start, heading=int(row['start_heading']),
               seed=V.EVAL_SEED, map_name=row['map_id'])
    need(n <= core.max_steps, 'Trace exceeds original horizon')
    positions = [start]; times = [0.0]; masks = []; budgets = []; headings = []
    boundaries = []; stricts = []; phases = []; destination = []; ratios = []
    hist = np.zeros(5, np.int64); bhist = hist.copy(); shist = hist.copy()
    phase_steps = {'direct': 0, 'progress': 0}; phase_sum = dict(phase_steps)
    chosen_strict = 0; budget_open = 0
    if core.complete_at_reset:
        need(n == 0, 'Singleton has movement after completion')
        core.step(0)  # native zero-movement completion acknowledgment
    for k, action in enumerate(actions):
        action = int(action)
        need(not core._done, f'Trace continues past native termination at {k}')
        mask = np.asarray(core.action_masks(), bool).copy()
        boundary = np.asarray(core.feasible_mask(), bool).copy()
        strict, phase = rt.bounded.ExperimentCore.strict_mask_details_no_cache(core)
        strict = np.asarray(strict, bool)
        b = int(core.detour_budget)
        need(mask.shape == (4,) and mask[action], f'Masked movement at step {k}')
        B = int(row['effective_B'])
        need(np.array_equal(mask, strict if B == 0 or b == 0 else boundary), 'B/mask mismatch')
        need((B == 0 and b == 0) or (B == 6 and 0 <= b <= 6), 'Budget mismatch')
        if check_stored:
            for key, value in [('positions', core.drone_pos), ('pre_action_masks', mask),
                               ('budget_before', b), ('heading_before', core.heading),
                               ('strict_masks', strict), ('boundary_masks', boundary)]:
                if key in trace:
                    need(np.array_equal(np.asarray(trace[key])[k], value), f'{key} mismatch at {k}')
            if 'time_sec' in trace:
                near(trace['time_sec'][k], core.elapsed_time_sec, f'pre-time {k}')
        masks.append(mask); boundaries.append(boundary); stricts.append(strict)
        budgets.append(b); headings.append(int(core.heading)); phases.append(phase)
        ratios.append(1 - core.remaining_priority_mass / core.total_priority_mass)
        r, c = core.drone_pos; dr, dc = rt.moves[action]
        destination.append(int(priority[r+dr, c+dc]))
        hist[mask.sum()] += 1; bhist[boundary.sum()] += 1; shist[strict.sum()] += 1
        chosen_strict += int(strict[action]); budget_open += int(b > 0)
        if phase in phase_steps:
            phase_steps[phase] += 1; phase_sum[phase] += int(mask.sum())
        _, _, term, trunc, _ = core.step(action)
        need(not (term and trunc), 'Native terminated and truncated simultaneously')
        need(bool(term or trunc) == (k == n-1), 'Stored episode length/native stopping disagree')
        positions.append(tuple(core.drone_pos)); times.append(float(core.elapsed_time_sec))
    need(core._done or core.complete_at_reset, 'Incomplete/cut-off cache: no native termination')
    core.assert_consistent()
    metric = V.safe(core.info(final=True))
    need(int(metric['steps']) == n and int(metric['invalid_action_count']) == 0, 'Move counter mismatch')
    need(np.array_equal(core.mask_counts, hist), 'Native mask cardinality mismatch')
    if V.truth(metric['is_success']):
        need(V.num(metric[LW]) is not None, 'Successful path has no Lw')
    else:
        need(V.num(metric[LW]) is None, 'Failure must not have complete-trajectory Lw')
    arr = dict(actions=actions, positions=np.asarray(positions, np.int16).reshape(-1, 2),
               time_sec=np.asarray(times, np.float64),
               pre_action_masks=np.asarray(masks, bool).reshape(-1, 4),
               boundary_masks=np.asarray(boundaries, bool).reshape(-1, 4),
               strict_masks=np.asarray(stricts, bool).reshape(-1, 4),
               budget_before=np.asarray(budgets, np.int16),
               heading_before=np.asarray(headings, np.int8),
               strict_phase=np.asarray(phases, dtype='<U8'),
               destination_priority=np.asarray(destination, np.uint8),
               covered_priority_ratio_before=np.asarray(ratios, float),
               priority_map=np.asarray(priority).copy(),
               first_visit_time_sec=core.first_visit_time.copy(),
               first_visit_step=core.first_visit_step.copy())
    if check_stored:
        for key in ('positions', 'first_visit_step'):
            if key in trace:
                need(np.array_equal(arr[key], trace[key]), 'Stored ' + key + ' differs')
        for key in ('time_sec', 'first_visit_time_sec'):
            if key in trace:
                need(arr[key].shape == np.asarray(trace[key]).shape and
                     np.allclose(arr[key], trace[key], rtol=1e-9, atol=ATOL, equal_nan=True),
                     'Stored ' + key + ' differs')
        for key in (T, LW, 'steps', 'max_steps', 'active_cells', 'coverage_pct',
                    'turn_count', 'revisit_steps', 'zero_transit_steps', 'path_length_m',
                    'time_straight_sec', 'time_turn_sec'):
            near(metric.get(key), row.get(key), 'Raw/native ' + key)
        need(V.truth(metric['is_success']) == V.truth(row['is_success']), 'Raw/native success differs')
    metric.update({k: row.get(k) for k in META})
    metric.update(decision_steps=n, chosen_strict_actions=chosen_strict,
                  chosen_strict_ratio=chosen_strict/n if n else None,
                  budget_open_steps=budget_open, native_config_budget_capacity=rt.cfg.detour_budget_max,
                  action_set_kind='native_progress_constraint_mask',
                  masked_entropy_mean=None, multiaction_normalized_entropy_mean=None,
                  policy_compute_time_sec=None, behavior_probability_scope='not_inferred_in_path_replay')
    for prefix, hh in [('allowed_actions', hist), ('boundary_actions', bhist),
                       ('strict_counterfactual_actions', shist)]:
        metric.update(V.hist_stats(hh, prefix))
    for phase in phase_steps:
        metric['strict_phase_' + phase + '_steps'] = phase_steps[phase]
        metric['strict_phase_' + phase + '_allowed_sum'] = phase_sum[phase]
    return V.safe(metric), arr


def common_refine(rt, raw, trace):
    """One engine/procedure for all B0/B6 policies and controls."""
    native_raw, normalized = replay(rt, raw, trace)
    priority = normalized['priority_map']; actions = normalized['actions']
    positions = normalized['positions']; B = int(raw['effective_B'])
    need(B in (0, 6), 'This experiment is B0/B6 only; B124 training is separate')
    raw_events = events(priority, positions, actions, int(raw['start_heading']))
    audit = dict(native_raw_replay=True, effective_B=B,
                 solver='uploaded_SegmentGraph_shortest_Astar',
                 objective='T_only_fixed_order_fixed_arrival_heading',
                 step_constraint='each_connector_steps_le_raw_connector_steps',
                 all_segments_optimal_under_stated_constraint=None,
                 numerical_optimality_tolerance_sec=1e-8,
                 failure_repaired=False, raw_steps=len(actions),
                 first_visit_sequence_sha256=V.digest([e[1:] for e in raw_events]))
    segment_rows = []
    if not V.truth(raw['is_success']):
        # No observed complete mission: don't optimize the prefix or remove tail.
        refined_actions = actions.copy()
        metric, arr = native_raw, normalized
        audit.update(status='failed_episode_preserved_no_search', n_segments=0,
                     native_refined_replay='same_verified_raw_actions', expanded_states=0)
    else:
        rr, cc = np.where(priority > 0)
        bbox = np.zeros(priority.shape, bool)
        bbox[rr.min():rr.max()+1, cc.min():cc.max()+1] = True
        covered = np.zeros(priority.shape, bool)
        covered[tuple(positions[0])] = priority[tuple(positions[0])] > 0
        ts, tt = float(raw['time_straight_sec']), float(raw['time_turn_sec'])
        total_mass = float(priority.sum()); ref_actions = []; s = 0; expanded = 0
        for e, r, c, arrival in raw_events:
            if e == 0:
                continue
            original = tuple(map(int, actions[s:e]))
            need(original, 'Empty connector')
            heading = int(raw['start_heading']) if s == 0 else int(actions[s-1])
            cost = math.fsum(ts if a == h else tt for a, h in zip(original, (heading,)+original[:-1]))
            ratio = float(priority[(priority > 0) & ~covered].sum()) / total_mass
            # Native replay already proved admissibility. A one-edge connection
            # with a fixed arrival heading cannot use fewer edges or turns.
            proof = 'one_edge_exact' if len(original) == 1 else 'resource_constrained_Astar_exact'
            if len(original) == 1:
                chosen = original; seconds = cost; ex = 0
            else:
                graph = SegmentGraph((priority > 0) & ~covered, bbox, ts, tt, bmax=B)
                near(graph.check_path(positions[s], heading, B, (r, c), arrival, original), cost, 'Input connector')
                opt = graph.shortest(positions[s], heading, B, (r, c), arrival,
                                     cost + 1e-9, step_limit=len(original))
                near(graph.check_path(positions[s], heading, B, (r, c), arrival, opt.actions),
                     opt.seconds, 'Optimized connector')
                need(opt.seconds <= cost + ATOL and len(opt.actions) <= len(original), 'Connector expanded/increased')
                # Keep the original realization on numerical/time ties; no
                # secondary reordering or heading choice is introduced.
                if cost - opt.seconds <= 1e-8:
                    chosen = original; seconds = cost
                else:
                    chosen = opt.actions; seconds = opt.seconds
                ex = opt.expanded; expanded += ex
            ref_actions.extend(chosen)
            segment_rows.append(dict(segment=len(segment_rows), raw_start_step=s, raw_end_step=e,
                                     target_r=r, target_c=c, start_heading=heading,
                                     arrival_heading=arrival, B=B, raw_steps=e-s,
                                     refined_steps=len(chosen), raw_T=cost, refined_T=seconds,
                                     saved_T=cost-seconds, remaining_priority_ratio=ratio,
                                     saved_Lw=ratio*(cost-seconds), proof=proof, expanded=ex,
                                     changed=tuple(chosen) != original))
            covered[r, c] = True; s = e
        need(s == len(actions), 'Successful path has an unobserved tail after completion')
        refined_actions = np.asarray(ref_actions, np.int8)
        metric, arr = replay(rt, raw, dict(priority_map=priority, actions=refined_actions), check_stored=False)
        audit.update(status='common_refinement_applied', n_segments=len(segment_rows),
                     native_refined_replay=True, expanded_states=expanded,
                     all_segments_optimal_under_stated_constraint=True)
    revents = events(priority, arr['positions'], arr['actions'], int(raw['start_heading']))
    need([e[1:] for e in raw_events] == [e[1:] for e in revents], 'First-visit order/heading changed')
    need(V.truth(metric['is_success']) == V.truth(raw['is_success']), 'Success status changed')
    need(len(refined_actions) <= len(actions), 'Refinement increased total step count')
    need(float(metric[T]) <= float(raw[T]) + ATOL, 'Refinement increased mission time')
    oldfv, newfv = normalized['first_visit_time_sec'], arr['first_visit_time_sec']
    need(np.array_equal(np.isfinite(oldfv), np.isfinite(newfv)), 'Visited set changed')
    reached = np.isfinite(oldfv)
    need(np.all(newfv[reached] <= oldfv[reached] + ATOL), 'First visit became later')
    if V.truth(raw['is_success']):
        near(math.fsum(s['saved_T'] for s in segment_rows), float(raw[T])-float(metric[T]), 'Total segment saving')
        near(math.fsum(s['saved_Lw'] for s in segment_rows), float(raw[LW])-float(metric[LW]), 'Weighted segment saving')
    else:
        need(np.array_equal(arr['actions'], actions), 'Failed trajectory was modified')
    changed = not np.array_equal(actions, arr['actions'])
    audit.update(order_preserved=True, arrival_heading_preserved=True, visited_set_preserved=True,
                 success_preserved=True, horizon_preserved=True, no_time_increase=True,
                 refined_steps=len(refined_actions), path_changed=changed,
                 changed_segments=sum(s['changed'] for s in segment_rows),
                 refined_first_visit_sequence_sha256=V.digest([e[1:] for e in revents]))
    metric.update(stage='refined_stage2_common_fixed_heading_T',
                  refinement_status=audit['status'], path_changed=changed,
                  changed_segments=audit['changed_segments'],
                  raw_mission_time_sec=raw[T], raw_Lw_sec=raw.get(LW),
                  saved_T_sec=float(raw[T])-float(metric[T]),
                  saved_Lw_sec=(float(raw[LW])-float(metric[LW])) if V.truth(raw['is_success']) else None,
                  native_replay_verified=True, refinement_solver=audit['solver'])
    return V.safe(metric), arr, V.safe(audit), segment_rows
