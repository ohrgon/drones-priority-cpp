"""Strict COST11 adapter. Physics and native progress mask unchanged. Only mask/budget audit guards and identity differ from the original COST11 adapter. Legacy cost5_* log prefixes are retained."""
from __future__ import annotations
import math
from typing import Any
import numpy as np
VERSION = 'cnn3_noclock_cost11_strict_progress_v1_20260929'
NORMALIZATION_STEP_FACTOR = 5.0
COST_MULTIPLIER = 11.0
OBS_KEYS = {'map', 'heading', 'detour_budget'}
REWARD_KEYS = ('coverage', 'latency', 'mission_time', 'terminal', 'invalid')
MOVES = ((-1, 0), (0, 1), (1, 0), (0, -1))

def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)

def reward_parts(*, new_count: int, n_active: int, mass_before: float, total_mass: float, dt: float, time_cap: float, beta: float, completed: bool, exhausted: bool) -> dict[str, float]:
    """One transition; completion has precedence over exhaustion at H."""
    require(new_count in (0, 1), 'NRG1 requires 0 or 1 newly covered cell per action.')
    require(n_active > 0 and total_mass > 0 and (time_cap > 0), 'Invalid normalizers.')
    require(0 <= beta <= 1 and dt >= 0, 'Invalid beta/dt.')
    require(-1e-08 <= mass_before <= total_mass + 1e-08, 'Invalid remaining mass.')
    return {'coverage': float(new_count) / n_active, 'latency': -COST_MULTIPLIER * beta * mass_before / total_mass * dt / time_cap, 'mission_time': -COST_MULTIPLIER * (1 - beta) * dt / time_cap, 'terminal': 2.0 if completed else 0.0, 'invalid': 0.0}

def reference_masks(core: Any) -> tuple[np.ndarray, np.ndarray]:
    """Read-only reference for auditing, NOT a replacement action mask."""
    (r, c) = core.drone_pos
    boundary = np.asarray([core._in_bbox(r + dr, c + dc) for (dr, dc) in MOVES], bool)
    direct = np.zeros(4, bool)
    for (a, (dr, dc)) in enumerate(MOVES):
        if boundary[a]:
            direct[a] = bool(core.active_mask[r + dr, c + dc] and (not core.covered_map[r + dr, c + dc]))
    if direct.any():
        return (boundary, direct)
    d = core.distance_field()
    strict = np.zeros(4, bool)
    for (a, (dr, dc)) in enumerate(MOVES):
        if boundary[a]:
            strict[a] = int(d[r + dr, c + dc]) == int(d[r, c]) - 1
    require(strict.any(), 'Nonterminal strict mask empty. No fallback was applied.')
    return (boundary, strict)

def observed_budget(obs: dict, bmax: int) -> int:
    require('detour_budget' in obs, 'Original B6 observation has no detour_budget.')
    a = np.asarray(obs['detour_budget'])
    require(a.shape == (1,), f'Unexpected detour_budget shape: {a.shape}')
    b = float(a[0]) * bmax
    require(np.isfinite(b) and -1e-05 <= b <= bmax + 1e-05, 'Invalid observed budget.')
    require(abs(b - round(b)) < 1e-05, 'Budget observation is not normalized b/B.')
    return int(round(b))

class C3NoClockMixin:
    """Mix into the original core configured with Strict Progress at runtime.

    Original code owns physics, masks, counters, first-visit times and stop cap.
    This adapter removes the deadline input and uses a fixed normalization scale.
    Native _done means "stop collection"; public terminated/truncated distinguish
    actual mission completion from external interruption. All inputs to the
    critic at truncation are the unreset last-state observation.
    """

    def reset(self, *args, **kwargs):
        self._nrg_ready = False
        self._nrg_audited_steps = 0
        self._nrg_detours = 0
        self._nrg_strict = 0
        self._nrg_open_steps = 0
        self._nrg_open_detours = 0
        self._nrg_straight_detours = 0
        self._nrg_pre_budget_hist = np.zeros(7, np.int64)
        self._nrg_last = None
        self._nrg_ack_reward = 0.0
        for name in ('coverage_reward_scale', 'objective_cost_scale', 'success_terminal_reward', 'failure_terminal_penalty'):
            require(getattr(self.config, name) == 0.0, f'Legacy {name} must be 0. NRG1 replaces, not stacks, rewards.')
        require(getattr(self.config, 'mask_mode', None) == 'progress', 'Strict requires mask_mode=progress; Relaxed is rejected.')
        require(getattr(self.config, 'detour_budget_max', None) == 6, 'B must remain 6.')
        (obs, _) = super().reset(*args, **kwargs)
        require(self.time_turn_sec >= self.time_straight_sec > 0, 'Invalid travel-time model.')
        self._nrg_n0 = int(self.covered_cells_count)
        self._normalization_steps = int(math.ceil(NORMALIZATION_STEP_FACTOR * self.target_cells_count))
        self._nrg_cap = float(self._normalization_steps * self.time_turn_sec)
        self._nrg_ready = True
        require(set(obs) == OBS_KEYS, f'Unexpected original observation keys: {set(obs)}')
        require(observed_budget(obs, 6) == 0, 'Strict budget observation must be zero at reset.')
        return (self.observation(), self.info(final=self.complete_at_reset))

    def observation(self):
        obs = dict(super().observation())
        require(set(obs) == OBS_KEYS, f'Underlying bounded core must expose map/heading/budget only: {set(obs)}')
        return obs

    def immediate_reward(self, action):
        if self.complete_at_reset:
            return 2.0 if not self._done else 0.0
        require(not self._done, 'Episode already ended.')
        require(isinstance(action, (int, np.integer)) and (not isinstance(action, bool)) and (0 <= int(action) < 4), 'Invalid action ID.')
        action = int(action)
        require(bool(self.action_masks()[action]), 'Action is masked out.')
        (r, c) = self.drone_pos
        (dr, dc) = MOVES[action]
        nxt = (r + dr, c + dc)
        require(self._in_bbox(*nxt), 'Invalid move in Strict Progress.')
        new = int(self.active_mask[nxt] and (not self.covered_map[nxt]))
        dt = self.time_straight_sec if action == self.heading else self.time_turn_sec
        p = reward_parts(new_count=new, n_active=self.target_cells_count, mass_before=self.remaining_priority_mass, total_mass=self.total_priority_mass, dt=dt, time_cap=self._nrg_cap, beta=self.config.tradeoff_beta, completed=self.covered_cells_count + new == self.target_cells_count, exhausted=self.steps + 1 >= self.max_steps)
        return float(sum(p.values()))

    def step(self, action):
        require(getattr(self, '_nrg_ready', False) and (not self._done), 'Reset required.')
        require(isinstance(action, (int, np.integer)) and (not isinstance(action, bool)) and (0 <= int(action) < 4), 'Invalid action ID.')
        action = int(action)
        singleton = bool(self.complete_at_reset)
        pre_obs = self.observation()
        budget_before = observed_budget(pre_obs, 6)
        old_total = float(self.total_return)
        old_sums = dict(self.reward_sums)
        require(set(old_sums) == set(REWARD_KEYS), f'Unexpected reward component keys: {set(old_sums)}')
        n_before = int(self.covered_cells_count)
        step_before = int(self.steps)
        mass = float(self.remaining_priority_mass)
        time_before = float(self.elapsed_time_sec)
        heading_before = int(self.heading)
        position_before = tuple(self.drone_pos)
        if not singleton:
            (boundary, strict) = reference_masks(self)
            actual_mask = np.asarray(self.action_masks(), bool)
            require(budget_before == 0, 'Strict budget cannot become positive.')
            expected_mask = strict
            require(np.array_equal(actual_mask, expected_mask), 'Strict action mask differs from the original progress definition.')
            require(bool(actual_mask[action]), 'Attempt to execute a masked action.')
            is_strict = bool(strict[action])
        else:
            is_strict = True
        (obs, legacy_reward, terminated, truncated, info) = super().step(action)
        require(not truncated, 'Unexpected native truncation: review the frozen bounded core interface.')
        require(abs(float(legacy_reward)) < 1e-09, 'Legacy engine emitted a nonzero reward despite neutral coefficients.')
        dn = int(self.covered_cells_count) - n_before
        dt = float(self.elapsed_time_sec) - time_before
        completed = self.covered_cells_count == self.target_cells_count
        exhausted = self.steps >= self.max_steps and (not completed)
        require(bool(terminated) == bool(completed or exhausted), 'Unexpected episode termination.')
        require(self.invalid_action_count == 0, 'Strict Progress executed an invalid action.')
        if not singleton:
            require(self.steps == step_before + 1, 'An action must consume exactly one step.')
            (dr, dc) = MOVES[action]
            require(tuple(self.drone_pos) == (position_before[0] + dr, position_before[1] + dc), 'U/R/D/L action convention differs from the source protocol.')
            expected_dt = self.time_straight_sec if action == heading_before else self.time_turn_sec
            require(math.isclose(dt, expected_dt, rel_tol=1e-09, abs_tol=1e-08), 'Travel time changed.')
            require(is_strict, 'Non-Strict action executed; no fallback is allowed.')
            expected_budget = 0
            require(observed_budget(self.observation(), 6) == expected_budget, 'Strict budget must remain zero after every step.')
            self._nrg_detours += int(not is_strict)
            self._nrg_strict += int(is_strict)
            self._nrg_open_steps += int(budget_before > 0)
            self._nrg_open_detours += int(budget_before > 0 and (not is_strict))
            self._nrg_straight_detours += int(not is_strict and action == heading_before)
            self._nrg_pre_budget_hist[budget_before] += 1
        p = reward_parts(new_count=dn, n_active=self.target_cells_count, mass_before=mass, total_mass=self.total_priority_mass, dt=dt, time_cap=self._nrg_cap, beta=self.config.tradeoff_beta, completed=completed, exhausted=exhausted)
        reward = float(sum(p.values()))
        self.reward_sums = {key: float(old_sums[key]) + p[key] for key in REWARD_KEYS}
        self.total_return = old_total + reward
        self._nrg_audited_steps += int(not singleton)
        if singleton:
            self._nrg_ack_reward = reward
        self._nrg_last = dict(budget_before=budget_before, budget_after=observed_budget(self.observation(), 6), selected_is_strict=int(is_strict), selected_is_detour=int(not is_strict and (not singleton)), is_straight=int(action == heading_before and (not singleton)))
        public_terminated = bool(completed)
        public_truncated = bool(exhausted and (not completed))
        clean = {k: v for (k, v) in info.items() if not k.startswith('step_reward_')}
        clean.update(self.info(final=bool(terminated)))
        clean.update({'step_reward_' + key: value for (key, value) in p.items()})
        clean.update({'nrg_step_' + key: value for (key, value) in self._nrg_last.items()})
        clean['nrg_reward_replaced'] = True
        clean.update(c3nt_terminated=public_terminated, c3nt_truncated=public_truncated, c3nt_bootstrap_required=public_truncated, c3nt_env_reward=reward, c3nt_last_position=list(self.drone_pos), c3nt_last_heading=int(self.heading), c3nt_stop_kind='complete' if completed else 'safety_cap' if exhausted else 'running')
        if terminated:
            self.assert_consistent()
        return (self.observation(), reward, public_terminated, public_truncated, clean)

    def info(self, final=False):
        result = super().info(final)
        if not getattr(self, '_nrg_ready', False):
            return result
        result.update(nrg_version=VERSION, nrg_time_cap_sec=self._nrg_cap, nrg_first_visit_coefficient=1.0 / self.target_cells_count, nrg_effective_legacy_o=COST_MULTIPLIER * self.time_straight_sec / self._nrg_cap, c3nt_cost_multiplier=COST_MULTIPLIER, nrg_initial_covered_count=self._nrg_n0, nrg_zero_movement_ack_reward=self._nrg_ack_reward, c3nt_clock_in_observation=False, c3nt_execution_cap_steps=self.max_steps, c3nt_normalization_steps=self._normalization_steps, c3nt_normalization_step_factor=NORMALIZATION_STEP_FACTOR, c3nt_observed_prefix_return=self.total_return, nrg_audited_steps=self._nrg_audited_steps, nrg_detour_steps=self._nrg_detours, nrg_strict_steps=self._nrg_strict, nrg_budget_open_steps=self._nrg_open_steps, nrg_straight_detour_steps=self._nrg_straight_detours, nrg_budget_open_detour_ratio=self._nrg_open_detours / self._nrg_open_steps if self._nrg_open_steps else 0.0, nrg_detour_ratio=self._nrg_detours / self.steps if self.steps else 0.0)
        completed = self.covered_cells_count == self.target_cells_count
        stopped = bool(self._done)
        result.update(c3nt_terminated=bool(stopped and completed), c3nt_truncated=bool(stopped and (not completed)), c3nt_stop_kind='complete' if stopped and completed else 'safety_cap' if stopped else 'running')
        if stopped and (not completed):
            result['c3nt_native_termination_reason'] = result.get('termination_reason')
            result['termination_reason'] = 'external_step_cap'
        for (b, count) in enumerate(self._nrg_pre_budget_hist):
            result[f'nrg_budget_before_{b}_count'] = int(count)
        if final and self._done:
            beta = self.config.tradeoff_beta
            wait = self.weighted_waiting_sum / self.total_priority_mass
            unscaled_cost = ((1 - beta) * self.elapsed_time_sec + beta * wait) / self._nrg_cap
            cost = COST_MULTIPLIER * unscaled_cost
            terminal = 2.0 if self.covered_cells_count == self.target_cells_count else 0.0
            expected = (self.covered_cells_count - self._nrg_n0) / self.target_cells_count - cost + terminal
            result.update(nrg_objective_cost=cost, nrg_expected_return=expected, c3nt_unscaled_objective_cost=unscaled_cost, c3nt_counterfactual_return_cost1=(self.covered_cells_count - self._nrg_n0) / self.target_cells_count - unscaled_cost + terminal, nrg_return_identity_error=abs(expected - self.total_return), c3nt_is_complete=bool(terminal > 0), c3nt_return_scope='complete_episode' if terminal > 0 else 'truncated_prefix_not_full_return', c3nt_failure_penalty=0.0, c3nt_bootstrap_excluded_from_env_log=True)
        return result

    def assert_consistent(self):
        super().assert_consistent()
        if getattr(self, '_nrg_ready', False) and self._done:
            result = self.info(final=True)
            require(result['nrg_return_identity_error'] < 1e-06, 'NRG1 return identity failed.')
            require(result['nrg_objective_cost'] >= -1e-07, 'Negative objective cost.')

def wrap_factory(original_factory):
    classes = {}

    def factory(config):
        base = original_factory(config)
        typ = type(base)
        if typ not in classes:
            wrapped = type('C3NT' + typ.__name__, (C3NoClockMixin, typ), {'__module__': __name__})
            globals()[wrapped.__name__] = wrapped
            classes[typ] = wrapped
        return classes[typ](config)
    return factory

def extend_observation_space(space):
    """Compatibility hook for the cloned trainer; do NOT append a time feature."""
    from gymnasium import spaces
    require(isinstance(space, spaces.Dict), 'Expected Dict observation_space.')
    require(set(space.spaces) == OBS_KEYS, f'Unexpected original observation space: {list(space.spaces)}')
    return space

def audit_model(model):
    require(model.gamma == 1.0 and model.gae_lambda == 0.95, 'Unexpected PPO gamma/lambda.')
    require(model.rollout_buffer.gamma == 1.0, 'Rollout buffer gamma is not 1.0.')
    require(set(model.observation_space.spaces) == OBS_KEYS, 'Unexpected observation including a clock.')

def audit_delivery(local, model):
    """Runs BEFORE SB3 timeout bootstrapping, never changes collected rewards.

    Record an independent expectation for each rollout reward; audit_rollout_end
    checks the ACTUAL buffer after native SB3 has applied its continuation value.
    """
    import torch
    infos = local['infos']
    delivered = np.asarray(local['rewards'], dtype=np.float64).reshape(-1).copy()
    expected = np.asarray([sum((float(i['step_reward_' + k]) for k in REWARD_KEYS)) for i in infos])
    require(all((i.get('nrg_reward_replaced') is True for i in infos)), 'Unpatched environment step.')
    require(np.allclose(delivered, expected, rtol=1e-06, atol=2e-07), 'Raw environment reward mismatch.')
    continuation = np.zeros_like(delivered)
    for (j, info) in enumerate(infos):
        truncated = bool(info.get('c3nt_truncated', False))
        require(bool(info.get('TimeLimit.truncated', False)) == truncated, 'DummyVecEnv truncation flag does not match core signal.')
        require(not (truncated and info.get('c3nt_terminated', False)), 'Both termination flags set.')
        if truncated:
            require(bool(local['dones'][j]), 'Truncation did not reset vector environment.')
            terminal = info.get('terminal_observation')
            require(isinstance(terminal, dict) and set(terminal) == OBS_KEYS, 'Missing original last-state observation for timeout continuation.')
            if 'start_r' in info:
                require(int(terminal['heading']) in range(4), 'Bad terminal heading.')
            with torch.no_grad():
                (tensor, _) = model.policy.obs_to_tensor(terminal)
                continuation[j] = float(model.policy.predict_values(tensor).flatten()[0].item()) * model.gamma
            require(np.isfinite(continuation[j]), 'Nonfinite timeout value.')
    if not hasattr(model, '_cost5_diag_raw'):
        model._cost5_diag_raw = []
        model._cost5_diag_bootstrap = []
    model._cost5_diag_raw.append(delivered.copy())
    model._cost5_diag_bootstrap.extend((float(continuation[j]) for (j, i) in enumerate(infos) if i.get('c3nt_truncated', False)))
    if not hasattr(model, '_c3nt_pending_rewards'):
        model._c3nt_pending_rewards = []
    model._c3nt_pending_rewards.append((delivered + continuation, int(sum((bool(i.get('c3nt_truncated')) for i in infos)))))
    model.logger.record('c3nt/raw_reward_delivery_max_abs_error', float(np.max(np.abs(delivered - expected))))

def audit_rollout_end(local, model):
    """Check SB3 added exactly gamma*V(unreset_last_obs), once, at truncations."""
    pending = getattr(model, '_c3nt_pending_rewards', [])
    require(len(pending) == int(model.n_steps), f'Rollout audit lengths differ: {len(pending)} vs {model.n_steps}')
    expected = np.stack([r for (r, _) in pending])
    actual = np.asarray(model.rollout_buffer.rewards, dtype=np.float64)
    require(actual.shape == expected.shape, 'Unexpected rollout buffer reward shape.')
    require(np.allclose(actual, expected, rtol=2e-05, atol=3e-06), 'SB3 timeout continuation was missing, wrong or applied more than once.')
    n = sum((n for (_, n) in pending))
    model.logger.record('c3nt/truncation_bootstraps_in_rollout', n)
    model.logger.record('c3nt/bootstrap_buffer_max_abs_error', float(np.max(np.abs(actual - expected))))
    log_cost5_rollout_diagnostics(model)
    model._c3nt_last_rollout_audit = dict(transitions=int(actual.size), truncations=n, max_abs_error=float(np.max(np.abs(actual - expected))))
    model._c3nt_pending_rewards = []

def log_cost5_rollout_diagnostics(model):
    """Read-only statistics before SB3 minibatch advantage normalization."""
    raw = np.asarray(getattr(model, '_cost5_diag_raw', []), dtype=np.float64)
    boot = np.asarray(getattr(model, '_cost5_diag_bootstrap', []), dtype=np.float64)
    buffer = model.rollout_buffer
    values = {'cost5_raw_env_reward_mean': float(raw.mean()) if raw.size else 0.0, 'cost5_raw_env_reward_std': float(raw.std()) if raw.size else 0.0, 'cost5_advantage_mean_pre_normalization': float(np.mean(buffer.advantages)), 'cost5_advantage_std_pre_normalization': float(np.std(buffer.advantages)), 'cost5_return_mean': float(np.mean(buffer.returns)), 'cost5_return_std': float(np.std(buffer.returns)), 'cost5_value_mean': float(np.mean(buffer.values)), 'cost5_timeout_value_count': int(boot.size), 'cost5_timeout_value_mean': float(boot.mean()) if boot.size else 0.0, 'cost5_timeout_value_min': float(boot.min()) if boot.size else 0.0, 'cost5_timeout_value_max': float(boot.max()) if boot.size else 0.0}
    require(all((np.isfinite(v) for v in values.values())), 'Nonfinite COST5 rollout diagnostics.')
    model._cost5_last_diagnostics = dict(values)
    for (key, value) in values.items():
        model.logger.record('train/' + key, value)
    model._cost5_diag_raw = []
    model._cost5_diag_bootstrap = []
