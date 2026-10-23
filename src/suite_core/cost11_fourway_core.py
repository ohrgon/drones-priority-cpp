"""Four masks with the frozen FULL COST11 objective, CNN3 and external H5 cap.

The original physical cores, six spatial channels, heading and B6 updates are
not rewritten. Only the previously Relaxed-only reward adapter is generalized.
Legal transitions use the ORIGINAL reward_parts() function. Out-of-bound control
attempts (possible only in No Mask) preserve the original configured invalid
penalty and control-time/heading conventions; this extension is explicit in spec.
"""
from __future__ import annotations
import math
import numpy as np
import c3nt_cost_sweep_generated_core as original
from c3nt_cost_sweep_generated_core import (
    reward_parts, reference_masks, observed_budget, extend_observation_space,
    audit_model, audit_delivery, audit_rollout_end,
)
VERSION = 'full_cost11_cnn3_fourway_externalH5_v1_20260928'
COST_MULTIPLIER = 11.0
NORMALIZATION_STEP_FACTOR = 5.0
OBS_KEYS = original.OBS_KEYS
REWARD_KEYS = original.REWARD_KEYS
MOVES = original.MOVES
MODES = ('none', 'boundary', 'relaxed', 'progress')
require = original.require


def selected_mask(core, boundary, strict, budget):
    mode = core.config.mask_mode
    require(mode in MODES, 'Unknown four-way mask mode.')
    if mode == 'none':
        return np.ones(4, dtype=bool)
    if mode == 'boundary':
        return boundary
    if mode == 'progress':
        return strict
    return boundary if budget > 0 else strict


def transition_parts(core, *, dn, mass, dt, completed, exhausted, invalid):
    """No second cost scaling and no failure penalty at a safety truncation."""
    p = reward_parts(new_count=int(dn), n_active=core.target_cells_count,
        mass_before=mass, total_mass=core.total_priority_mass, dt=dt,
        time_cap=core._nrg_cap, beta=core.config.tradeoff_beta,
        completed=bool(completed), exhausted=bool(exhausted))
    # The inherited convention is shared by every condition; only None can
    # actually select an out-of-bound action. It is NOT a phantom movement.
    p['invalid'] = -float(core.config.invalid_action_penalty) * int(invalid)
    return p


class FourWayMixin(original.C3NoClockMixin):
    """Keep the frozen native core behind C3NoClockMixin in the MRO."""
    def _physical(self):
        # Skip the old Relaxed-only adapter, NOT the original B6/physics layer.
        return super(original.C3NoClockMixin, self)

    def reset(self, *args, **kwargs):
        self._nrg_ready = False
        self._nrg_audited_steps = 0
        self._nrg_detours = self._nrg_strict = self._nrg_open_steps = 0
        self._nrg_open_detours = self._nrg_straight_detours = 0
        self._nrg_pre_budget_hist = np.zeros(7, np.int64)
        self._nrg_last = None
        self._nrg_ack_reward = 0.0
        self._f4_non_strict = 0
        for name in ('coverage_reward_scale', 'objective_cost_scale',
                     'success_terminal_reward', 'failure_terminal_penalty'):
            require(getattr(self.config, name) == 0.0, 'Legacy reward must be neutral: '+name)
        require(self.config.mask_mode in MODES, 'Expected one of the four original masks.')
        require(self.config.detour_budget_max == 6, 'Shared budget interface must keep Bmax=6.')
        require(self.config.invalid_time_rule == 'straight_control_interval', 'Invalid control-time rule changed.')
        require(self.config.invalid_heading_rule == 'preserve', 'Invalid heading rule changed.')
        require(self.config.invalid_action_penalty == 1.0, 'Original invalid-action penalty must stay 1.0.')
        obs, _ = self._physical().reset(*args, **kwargs)
        require(self.time_turn_sec >= self.time_straight_sec > 0, 'Invalid time model.')
        self._nrg_n0 = int(self.covered_cells_count)
        self._normalization_steps = int(math.ceil(5 * self.target_cells_count))
        self._nrg_cap = self._normalization_steps * self.time_turn_sec
        self._nrg_ready = True
        require(set(obs) == OBS_KEYS, 'Original observation keys changed.')
        require(observed_budget(obs, 6) == (6 if self.config.mask_mode == 'relaxed' else 0), 'Budget reset differs.')
        return self.observation(), self.info(final=self.complete_at_reset)

    def observation(self):
        obs = dict(self._physical().observation())
        require(set(obs) == OBS_KEYS, 'Observation must remain map/heading/detour_budget.')
        if self.config.mask_mode != 'relaxed':
            require(observed_budget(obs, 6) == 0, 'Non-relaxed budget must be constant zero.')
        return obs

    def immediate_reward(self, action):
        if self.complete_at_reset:
            return 2.0 if not self._done else 0.0
        require(not self._done, 'Episode already stopped.')
        require(isinstance(action, (int, np.integer)) and not isinstance(action, bool) and 0 <= int(action) < 4, 'Bad action ID.')
        action = int(action)
        require(bool(self.action_masks()[action]), 'Selected action is masked out.')
        r, c = self.drone_pos; dr, dc = MOVES[action]; nxt = r+dr, c+dc
        valid = self._in_bbox(*nxt)
        dn = int(valid and self.active_mask[nxt] and not self.covered_map[nxt])
        dt = self.time_straight_sec if (not valid or action == self.heading) else self.time_turn_sec
        p = transition_parts(self, dn=dn, mass=self.remaining_priority_mass, dt=dt,
            completed=self.covered_cells_count+dn == self.target_cells_count,
            exhausted=self.steps+1 >= self.max_steps, invalid=not valid)
        return float(sum(p.values()))

    def step(self, action):
        require(getattr(self, '_nrg_ready', False) and not self._done, 'Reset required.')
        require(isinstance(action, (int, np.integer)) and not isinstance(action, bool) and 0 <= int(action) < 4, 'Bad action ID.')
        action = int(action); singleton = bool(self.complete_at_reset)
        budget = observed_budget(self.observation(), 6)
        previous_return = float(self.total_return); sums = dict(self.reward_sums)
        count = int(self.covered_cells_count); steps = int(self.steps)
        mass = float(self.remaining_priority_mass); clock = float(self.elapsed_time_sec)
        position = tuple(self.drone_pos); heading = int(self.heading)
        invalid_before = int(self.invalid_action_count)
        valid_moves_before = int(self.valid_move_count)
        if singleton:
            valid = True; is_strict = True
        else:
            boundary, strict = reference_masks(self)
            expected = selected_mask(self, boundary, strict, budget)
            require(np.array_equal(self.action_masks(), expected), 'Applied mask differs from selected four-way condition.')
            require(bool(expected[action]), 'Masked action selected.')
            valid = bool(boundary[action]); is_strict = bool(strict[action])
        obs, legacy, done, native_truncated, info = self._physical().step(action)
        require(not native_truncated, 'Frozen physical stop interface changed.')
        invalid = not valid and not singleton
        require(math.isclose(float(legacy), -self.config.invalid_action_penalty*int(invalid), abs_tol=1e-9),
                'Unexpected legacy reward; refuse to stack objectives.')
        dn = self.covered_cells_count-count; dt = self.elapsed_time_sec-clock
        completed = self.covered_cells_count == self.target_cells_count
        exhausted = self.steps >= self.max_steps and not completed
        require(bool(done) == bool(completed or exhausted), 'Native termination differs.')
        if not singleton:
            require(self.steps == steps+1, 'Every control attempt must consume one step.')
            dr, dc = MOVES[action]
            require(tuple(self.drone_pos) == ((position[0]+dr, position[1]+dc) if valid else position), 'Position transition changed.')
            require(int(self.heading) == (action if valid else heading), 'Heading transition changed.')
            require(self.invalid_action_count == invalid_before+int(invalid), 'Invalid count changed.')
            require(self.valid_move_count == valid_moves_before+int(valid), 'Attempt mistaken for movement.')
            expected_dt = self.time_straight_sec if (invalid or action == heading) else self.time_turn_sec
            require(math.isclose(dt, expected_dt, abs_tol=1e-8, rel_tol=1e-9), 'Control/movement time changed.')
            expected_budget = (6 if dn else budget if is_strict else budget-1) if self.config.mask_mode == 'relaxed' else 0
            require(observed_budget(obs, 6) == expected_budget, 'Native budget update changed.')
            detour = self.config.mask_mode == 'relaxed' and not is_strict
            self._nrg_detours += int(detour)
            self._nrg_strict += int(is_strict)
            self._nrg_open_steps += int(budget > 0)
            self._nrg_open_detours += int(budget > 0 and detour)
            self._nrg_straight_detours += int(detour and action == heading)
            self._nrg_pre_budget_hist[budget] += 1
            self._f4_non_strict += int(not is_strict)
        p = transition_parts(self, dn=dn, mass=mass, dt=dt, completed=completed, exhausted=exhausted, invalid=invalid)
        reward = float(sum(p.values()))
        self.reward_sums = {k:float(sums[k])+p[k] for k in REWARD_KEYS}
        self.total_return = previous_return+reward
        self._nrg_audited_steps += int(not singleton)
        if singleton:self._nrg_ack_reward = reward
        self._nrg_last = dict(budget_before=budget, budget_after=observed_budget(obs, 6),
            selected_is_strict=int(is_strict), selected_is_detour=int(self.config.mask_mode == 'relaxed' and not is_strict and not singleton),
            is_straight=int(valid and action == heading and not singleton))
        public_terminated = bool(completed); public_truncated = bool(exhausted)
        clean = {k:v for k,v in info.items() if not k.startswith('step_reward_')}
        clean.update(self.info(final=bool(done)))
        clean.update({'step_reward_'+k:v for k,v in p.items()})
        clean.update({'nrg_step_'+k:v for k,v in self._nrg_last.items()})
        clean.update(nrg_reward_replaced=True, c3nt_terminated=public_terminated,
            c3nt_truncated=public_truncated, c3nt_bootstrap_required=public_truncated,
            c3nt_env_reward=reward, c3nt_last_position=list(self.drone_pos), c3nt_last_heading=int(self.heading),
            c3nt_stop_kind='complete' if completed else 'safety_cap' if exhausted else 'running')
        if done:self.assert_consistent()
        return self.observation(), reward, public_terminated, public_truncated, clean

    def info(self, final=False):
        result = self._physical().info(final)
        if not getattr(self, '_nrg_ready', False):return result
        result.update(nrg_version=VERSION, nrg_time_cap_sec=self._nrg_cap,
            nrg_first_visit_coefficient=1.0/self.target_cells_count,
            nrg_effective_legacy_o=11*self.time_straight_sec/self._nrg_cap,
            c3nt_cost_multiplier=11.0, nrg_initial_covered_count=self._nrg_n0,
            nrg_zero_movement_ack_reward=self._nrg_ack_reward,
            c3nt_clock_in_observation=False, c3nt_execution_cap_steps=self.max_steps,
            c3nt_normalization_steps=self._normalization_steps, c3nt_normalization_step_factor=5.0,
            c3nt_observed_prefix_return=self.total_return, nrg_audited_steps=self._nrg_audited_steps,
            nrg_detour_steps=self._nrg_detours, nrg_strict_steps=self._nrg_strict,
            nrg_budget_open_steps=self._nrg_open_steps, nrg_straight_detour_steps=self._nrg_straight_detours,
            nrg_budget_open_detour_ratio=self._nrg_open_detours/self._nrg_open_steps if self._nrg_open_steps else 0.,
            nrg_detour_ratio=self._nrg_detours/self.steps if self.steps else 0.,
            fourway_mask=self.config.mask_mode, fourway_nonstrict_selected_count=self._f4_non_strict,
            fourway_invalid_penalty=self.config.invalid_action_penalty,
            fourway_actual_movement_count=self.valid_move_count,
            fourway_attempted_action_count=self.steps)
        completed = self.covered_cells_count == self.target_cells_count
        stopped = bool(self._done)
        result.update(c3nt_terminated=bool(stopped and completed), c3nt_truncated=bool(stopped and not completed),
            c3nt_stop_kind='complete' if stopped and completed else 'safety_cap' if stopped else 'running')
        if stopped and not completed:
            result['c3nt_native_termination_reason'] = result.get('termination_reason')
            result['termination_reason'] = 'external_step_cap'
        for b,n in enumerate(self._nrg_pre_budget_hist):result[f'nrg_budget_before_{b}_count'] = int(n)
        if final and self._done:
            beta=self.config.tradeoff_beta; wait=self.weighted_waiting_sum/self.total_priority_mass
            unscaled=((1-beta)*self.elapsed_time_sec+beta*wait)/self._nrg_cap
            invalid_cost=self.config.invalid_action_penalty*self.invalid_action_count
            expected=(self.covered_cells_count-self._nrg_n0)/self.target_cells_count-11*unscaled+(2. if completed else 0.)-invalid_cost
            result.update(nrg_objective_cost=11*unscaled, nrg_expected_return=expected,
                c3nt_unscaled_objective_cost=unscaled,
                c3nt_counterfactual_return_cost1=(self.covered_cells_count-self._nrg_n0)/self.target_cells_count-unscaled+(2. if completed else 0.)-invalid_cost,
                nrg_return_identity_error=abs(expected-self.total_return), c3nt_is_complete=bool(completed),
                c3nt_return_scope='complete_episode' if completed else 'truncated_prefix_not_full_return',
                c3nt_failure_penalty=0.0, c3nt_bootstrap_excluded_from_env_log=True)
        return result


def wrap_factory(factory):
    classes={}
    def make(config):
        base=factory(config);typ=type(base)
        if typ not in classes:
            new=type('FULL4Way'+typ.__name__, (FourWayMixin,typ), {'__module__':__name__})
            globals()[new.__name__]=new;classes[typ]=new
        return classes[typ](config)
    return make
