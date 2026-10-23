"""Chapter 3 physics + Chapter 4 none/boundary/progress action constraints.

Source-defined formulae are inherited from coverage_core, including pre-action W_t.
Unspecified invalid-action conventions are explicit configuration decisions, NOT
asserted to be values in the manuscript. Classical trajectories use boundary mode.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import math
import numpy as np
from coverage_core import CoverageCore, EnvConfig, MOVES, ProgressInvariantError

@dataclass(frozen=True)
class ExperimentEnvConfig(EnvConfig):
    mask_mode: str = "progress"
    paper_mode: str = "none"
    paper_local_radius: int = 5
    paper_coarse_factor: int = 4
    invalid_action_penalty: float = 1.0
    invalid_time_rule: str = "straight_control_interval"
    invalid_heading_rule: str = "preserve"
    # Additional choices have provenance in protocol_decisions.json / README.
    def __post_init__(self):
        super().__post_init__()
        if self.mask_mode not in {"none", "boundary", "progress"}:
            raise ValueError("mask_mode must be none, boundary, or progress")
        if not np.isfinite(self.invalid_action_penalty) or self.invalid_action_penalty < 0:
            raise ValueError("invalid_action_penalty must be nonnegative")
        if self.invalid_time_rule not in {"straight_control_interval", "attempted_turn_interval"}:
            raise ValueError("Choose an explicit invalid_time_rule; zero-time stays are not silently allowed")
        if self.invalid_heading_rule not in {"preserve", "attempted_action"}:
            raise ValueError("invalid_heading_rule must be preserve or attempted_action")

class ExperimentCore(CoverageCore):
    """Same movement/reward accounting for every method; masks are a separate layer."""
    def __init__(self, config: Optional[ExperimentEnvConfig] = None):
        super().__init__(config or ExperimentEnvConfig())

    def reset(self, *args, **kwargs):
        self.invalid_control_time_sec = 0.0
        self.invalid_heading_change_count = 0
        self.nonprogress_steps = 0
        obs, info = super().reset(*args, **kwargs)
        self.reward_sums["invalid"] = 0.0
        return obs, self.info(final=self.complete_at_reset)

    def feasible_mask(self):
        r, c = self.drone_pos
        return np.array([self._in_bbox(r+dr, c+dc) for dr, dc in MOVES], dtype=bool)

    def progress_mask_details(self):
        # Base cache is only used for the progress constraint.
        return CoverageCore.mask_details(self)

    def mask_details(self):
        if not self._ready:
            raise RuntimeError("reset must precede mask computation")
        if self.complete_at_reset or self._done:
            return np.ones(4, dtype=bool), "terminal_sentinel"
        if self.config.mask_mode == "none":
            return np.ones(4, dtype=bool), "unmasked"
        if self.config.mask_mode == "boundary":
            m = self.feasible_mask()
            if not m.any():
                raise ProgressInvariantError("Nonterminal feasible graph has no edge")
            return m, "boundary"
        return self.progress_mask_details()

    def immediate_reward(self, action):
        if not isinstance(action, (int, np.integer)) or action not in range(4):
            raise ValueError("Invalid action ID")
        if self.complete_at_reset:
            return 0.0
        if not self.action_masks()[action]:
            raise ValueError("Action is not allowed under selected mask")
        r, c = self.drone_pos
        dr, dc = MOVES[action]
        nxt = (r+dr,c+dc)
        valid = self._in_bbox(*nxt)
        turn = action != self.heading
        dt = self.time_turn_sec if turn else self.time_straight_sec
        if not valid and self.config.invalid_time_rule == "straight_control_interval":
            dt = self.time_straight_sec
        new = int(valid and self.active_mask[nxt] and not self.covered_map[nxt])
        k = self.config
        reward = k.coverage_reward_scale*new - k.objective_cost_scale*(
            k.tradeoff_beta*self.remaining_priority_mass/self.total_priority_mass+1-k.tradeoff_beta)*dt/self.time_straight_sec
        if not valid:
            reward -= k.invalid_action_penalty
        if self.covered_cells_count+new == self.target_cells_count:
            reward += k.success_terminal_reward
        elif self.steps+1 >= self.max_steps:
            reward -= k.failure_terminal_penalty
        return float(reward)

    def step(self, action):
        if not self._ready or self._done:
            raise RuntimeError("Reset required before another action")
        if self.complete_at_reset:
            self._done, self.termination_reason = True, "complete_at_reset"
            return self.observation(), 0.0, True, False, self.info(final=True)
        if isinstance(action, bool) or not isinstance(action,(int,np.integer)) or action not in range(4):
            raise ValueError("action must be integer 0..3")
        action = int(action)
        mask, phase = self.mask_details()
        if not mask[action]:
            raise ProgressInvariantError(f"Masked-out action {action}; mode={self.config.mask_mode}")
        pos = self.drone_pos
        dr, dc = MOVES[action]
        nxt = (pos[0]+dr,pos[1]+dc)
        valid = self._in_bbox(*nxt)
        turned = action != self.heading
        dt = self.time_turn_sec if turned else self.time_straight_sec
        if not valid and self.config.invalid_time_rule == "straight_control_interval":
            dt = self.time_straight_sec
        mass = self.remaining_priority_mass
        self.mask_counts[int(mask.sum())] += 1
        self.steps += 1
        self.elapsed_time_sec += dt
        self.weighted_waiting_sum += mass*dt
        self.collected_priority_area += (1-mass/self.total_priority_mass)*dt
        self.direct_steps += int(phase == "direct")
        self.progress_steps += int(phase == "progress")
        new = False
        if valid:
            # Explicit progress verification, also useful for revisit diagnostics.
            is_new = bool(self.active_mask[nxt] and not self.covered_map[nxt])
            improves = False
            if not is_new:
                d = self.distance_field()
                improves = int(d[nxt]) == int(d[pos])-1
            if phase == "progress" and not improves:
                raise ProgressInvariantError("BFS progress invariant violated")
            self.nonprogress_steps += int(not is_new and not improves)
            self.valid_move_count += 1
            self.move_turn_count += int(turned)
            self.move_straight_count += int(not turned)
            self.path_length_m = self.valid_move_count*self.config.cell_size_m
            revisit = bool(self.active_mask[nxt] and self.covered_map[nxt])
            self.revisit_steps += int(revisit)
            self.revisit_progress_count += int(revisit and improves)
            self.zero_transit_steps += int(not self.active_mask[nxt])
            self.oscillation_steps += int(self.prev_pos is not None and nxt == self.prev_pos)
            self.prev_pos, self.drone_pos = pos, nxt
            self.heading = action
            self.visited_map[nxt] = True
            self.position_history *= self.config.history_alpha
            self.position_history[nxt] = 1.0
            new = self._record_first_visit(nxt)
            self.new_cover_count += int(new)
        else:
            self.invalid_action_count += 1
            self.invalid_control_time_sec += dt
            if self.config.invalid_heading_rule == "attempted_action":
                self.invalid_heading_change_count += int(turned)
                self.heading = action
            # No phantom movement, coverage, revisit, or spatial history decay.
        k = self.config
        pieces = {
            "coverage": k.coverage_reward_scale*int(new),
            "latency": -k.objective_cost_scale*k.tradeoff_beta*mass/self.total_priority_mass*dt/self.time_straight_sec,
            "mission_time": -k.objective_cost_scale*(1-k.tradeoff_beta)*dt/self.time_straight_sec,
            "terminal": 0.0,
            "invalid": -k.invalid_action_penalty*int(not valid),
        }
        if self.covered_cells_count == self.target_cells_count:
            self._done, self.termination_reason = True,"complete"
            pieces["terminal"] = k.success_terminal_reward
        elif self.steps >= self.max_steps:
            self._done, self.termination_reason = True,"budget_exhausted"
            pieces["terminal"] = -k.failure_terminal_penalty
        reward = float(sum(pieces.values()))
        for key,val in pieces.items(): self.reward_sums[key] += val
        self.total_return += reward
        self._mask_cache = None
        info = self.info(final=self._done)
        info.update({"step_reward_"+key:val for key,val in pieces.items()})
        info.update(mask_phase=phase, mask_cardinality=int(mask.sum()),
                    remaining_priority_before=mass, traversal_time_sec=dt,
                    is_new_active_cell=int(new), is_turn=int(turned and valid),
                    is_valid_movement=int(valid), executed_grid_action=action)
        return self.observation(),reward,self._done,False,info

    def info(self, final=False):
        out = super().info(final)
        out.update(mask_mode=self.config.mask_mode,
                   invalid_control_time_sec=self.invalid_control_time_sec,
                   invalid_heading_change_count=self.invalid_heading_change_count,
                   nonprogress_steps=self.nonprogress_steps)
        return out

    def assert_consistent(self):
        assert self.steps == self.valid_move_count+self.invalid_action_count
        assert self.steps == self.mask_counts.sum()
        assert self.valid_move_count == self.move_turn_count+self.move_straight_count
        assert self.valid_move_count == self.new_cover_count+self.revisit_steps+self.zero_transit_steps
        assert self.covered_cells_count == self.initial_covered_count+self.new_cover_count
        assert self.covered_cells_count == np.isfinite(self.first_visit_time[self.active_mask]).sum()
        expected = self.move_turn_count*self.time_turn_sec+self.move_straight_count*self.time_straight_sec+self.invalid_control_time_sec
        assert math.isclose(expected,self.elapsed_time_sec,rel_tol=1e-9,abs_tol=1e-7)
        assert math.isclose(self.path_length_m,self.valid_move_count*self.config.cell_size_m)
        assert math.isclose(self.total_return,sum(self.reward_sums.values()),rel_tol=1e-8,abs_tol=1e-7)
        assert math.isclose(self.remaining_priority_mass,float(self.weight_map[self.uncovered_mask()].sum()))
        assert self.exact_metrics()["weighted_wait_identity_abs_error"] <= 1e-7*max(1,abs(self.weighted_waiting_sum))
        if self.config.mask_mode == "progress":
            assert self.invalid_action_count == 0 and self.nonprogress_steps == 0


def make_core(config):
    if config.paper_mode != 'none':
        from legacy_ch4 import PaperAdaptedCore
        return PaperAdaptedCore(config)
    return ExperimentCore(config)
