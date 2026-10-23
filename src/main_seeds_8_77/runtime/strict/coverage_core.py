"""NumPy-only implementation of the revised FULL planner environment.

The old environment is intentionally NOT imported: legacy rewards cannot leak in.
Action IDs and default priority-mass initialization match the uploaded source.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Optional
import math
import numpy as np

VERSION = "full_progress_objective_v2.0"
MOVES = ((-1, 0), (0, 1), (1, 0), (0, -1))  # up, right, down, left
INF = 10**9


@dataclass(frozen=True)
class EnvConfig:
    grid_size: int = 50
    step_budget_factor: float = 5.0
    cell_size_m: float = 16.0
    v_straight: float = 11.462
    v_turn: float = 5.670
    history_alpha: float = 0.96
    coverage_reward_scale: float = 0.03
    objective_cost_scale: float = 0.006
    tradeoff_beta: float = 0.5
    success_terminal_reward: float = 45.0
    failure_terminal_penalty: float = 60.0
    initialization: str = "priority_mass"
    # Reward defaults are PROVISIONAL, not claimed to be validation-selected.

    def __post_init__(self) -> None:
        if isinstance(self.grid_size, bool) or int(self.grid_size) != self.grid_size or self.grid_size < 1:
            raise ValueError("grid_size must be a positive integer")
        for key in ("step_budget_factor", "cell_size_m", "v_straight", "v_turn"):
            val = getattr(self, key)
            if not np.isfinite(val) or val <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if self.v_turn > self.v_straight:
            raise ValueError("v_turn must not exceed v_straight for the stated turn-time surrogate")
        for key in ("coverage_reward_scale", "objective_cost_scale", "success_terminal_reward", "failure_terminal_penalty"):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        for key in ("history_alpha", "tradeoff_beta"):
            if not np.isfinite(getattr(self, key)) or not 0 <= getattr(self, key) <= 1:
                raise ValueError(f"{key} must be in [0, 1]")
        if self.initialization not in {"priority_mass", "fixed_corner", "random_active", "priority_neutral"}:
            raise ValueError("Unsupported initialization")


def validate_priority_map(array: np.ndarray, grid_size: int, min_active: int = 1) -> np.ndarray:
    """Strict loading: never silently crop, pad, quantize, or invent a dummy map."""
    a = np.asarray(array)
    if a.ndim != 2 or a.shape != (grid_size, grid_size):
        raise ValueError(f"Expected {(grid_size, grid_size)} map, got {a.shape}; no automatic resize")
    if a.dtype.kind not in "biuf" or not np.all(np.isfinite(a)):
        raise ValueError("Map must contain finite numeric values")
    if np.any(a < 0) or np.any(a > 5) or np.any(a != np.rint(a)):
        raise ValueError("Map priorities must be integer values in {0,1,2,3,4,5}")
    if np.count_nonzero(a) < min_active:
        raise ValueError(f"Map must have at least {min_active} active cell(s)")
    return np.array(a, dtype=np.uint8, copy=True)


class ProgressInvariantError(RuntimeError):
    """Graph/mask implementation mismatch; never silently use a fallback."""


class CoverageCore:
    """Full transition/reward/metric logic, testable without Gym or SB3.

    Invalid or non-progress actions are programming errors in this FULL-only core.
    They raise before altering state. This is not the unmasked-PPO baseline engine.
    """

    def __init__(self, config: Optional[EnvConfig] = None):
        self.config = config or EnvConfig()
        self.grid_size = self.config.grid_size
        self.time_straight_sec = self.config.cell_size_m / self.config.v_straight
        self.time_turn_sec = self.config.cell_size_m / self.config.v_turn
        self._ready = False

    def reset(self, priority_map: np.ndarray, *, seed: int = 0,
              start: Optional[tuple[int, int]] = None, heading: Optional[int] = None,
              map_name: str = "in_memory") -> tuple[dict, dict]:
        self.weight_map = validate_priority_map(priority_map, self.grid_size)
        self.active_mask = self.weight_map > 0
        rc = np.argwhere(self.active_mask)
        self.min_r, self.min_c = (int(x) for x in rc.min(axis=0))
        self.max_r, self.max_c = (int(x) for x in rc.max(axis=0))
        self.bbox_mask = np.zeros_like(self.active_mask)
        self.bbox_mask[self.min_r:self.max_r + 1, self.min_c:self.max_c + 1] = True
        self.bbox_area = int(self.bbox_mask.sum())
        self.target_cells_count = int(self.active_mask.sum())
        self.total_priority_mass = float(self.weight_map.sum())
        self.remaining_priority_mass = self.total_priority_mass
        self.covered_map = np.zeros_like(self.active_mask)
        self.visited_map = np.zeros_like(self.active_mask)  # actual traversals, INCLUDING transit
        self.position_history = np.zeros(self.weight_map.shape, dtype=np.float32)
        self.first_visit_time = np.full(self.weight_map.shape, np.nan, dtype=np.float64)
        self.first_visit_distance = np.full(self.weight_map.shape, np.nan, dtype=np.float64)
        self.first_visit_step = np.full(self.weight_map.shape, -1, dtype=np.int64)
        self.current_map_name = str(map_name)
        self.rng = np.random.default_rng(seed)
        self.covered_cells_count = 0
        self.initial_covered_count = 0
        self.steps = self.valid_move_count = self.invalid_action_count = 0
        self.move_straight_count = self.move_turn_count = 0
        self.new_cover_count = self.revisit_steps = self.zero_transit_steps = 0
        self.oscillation_steps = self.revisit_progress_count = 0
        self.max_steps = int(math.ceil(self.config.step_budget_factor * self.target_cells_count))
        self.elapsed_time_sec = self.path_length_m = 0.0
        self.weighted_waiting_sum = self.collected_priority_area = 0.0
        self.reward_sums = dict(coverage=0.0, latency=0.0, mission_time=0.0, terminal=0.0)
        self.total_return = 0.0
        self.mask_counts = np.zeros(5, dtype=np.int64)
        self.direct_steps = self.progress_steps = 0
        self.prev_pos = None
        self._dist_cache = None
        self._dist_version = -1
        self._mask_cache = None
        self._done = False
        self.termination_reason = "ongoing"
        if start is None:
            self.drone_pos = self._choose_start()
        else:
            if len(start) != 2 or any(int(x) != x for x in start):
                raise ValueError("start must be two integer grid coordinates")
            self.drone_pos = tuple(int(x) for x in start)
            if not self._in_bbox(*self.drone_pos):
                raise ValueError("start must lie inside active bounding box")
        self.start_pos = self.drone_pos
        self.visited_map[self.drone_pos] = True
        self.position_history[self.drone_pos] = 1.0
        self._record_first_visit(self.drone_pos)  # visit at t=0, but NO unreturned reward
        self.initial_covered_count = self.covered_cells_count
        if heading is not None and (not isinstance(heading, (int, np.integer)) or heading not in range(4)):
            raise ValueError("heading must be an action ID 0..3")
        self.heading = int(heading) if heading is not None else self._choose_initial_heading()
        self.start_heading = self.heading
        self._ready = True
        self.complete_at_reset = self.covered_cells_count == self.target_cells_count
        # Evaluation returns success immediately on a singleton. Gym's first step
        # can acknowledge it once with zero elapsed time/action count.
        return self.observation(), self.info(final=self.complete_at_reset)

    def _in_bbox(self, r: int, c: int) -> bool:
        return self.min_r <= r <= self.max_r and self.min_c <= c <= self.max_c

    def _choose_start(self) -> tuple[int, int]:
        rc = np.argwhere(self.active_mask)
        mode = self.config.initialization
        if mode == "fixed_corner":
            return self.min_r, self.min_c  # may be a traversable transit cell
        if mode == "random_active":
            return tuple(int(x) for x in rc[self.rng.integers(len(rc))])
        if mode == "priority_neutral":
            boundary = self.active_mask & self.dilate4(self.bbox_mask & ~self.active_mask)
            boundary[self.min_r, :] |= self.active_mask[self.min_r, :]
            boundary[self.max_r, :] |= self.active_mask[self.max_r, :]
            boundary[:, self.min_c] |= self.active_mask[:, self.min_c]
            boundary[:, self.max_c] |= self.active_mask[:, self.max_c]
            cells = np.argwhere(boundary)
            # geometry only; row-major tie after smallest r+c
            return tuple(int(x) for x in cells[np.argmin(cells[:, 0] + cells[:, 1])])
        # Copied tie conventions from the uploaded original: TL, BL, TR, BR.
        mr, mc = (self.min_r + self.max_r) // 2, (self.min_c + self.max_c) // 2
        qs = [(self.min_r, mr+1, self.min_c, mc+1),
              (mr+1, self.max_r+1, self.min_c, mc+1),
              (self.min_r, mr+1, mc+1, self.max_c+1),
              (mr+1, self.max_r+1, mc+1, self.max_c+1)]
        sums = [float(self.weight_map[a:b, c:d].sum()) for a, b, c, d in qs]
        q = int(np.argmax(sums))
        a, b, c, d = qs[q]
        cells = np.argwhere(self.active_mask[a:b, c:d]) + (a, c)
        sr, sc = ((1, 1), (-1, 1), (1, -1), (-1, -1))[q]
        return tuple(int(x) for x in cells[np.argmin(sr*cells[:, 0] + sc*cells[:, 1])])

    def _choose_initial_heading(self) -> int:
        r, c = self.drone_pos
        cr, cc = .5*(self.min_r+self.max_r), .5*(self.min_c+self.max_c)
        cand = []
        for a, (dr, dc) in enumerate(MOVES):
            nr, nc = r+dr, c+dc
            if self._in_bbox(nr, nc):
                new = int(self.active_mask[nr, nc] and not self.covered_map[nr, nc])
                # Sensitivity modes other than priority_mass use a priority-neutral heading.
                w = int(self.weight_map[nr, nc]) if new and self.config.initialization == "priority_mass" else 0
                cand.append((new, w, -abs(nr-cr)-abs(nc-cc), -a, a))
        return max(cand)[-1] if cand else 0

    @staticmethod
    def dilate4(mask: np.ndarray) -> np.ndarray:
        out = np.zeros_like(mask, dtype=bool)
        out[1:] |= mask[:-1]
        out[:-1] |= mask[1:]
        out[:, 1:] |= mask[:, :-1]
        out[:, :-1] |= mask[:, 1:]
        return out

    def uncovered_mask(self) -> np.ndarray:
        return self.active_mask & ~self.covered_map

    def frontier_mask(self) -> np.ndarray:
        u = self.uncovered_mask()
        return u & self.dilate4(self.bbox_mask & ~u)

    def distance_field(self) -> np.ndarray:
        """Multi-source BFS over ALL traversable bbox cells, not only active cells."""
        if self._dist_cache is not None and self._dist_version == self.covered_cells_count:
            return self._dist_cache
        dist = np.full(self.weight_map.shape, INF, dtype=np.int32)
        sources = np.argwhere(self.uncovered_mask())
        if len(sources) == 0:
            dist[self.bbox_mask] = 0
        else:
            q = deque((int(r), int(c)) for r, c in sources)
            dist[sources[:, 0], sources[:, 1]] = 0
            while q:
                r, c = q.popleft()
                d = int(dist[r, c]) + 1
                for dr, dc in MOVES:
                    nr, nc = r+dr, c+dc
                    if self._in_bbox(nr, nc) and dist[nr, nc] == INF:
                        dist[nr, nc] = d
                        q.append((nr, nc))
        self._dist_cache, self._dist_version = dist, self.covered_cells_count
        return dist

    def mask_details(self) -> tuple[np.ndarray, str]:
        if not self._ready:
            raise RuntimeError("reset() must be called before action_masks()")
        if self.covered_cells_count == self.target_cells_count or self._done:
            # Library sentinel only; never a selectable nonterminal fallback.
            return np.ones(4, dtype=bool), "terminal_sentinel"
        if self._mask_cache is not None:
            mask, phase = self._mask_cache
            return mask.copy(), phase
        r, c = self.drone_pos
        valid = np.array([self._in_bbox(r+dr, c+dc) for dr, dc in MOVES], dtype=bool)
        direct = np.zeros(4, dtype=bool)
        for a, (dr, dc) in enumerate(MOVES):
            if valid[a]:
                direct[a] = self.active_mask[r+dr, c+dc] and not self.covered_map[r+dr, c+dc]
        if direct.any():
            mask, phase = direct, "direct"
        else:
            dist = self.distance_field()
            mask = np.array([bool(valid[a] and dist[r+dr, c+dc] == dist[r, c]-1)
                             for a, (dr, dc) in enumerate(MOVES)], dtype=bool)
            phase = "progress"
        if not mask.any():
            raise ProgressInvariantError(f"No progress action: map={self.current_map_name}, pos={self.drone_pos}, remaining={self.target_cells_count-self.covered_cells_count}")
        self._mask_cache = (mask.copy(), phase)
        return mask.copy(), phase

    def action_masks(self) -> np.ndarray:
        return self.mask_details()[0]

    def observation(self) -> dict[str, Any]:
        p = np.zeros_like(self.weight_map)
        p[self.drone_pos] = 255
        return {"map": np.stack([
            self.weight_map * np.uint8(51),
            self.bbox_mask.astype(np.uint8)*255,
            self.uncovered_mask().astype(np.uint8)*255,
            self.frontier_mask().astype(np.uint8)*255,
            p,
            (np.clip(self.position_history, 0, 1)*255).astype(np.uint8),
        ]), "heading": int(self.heading)}

    def immediate_reward(self, action: int) -> float:
        """One-step reward only. It does not use successor priority as an extra bonus."""
        mask = self.action_masks()
        if not isinstance(action, (int, np.integer)) or action not in range(4) or not mask[action]:
            raise ValueError("action is not admissible")
        if self.covered_cells_count == self.target_cells_count:
            return 0.0
        r, c = self.drone_pos
        dr, dc = MOVES[action]
        new = bool(self.active_mask[r+dr, c+dc] and not self.covered_map[r+dr, c+dc])
        dt = self.time_turn_sec if action != self.heading else self.time_straight_sec
        k = self.config
        reward = k.coverage_reward_scale*new - k.objective_cost_scale*(k.tradeoff_beta*self.remaining_priority_mass/self.total_priority_mass + 1-k.tradeoff_beta)*dt/self.time_straight_sec
        if self.covered_cells_count + int(new) == self.target_cells_count:
            reward += k.success_terminal_reward
        elif self.steps+1 >= self.max_steps:
            reward -= k.failure_terminal_penalty
        return float(reward)

    def greedy_action(self) -> int:
        actions = np.flatnonzero(self.action_masks()).tolist()
        # Exact immediate-reward maximization; same reward -> straight -> U,R,D,L.
        return max(actions, key=lambda a: (self.immediate_reward(a), int(a == self.heading), -a))

    def _record_first_visit(self, pos: tuple[int, int]) -> bool:
        if not self.active_mask[pos] or self.covered_map[pos]:
            return False
        self.covered_map[pos] = True
        self.covered_cells_count += 1
        self.remaining_priority_mass -= float(self.weight_map[pos])
        self.first_visit_time[pos] = self.elapsed_time_sec
        self.first_visit_distance[pos] = self.path_length_m
        self.first_visit_step[pos] = self.steps
        self._dist_cache, self._dist_version = None, -1
        return True

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        if not self._ready:
            raise RuntimeError("Call reset() before step()")
        if self._done:
            raise RuntimeError("Episode ended; reset before another step")
        if self.complete_at_reset:
            self._done, self.termination_reason = True, "complete_at_reset"
            return self.observation(), 0.0, True, False, self.info(final=True)
        if not isinstance(action, (int, np.integer)) or isinstance(action, bool) or action not in range(4):
            raise ValueError("action must be an integer in 0..3")
        action = int(action)
        mask, phase = self.mask_details()
        if not mask[action]:
            raise ProgressInvariantError(f"Attempted masked-out action {action}, mask={mask.tolist()}, map={self.current_map_name}, pos={self.drone_pos}")
        r, c = self.drone_pos
        dr, dc = MOVES[action]
        next_pos = (r+dr, c+dc)
        turned = action != self.heading  # INCLUDES the first action vs initial heading
        dt = self.time_turn_sec if turned else self.time_straight_sec
        before_mass = self.remaining_priority_mass  # pre-transition W_t is essential
        if phase == "progress":
            dist = self.distance_field()
            if int(dist[next_pos]) != int(dist[self.drone_pos])-1:
                raise ProgressInvariantError("BFS distance did not decrease exactly by one")
        self.mask_counts[int(mask.sum())] += 1  # count an EXECUTED decision, not mask API calls
        self.direct_steps += int(phase == "direct")
        self.progress_steps += int(phase == "progress")
        self.steps += 1
        self.valid_move_count += 1
        self.move_turn_count += int(turned)
        self.move_straight_count += int(not turned)
        self.elapsed_time_sec += dt
        self.path_length_m = self.valid_move_count * self.config.cell_size_m
        self.weighted_waiting_sum += before_mass*dt
        self.collected_priority_area += (1-before_mass/self.total_priority_mass)*dt
        self.revisit_steps += int(self.active_mask[next_pos] and self.covered_map[next_pos])
        self.revisit_progress_count += int(phase == "progress" and self.active_mask[next_pos] and self.covered_map[next_pos])
        self.zero_transit_steps += int(not self.active_mask[next_pos])
        self.oscillation_steps += int(self.prev_pos is not None and next_pos == self.prev_pos)
        self.prev_pos, self.drone_pos = self.drone_pos, next_pos
        self.heading = action
        self.visited_map[next_pos] = True
        self.position_history *= self.config.history_alpha
        self.position_history[next_pos] = 1.0
        new = self._record_first_visit(next_pos)
        self.new_cover_count += int(new)
        k = self.config
        pieces = {
            "coverage": k.coverage_reward_scale * int(new),
            "latency": -k.objective_cost_scale*k.tradeoff_beta*(before_mass/self.total_priority_mass)*(dt/self.time_straight_sec),
            "mission_time": -k.objective_cost_scale*(1-k.tradeoff_beta)*(dt/self.time_straight_sec),
            "terminal": 0.0,
        }
        if self.covered_cells_count == self.target_cells_count:
            self._done, self.termination_reason = True, "complete"
            pieces["terminal"] = k.success_terminal_reward
        elif self.steps >= self.max_steps:
            self._done, self.termination_reason = True, "budget_exhausted"
            pieces["terminal"] = -k.failure_terminal_penalty
        reward = float(sum(pieces.values()))
        self.total_return += reward
        for name, value in pieces.items():
            self.reward_sums[name] += value
        self._mask_cache = None
        info = self.info(final=self._done)
        info.update({f"step_reward_{name}": float(value) for name, value in pieces.items()})
        info.update(mask_phase=phase, mask_cardinality=int(mask.sum()),
                    remaining_priority_before=before_mass, traversal_time_sec=dt,
                    is_new_active_cell=int(new), is_turn=int(turned))
        # Budget is a task-terminal failure, NOT TimeLimit.truncated for bootstrapping.
        return self.observation(), reward, self._done, False, info

    def exact_metrics(self) -> dict[str, Any]:
        w = self.weight_map[self.active_mask].astype(np.float64)
        times = self.first_visit_time[self.active_mask]
        distances = self.first_visit_distance[self.active_mask]
        reached = np.isfinite(times)
        success = bool(reached.all())
        lw = float(np.dot(w, times)/self.total_priority_mass) if success else math.nan
        dw = float(np.dot(w, distances)/self.total_priority_mass) if success else math.nan
        auc = self.collected_priority_area/self.elapsed_time_sec if self.elapsed_time_sec > 0 else (1.0 if success else 0.0)
        # Explicitly censored waiting time, NEVER called complete-trajectory L_w.
        censored = np.where(reached, times, self.elapsed_time_sec)
        weighted_censored = float(np.dot(w, censored)/self.total_priority_mass)
        out: dict[str, Any] = dict(
            is_success=int(success), coverage_pct=100.0*self.covered_cells_count/self.target_cells_count,
            priority_weighted_first_visit_latency_sec=lw,
            priority_weighted_first_visit_distance_m=dw,
            censored_weighted_waiting_time_sec=weighted_censored,
            cumulative_priority_auc_normalized=float(auc),
            covered_priority_ratio=1-self.remaining_priority_mass/self.total_priority_mass,
            first_visit_count=int(reached.sum()),
            weighted_wait_identity_abs_error=abs(self.weighted_waiting_sum-self.total_priority_mass*weighted_censored),
            auc_identity_abs_error=abs(auc-(1-lw/self.elapsed_time_sec)) if success and self.elapsed_time_sec > 0 else math.nan,
            success_conditioned_mission_time_sec=self.elapsed_time_sec if success else math.nan,
            failure_penalized_mission_time_sec=self.elapsed_time_sec if success else self.max_steps*max(self.time_straight_sec, self.time_turn_sec),
        )
        idx = np.argsort(times[reached], kind="stable")
        ts, ws = times[reached][idx], w[reached][idx]
        masses = np.cumsum(ws)
        for fraction in (.25, .50, .75, .90):
            i = int(np.searchsorted(masses, fraction*self.total_priority_mass, side="left"))
            out[f"priority_arrival_{int(100*fraction)}_sec"] = float(ts[i]) if i < len(ts) else math.nan
        for p in range(1, 6):
            group = self.weight_map == p
            n = int(group.sum())
            gt = self.first_visit_time[group]
            nr = int(np.isfinite(gt).sum())
            out[f"p{p}_cell_count"] = n
            out[f"p{p}_visited_count"] = nr
            out[f"p{p}_first_visit_mean_sec"] = float(gt.mean()) if n and nr == n else math.nan
        return out

    def info(self, final: bool = False) -> dict[str, Any]:
        success = self.covered_cells_count == self.target_cells_count
        out: dict[str, Any] = dict(
            version=VERSION, map_name=self.current_map_name, start_r=self.start_pos[0], start_c=self.start_pos[1],
            start_heading=self.start_heading, is_success=int(success),
            termination_reason="complete_at_reset" if self.complete_at_reset else self.termination_reason,
            already_complete=self.complete_at_reset, coverage_percent=self.covered_cells_count/self.target_cells_count,
            coverage_pct=100*self.covered_cells_count/self.target_cells_count,
            active_cells=self.target_cells_count, active_ratio=self.target_cells_count/self.bbox_area,
            bbox_area=self.bbox_area, steps=self.steps, attempted_action_count=self.steps,
            valid_move_count=self.valid_move_count, invalid_action_count=self.invalid_action_count,
            move_straight_count=self.move_straight_count, turn_count=self.move_turn_count,
            move_turn_count=self.move_turn_count, revisit_steps=self.revisit_steps,
            revisit_progress_count=self.revisit_progress_count, zero_transit_steps=self.zero_transit_steps,
            oscillation_steps=self.oscillation_steps, new_cover_count=self.new_cover_count,
            max_steps=self.max_steps, mission_time_sec=self.elapsed_time_sec, path_length_m=self.path_length_m,
            total_priority_mass=self.total_priority_mass, remaining_priority_mass=self.remaining_priority_mass,
            direct_steps=self.direct_steps, progress_steps=self.progress_steps,
            total_reward=self.total_return, time_straight_sec=self.time_straight_sec,
            time_turn_sec=self.time_turn_sec, progress_violation_count=0,
        )
        out.update({f"reward_sum_{key}": float(val) for key, val in self.reward_sums.items()})
        out["reward_sum_abs_error"] = abs(self.total_return-sum(self.reward_sums.values()))
        out.update({f"mask_k{k}_count": int(self.mask_counts[k]) for k in range(1, 5)})
        out["single_action_mask_ratio"] = float(self.mask_counts[1]/self.steps) if self.steps else math.nan
        if final:
            out.update(self.exact_metrics())
        return out

    def assert_consistent(self) -> None:
        if self.valid_move_count != self.move_straight_count+self.move_turn_count:
            raise AssertionError("Movement counters disagree")
        if self.valid_move_count != self.steps or self.mask_counts.sum() != self.steps:
            raise AssertionError("Attempt / mask decision accounting disagrees")
        if self.covered_cells_count != int(np.isfinite(self.first_visit_time[self.active_mask]).sum()):
            raise AssertionError("Last or initial first-visit record missing")
        if self.covered_cells_count != self.initial_covered_count+self.new_cover_count:
            raise AssertionError("Coverage counting disagreement")
        if self.steps != self.new_cover_count+self.revisit_steps+self.zero_transit_steps:
            raise AssertionError("New/revisit/transit movement partition disagrees")
        nominal = self.move_straight_count*self.time_straight_sec+self.move_turn_count*self.time_turn_sec
        if not math.isclose(self.elapsed_time_sec, nominal, rel_tol=1e-9, abs_tol=1e-8):
            raise AssertionError("Mission-time accounting mismatch")
        if not math.isclose(self.total_return, sum(self.reward_sums.values()), rel_tol=1e-8, abs_tol=1e-7):
            raise AssertionError("Returned reward and reward sums disagree")
        m = self.exact_metrics()
        if m["weighted_wait_identity_abs_error"] > 1e-7*max(1, abs(self.weighted_waiting_sum)):
            raise AssertionError("Pre-action weighted waiting-time identity failed")
        if not math.isclose(self.remaining_priority_mass, float(self.weight_map[self.uncovered_mask()].sum())):
            raise AssertionError("Remaining priority sum disagrees")

    def snapshot(self) -> dict[str, np.ndarray]:
        """NPZ-safe arrays, including NaN first visits for unvisited/non-active cells."""
        return dict(priority_map=self.weight_map.copy(), covered_map=self.covered_map.copy(),
                    first_visit_time_sec=self.first_visit_time.copy(),
                    first_visit_distance_m=self.first_visit_distance.copy(),
                    first_visit_step=self.first_visit_step.copy(),
                    current_position=np.asarray(self.drone_pos), current_heading=np.asarray(self.heading))
