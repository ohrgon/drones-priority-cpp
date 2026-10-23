"""Isolated B=6 Bounded-Detour core.

Keeps the original six spatial channels and heading.
Adds exactly one scalar observation:
    detour_budget = remaining_budget / B
No mission-time observation is added.

This module does NOT modify core_ch4.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from coverage_core import EnvConfig, CoverageCore, MOVES, ProgressInvariantError
from core_ch4 import ExperimentCore as _BaseExperimentCore
from core_ch4 import ExperimentEnvConfig as _BaseExperimentEnvConfig


@dataclass(frozen=True)
class ExperimentEnvConfig(_BaseExperimentEnvConfig):
    """Original ExperimentEnvConfig + bounded-detour parameter."""
    detour_budget_max: int = 6

    def __post_init__(self):
        # Skip the original mask-mode validator because it does not know "relaxed".
        EnvConfig.__post_init__(self)

        if self.mask_mode not in {"none", "boundary", "progress", "relaxed"}:
            raise ValueError(
                "mask_mode must be none, boundary, progress, or relaxed"
            )
        if not np.isfinite(self.invalid_action_penalty) or self.invalid_action_penalty < 0:
            raise ValueError("invalid_action_penalty must be nonnegative")
        if self.invalid_time_rule not in {
            "straight_control_interval",
            "attempted_turn_interval",
        }:
            raise ValueError(
                "Choose an explicit invalid_time_rule; zero-time stays are not silently allowed"
            )
        if self.invalid_heading_rule not in {"preserve", "attempted_action"}:
            raise ValueError(
                "invalid_heading_rule must be preserve or attempted_action"
            )
        if (
            isinstance(self.detour_budget_max, bool)
            or not isinstance(self.detour_budget_max, int)
            or self.detour_budget_max < 0
        ):
            raise ValueError("detour_budget_max must be a nonnegative int")


class ExperimentCore(_BaseExperimentCore):
    """Bounded-detour extension of the existing Chapter-4 core."""

    def __init__(self, config: Optional[ExperimentEnvConfig] = None):
        super().__init__(config or ExperimentEnvConfig())

    def reset(self, *args, **kwargs):
        # Must exist before CoverageCore.reset() calls self.observation().
        self.detour_budget = (
            int(self.config.detour_budget_max)
            if self.config.mask_mode == "relaxed"
            else 0
        )
        self.detour_action_count = 0
        self.strict_action_selected_count = 0
        self.budget_reset_count = 0
        self.budget_exhaustion_count = 0
        return super().reset(*args, **kwargs)

    def observation(self):
        obs = super().observation()

        if (
            self.config.mask_mode == "relaxed"
            and self.config.detour_budget_max > 0
        ):
            value = (
                float(self.detour_budget)
                / float(self.config.detour_budget_max)
            )
        else:
            # Same network input key exists for every mask variant.
            value = 0.0

        obs["detour_budget"] = np.asarray(
            [value], dtype=np.float32
        )
        return obs

    def strict_mask_details_no_cache(self):
        """Current Strict Progress mask from the pre-action state."""
        if not self._ready:
            raise RuntimeError(
                "reset must precede mask computation"
            )
        if self.complete_at_reset or self._done:
            return np.ones(4, dtype=bool), "terminal_sentinel"

        r, c = self.drone_pos
        valid = self.feasible_mask()

        direct = np.zeros(4, dtype=bool)
        for a, (dr, dc) in enumerate(MOVES):
            if not valid[a]:
                continue
            nr, nc = r + dr, c + dc
            direct[a] = bool(
                self.active_mask[nr, nc]
                and not self.covered_map[nr, nc]
            )

        if direct.any():
            return direct, "direct"

        dist = self.distance_field()
        cur_d = int(dist[r, c])

        progress = np.zeros(4, dtype=bool)
        for a, (dr, dc) in enumerate(MOVES):
            if not valid[a]:
                continue
            nr, nc = r + dr, c + dc
            progress[a] = (
                int(dist[nr, nc]) == cur_d - 1
            )

        if not progress.any():
            raise ProgressInvariantError(
                f"No strict progress action: "
                f"map={self.current_map_name}, "
                f"pos={self.drone_pos}, "
                f"remaining="
                f"{self.target_cells_count-self.covered_cells_count}"
            )

        return progress, "progress"

    def mask_details(self):
        if self.config.mask_mode != "relaxed":
            return super().mask_details()

        if not self._ready:
            raise RuntimeError(
                "reset must precede mask computation"
            )
        if self.complete_at_reset or self._done:
            return np.ones(4, dtype=bool), "terminal_sentinel"

        # Budget remains -> all boundary-valid actions are available.
        if self.detour_budget > 0:
            m = self.feasible_mask()
            if not m.any():
                raise ProgressInvariantError(
                    "Nonterminal feasible graph has no edge"
                )
            return m, "relaxed"

        # Budget exhausted -> Strict Progress is forced.
        return self.strict_mask_details_no_cache()

    def step(self, action):
        # Strict membership MUST be defined before action execution.
        strict_before = None
        budget_before = int(
            getattr(self, "detour_budget", 0)
        )

        if self.config.mask_mode == "relaxed":
            strict_before, _ = (
                self.strict_mask_details_no_cache()
            )

        _, reward, terminated, truncated, info = (
            super().step(action)
        )

        is_strict = True
        is_detour = False

        if self.config.mask_mode == "relaxed":
            action = int(action)
            is_strict = bool(strict_before[action])
            is_detour = not is_strict

            if is_detour:
                self.detour_action_count += 1
            else:
                self.strict_action_selected_count += 1

            new_coverage = bool(
                info.get("is_new_active_cell", 0)
            )

            # Reset ONLY when a new active cell is first visited.
            if new_coverage:
                self.detour_budget = int(
                    self.config.detour_budget_max
                )
                self.budget_reset_count += 1
            elif is_detour:
                self.detour_budget = max(
                    0, budget_before - 1
                )
                if (
                    budget_before > 0
                    and self.detour_budget == 0
                ):
                    self.budget_exhaustion_count += 1
            else:
                self.detour_budget = budget_before

        # super().step() created its observation before the budget update.
        # Return a fresh observation with the updated normalized budget.
        obs = self.observation()

        info.update(
            detour_budget_before=budget_before,
            detour_budget_after=int(
                getattr(self, "detour_budget", 0)
            ),
            is_strict_action=int(is_strict),
            is_detour_action=int(is_detour),
            detour_action_count=int(
                getattr(self, "detour_action_count", 0)
            ),
            strict_action_selected_count=int(
                getattr(
                    self,
                    "strict_action_selected_count",
                    0,
                )
            ),
            budget_reset_count=int(
                getattr(self, "budget_reset_count", 0)
            ),
            budget_exhaustion_count=int(
                getattr(
                    self,
                    "budget_exhaustion_count",
                    0,
                )
            ),
        )
        return obs, reward, terminated, truncated, info

    def info(self, final=False):
        out = super().info(final)
        out.update(
            detour_budget_max=int(
                self.config.detour_budget_max
            ),
            detour_budget_remaining=int(
                getattr(self, "detour_budget", 0)
            ),
            detour_action_count=int(
                getattr(self, "detour_action_count", 0)
            ),
            strict_action_selected_count=int(
                getattr(
                    self,
                    "strict_action_selected_count",
                    0,
                )
            ),
            budget_reset_count=int(
                getattr(self, "budget_reset_count", 0)
            ),
            budget_exhaustion_count=int(
                getattr(
                    self,
                    "budget_exhaustion_count",
                    0,
                )
            ),
        )
        return out

    def assert_consistent(self):
        super().assert_consistent()
        if self.config.mask_mode == "relaxed":
            assert self.invalid_action_count == 0
            assert (
                0
                <= self.detour_budget
                <= self.config.detour_budget_max
            )
            if self.config.detour_budget_max == 0:
                assert self.detour_action_count == 0


def make_core(config):
    if config.paper_mode != "none":
        raise ValueError(
            "Bounded-detour 4-way experiment expects "
            "paper_mode='none'."
        )
    return ExperimentCore(config)
