"""Global configuration for the warehouse repair project.

A single :class:`Config` dataclass holds every tunable.  The fields are the
union of both specification drafts (v1 "executable specification" and v2
"POCL + plan modification"); see ``SPEC.md`` sections 4 and 0.1.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

# Modes of the repair loop.  ``negotiate`` is the main algorithm, ``self_only``
# and ``global`` are baselines, ``timeindexed`` is the v1 ablation.
VALID_MODES = ("negotiate", "self_only", "global", "timeindexed")


@dataclass
class Config:
    # ---- world / simulation ------------------------------------------
    H: int = 30
    W: int = 30
    n_agents: int = 20
    tasks_per_agent: int = 3
    seed: int = 0
    T_max: int = 600

    # ---- negotiation / repair rules ----------------------------------
    comm_radius: int = 6
    max_depth: int = 3
    lambda_soft: int = 15
    delay_threshold: int = 8

    # ---- disruptions -------------------------------------------------
    obstacle_density: float = 0.05
    n_breakdowns: int = 2
    n_emergencies: int = 2
    breakdown_duration_range: Tuple[int, int] = (5, 15)
    p_permanent_breakdown: float = 0.2
    block_duration_range: Tuple[int, int] = (5, 25)

    # ---- plan-space search budgets (v2) ------------------------------
    pocl_max_nodes: int = 20_000
    repair_pocl_max_nodes: int = 3_000
    k_candidates: int = 3
    beta_alter: int = 10
    failure_wait: int = 60

    # ---- mode --------------------------------------------------------
    replan_mode: str = "negotiate"

    def __post_init__(self) -> None:
        if self.replan_mode not in VALID_MODES:
            raise ValueError(
                f"replan_mode must be one of {VALID_MODES}, got {self.replan_mode!r}"
            )
        if self.H < 6 or self.W < 6:
            raise ValueError("H and W must be at least 6")
        if self.n_agents < 1:
            raise ValueError("n_agents must be >= 1")
        if self.tasks_per_agent < 0:
            raise ValueError("tasks_per_agent must be >= 0")
        if self.comm_radius < 1:
            raise ValueError("comm_radius must be >= 1")
        if self.T_max < 1:
            raise ValueError("T_max must be >= 1")
        if self.obstacle_density < 0:
            raise ValueError("obstacle_density must be >= 0")

    # ------------------------------------------------------------------
    @property
    def n_tasks(self) -> int:
        """Total number of tasks in the scenario."""
        return self.n_agents * self.tasks_per_agent

    def n_blockages(self, n_free_cells: int) -> int:
        """Number of blockage events implied by ``obstacle_density``."""
        return int(round(self.obstacle_density * n_free_cells))

    def replace(self, **kwargs) -> "Config":
        """Return a copy with the given fields replaced (like dataclasses.replace)."""
        from dataclasses import replace as _replace

        return _replace(self, **kwargs)
