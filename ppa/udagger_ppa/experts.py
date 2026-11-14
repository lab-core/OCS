"""Experts of the physician-to-patient assignment problem.

The experts follow the taxonomy of the paper (treatment of uncertainty):

* ``greedy``: preferred physician if possible, else a random eligible one;
* ``full``: full-information expert, solving the deterministic MIP on the
  realised future of the episode;
* ``det``: deterministic expert, solving it on one future sampled from the
  posterior of the remaining requests;
* ``agg``: aggregated deterministic expert, solving ``K`` such models
  independently and returning the distribution of their decisions;
* ``saa``: two-stage stochastic expert with ``K`` scenarios.

The models are built and solved exactly as by ``IO_learning`` in the
original scripts (``legacy/Learn_Module.py``); as there, the executed action
is drawn from the votes of the solved models.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from udagger import Label, aggregate_votes, one_hot

from udagger_ppa.features import blocked
from udagger_ppa.instance import Instance, make_scheduler, make_stochastic

if TYPE_CHECKING:
    from udagger_ppa.problem import PPAEpisode

EXPERT_KINDS = ("greedy", "full", "det", "agg", "saa")
# Scenario models of an episode are seeded with offset + 1000 * episode seed
# (+ j for the j-th deterministic model), above every episode seed.
EXPERT_SEED_OFFSET = 10**12


@dataclass
class Expert:
    """A configurable expert.

    Attributes:
        kind: One of ``EXPERT_KINDS``.
        instance: Instance parameters (used to build the scenario models).
        n_scenarios: Number of scenarios for ``agg`` and ``saa``.
        mip_gap: Relative MIP gap stopping criterion.
        time_limit: Time limit per MIP in seconds.
        sample_vote: Draw the executed action from the votes (original
            study) instead of taking the majority.
    """

    kind: str
    instance: Instance = field(default_factory=Instance)
    n_scenarios: int = 1
    mip_gap: float = 0.02
    time_limit: float = 30.0
    sample_vote: bool = True

    def __post_init__(self) -> None:
        """Validate the configuration."""
        if self.kind not in EXPERT_KINDS:
            raise ValueError(f"unknown expert {self.kind!r}")
        if self.kind in ("greedy", "full", "det"):
            self.n_scenarios = 1
        self._models: list = []
        self._episode: tuple[int, int] | None = None

    def __getstate__(self) -> dict:
        """Pickle the configuration only (models are rebuilt per process)."""
        state = self.__dict__.copy()
        state.update(_models=[], _episode=None)
        return state

    @property
    def name(self) -> str:
        """Short identifier used in file names and tables."""
        if self.kind in ("agg", "saa"):
            return f"{self.kind}{self.n_scenarios}"
        return self.kind

    def _setup(self, episode: PPAEpisode) -> None:
        """Build the scenario models of a new episode."""
        base = EXPERT_SEED_OFFSET + 1000 * episode.seed
        args = (self.time_limit, self.mip_gap)
        if self.kind == "saa":
            self._models = [
                make_stochastic(self.instance, base, self.n_scenarios, *args)
            ]
        else:
            self._models = [
                make_scheduler(self.instance, base + j, *args)
                for j in range(self.n_scenarios)
            ]

    def __call__(
        self, episode: PPAEpisode, state: dict, rng: np.random.Generator
    ) -> Label:
        """Label a state of an episode (``udagger.Expert`` interface)."""
        n_actions = self.instance.n_actions
        if self.kind == "greedy":
            return _greedy(episode, state, n_actions)
        key = (os.getpid(), episode.uid)
        if self._episode != key:
            self._setup(episode)
            self._episode = key
        votes, solve_time, gap = [], 0.0, 0.0
        for model in self._models:
            if self.kind == "full":
                model.regenerate_scenario(state, episode.sim.patients, 0)
            else:
                model.regenerate_scenario(state)
            if self.kind == "saa":
                model.build_model(state=state, new=1)
            else:
                new = 0 if self.kind == "full" else 1
                model.build_model(model.regenerated_patients, state, new=new)
            _, _, model_gap, sol_time = model.solve(tee=False)
            if self.kind == "saa":
                model.extract_solution()
            else:
                model.extract_solution(model.regenerated_patients)
            votes.append(int(model.get_current_action(0)))
            solve_time += float(sol_time or 0.0)
            gap = max(gap, float(model_gap or 0.0) / 100.0)
        action, dist = aggregate_votes(votes, n_actions, rng, sample=self.sample_vote)
        return Label(action, dist, solve_time, gap)


def _greedy(episode: PPAEpisode, state: dict, n_actions: int) -> Label:
    """Greedy rule; the target is uniform over the eligible physicians.

    The executed action is drawn by the simulator, as in the original study.
    """
    action = int(episode.sim.select_physician(state))
    n_phys = n_actions - 1
    ok = [p for p, b in enumerate(blocked(state, n_phys)) if not b]
    preferred = int(state["current_patient"].get("preferred_phys", 0))
    if preferred in ok or not ok:
        return Label(action, one_hot(action, n_actions))
    dist = np.zeros(n_actions)
    dist[ok] = 1.0 / len(ok)
    return Label(action, dist)
