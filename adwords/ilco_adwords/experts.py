"""Experts for the stochastic AdWords problem.

The experts follow the taxonomy of the paper (treatment of uncertainty):

* ``greedy`` and ``msvv``: myopic rules (highest bid, and the
  budget-discounted bid of Mehta, Saberi, Vazirani and Vazirani).
* ``full``: full-information (hindsight) expert, solving the count MIP on the
  realised future of the episode.
* ``det``: deterministic expert, solving the count MIP on one scenario sampled
  from the posterior.
* ``agg``: aggregated deterministic expert, solving ``K`` deterministic MIPs
  independently and returning the empirical distribution of their decisions.
* ``saa``: two-stage stochastic expert, solving the sample average
  approximation with ``K`` scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ilco_adwords.env import SKIP, State, feasible_mask, sample_future_counts
from ilco_adwords.market import Market
from ilco_adwords.solver import solve_count_mip

EXPERT_KINDS = ("greedy", "msvv", "full", "det", "agg", "saa")


@dataclass
class Label:
    """Output of an expert query.

    Attributes:
        action: Action recommended by the expert.
        dist: Distribution over actions used as (soft) training target.
        solve_time: Seconds spent solving optimisation models.
        gap: Largest relative MIP gap of the solved models.
    """

    action: int
    dist: np.ndarray
    solve_time: float = 0.0
    gap: float = 0.0


def _one_hot(action: int, size: int) -> np.ndarray:
    dist = np.zeros(size)
    dist[action] = 1.0
    return dist


def _rel_gap(obj: float, bound: float) -> float:
    return abs(bound - obj) / max(abs(obj), 1e-9)


@dataclass
class Expert:
    """A configurable expert.

    Attributes:
        kind: One of ``EXPERT_KINDS``.
        n_scenarios: Number of scenarios for ``agg`` and ``saa``.
        mip_gap: Relative MIP gap stopping criterion.
        time_limit: Time limit per MIP in seconds.
        sample_vote: Whether the aggregated expert acts by sampling from its
            votes (as in the physician study) instead of taking the majority.
    """

    kind: str
    n_scenarios: int = 1
    mip_gap: float = 0.0
    time_limit: float = 10.0
    sample_vote: bool = False

    def __post_init__(self) -> None:
        """Validate the configuration."""
        if self.kind not in EXPERT_KINDS:
            raise ValueError(f"unknown expert {self.kind!r}")
        if self.kind in ("full", "det"):
            self.n_scenarios = 1

    @property
    def name(self) -> str:
        """Short identifier used in file names and tables."""
        if self.kind in ("agg", "saa"):
            return f"{self.kind}{self.n_scenarios}"
        return self.kind

    def __call__(
        self,
        market: Market,
        state: State,
        rng: np.random.Generator,
        true_future: np.ndarray | None = None,
    ) -> Label:
        """Query the expert in a state.

        Args:
            market: The market.
            state: Current state.
            rng: Random generator used for scenario sampling.
            true_future: Realised future type counts (required by ``full``).

        Returns:
            The expert label.

        Raises:
            ValueError: If ``full`` is queried without the realised future.
        """
        size = market.n_adv + 1
        mask = feasible_mask(market, state)
        if not mask[1:].any():
            return Label(SKIP, _one_hot(SKIP, size))
        bid = market.bids[:, state.query].astype(float)

        if self.kind == "greedy":
            score = np.where(mask[1:], bid, -np.inf)
            action = int(np.argmax(score)) + 1
            return Label(action, _one_hot(action, size))
        if self.kind == "msvv":
            spent = 1.0 - state.budgets / market.budgets
            score = np.where(mask[1:], bid * (1.0 - np.exp(spent - 1.0)), -np.inf)
            action = int(np.argmax(score)) + 1
            return Label(action, _one_hot(action, size))

        if self.kind == "full":
            if true_future is None:
                raise ValueError("the full-information expert needs the future")
            scenario_sets = [true_future[None, :]]
        elif self.kind == "saa":
            scenario_sets = [sample_future_counts(market, state, self.n_scenarios, rng)]
        else:
            counts = sample_future_counts(market, state, self.n_scenarios, rng)
            scenario_sets = [counts[s : s + 1] for s in range(counts.shape[0])]

        dist = np.zeros(size)
        solve_time = 0.0
        gap = 0.0
        for scenarios in scenario_sets:
            res = solve_count_mip(
                market,
                state.budgets,
                state.query,
                scenarios,
                mip_gap=self.mip_gap,
                time_limit=self.time_limit,
            )
            dist[res.action] += 1.0
            solve_time += res.runtime
            gap = max(gap, _rel_gap(res.objective, res.bound))
        dist /= dist.sum()
        if self.sample_vote:
            # Physician study: the executed action is drawn from the votes.
            action = int(rng.choice(size, p=dist))
        else:
            action = int(np.argmax(dist))
        return Label(action, dist, solve_time, gap)
