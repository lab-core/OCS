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
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from udagger import Label, aggregate_votes, one_hot
from udagger.experts import ExpertSpec

from udagger_adwords.simulator import SKIP, State, feasible_mask, sample_future_counts
from udagger_adwords.instance import Market
from udagger_adwords.solver import solve_count_mip

if TYPE_CHECKING:
    from udagger_adwords.problem import AdwordsEpisode

EXPERT_KINDS = ("greedy", "msvv", "full", "det", "agg", "saa")


def _rel_gap(obj: float, bound: float) -> float:
    return abs(bound - obj) / max(abs(obj), 1e-9)


@dataclass
class Expert(ExpertSpec):
    """An AdWords expert (settings in :class:`udagger.experts.ExpertSpec`)."""

    KINDS: ClassVar[tuple[str, ...]] = EXPERT_KINDS

    def __call__(
        self, episode: AdwordsEpisode, state: State, rng: np.random.Generator
    ) -> Label:
        """Label a state of an episode (``udagger.Expert`` interface).

        The full-information expert reads the realised future of the episode.
        """
        future = episode.sim.true_future_counts() if self.kind == "full" else None
        return self.decide(episode.market, state, rng, future)

    def decide(
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
            return Label(SKIP, one_hot(SKIP, size))
        bid = market.bids[:, state.query].astype(float)

        if self.kind == "greedy":
            score = np.where(mask[1:], bid, -np.inf)
            action = int(np.argmax(score)) + 1
            return Label(action, one_hot(action, size))
        if self.kind == "msvv":
            spent = 1.0 - state.budgets / market.budgets
            score = np.where(mask[1:], bid * (1.0 - np.exp(spent - 1.0)), -np.inf)
            action = int(np.argmax(score)) + 1
            return Label(action, one_hot(action, size))

        if self.kind == "full":
            if true_future is None:
                raise ValueError("the full-information expert needs the future")
            scenario_sets = [true_future[None, :]]
        elif self.kind == "saa":
            scenario_sets = [sample_future_counts(market, state, self.n_scenarios, rng)]
        else:
            counts = sample_future_counts(market, state, self.n_scenarios, rng)
            scenario_sets = [counts[s : s + 1] for s in range(counts.shape[0])]

        votes = []
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
            votes.append(res.action)
            solve_time += res.runtime
            gap = max(gap, _rel_gap(res.objective, res.bound))
        # Physician study: with sample_vote the executed action is drawn from
        # the votes, which uses one random draw even for single-solve experts.
        action, dist = aggregate_votes(votes, size, rng, sample=self.sample_vote)
        return Label(action, dist, solve_time, gap)
