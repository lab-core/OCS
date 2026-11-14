"""The stochastic AdWords problem as a ``udagger`` problem."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from udagger_adwords.env import Simulator, State, feasible_mask
from udagger_adwords.features import features
from udagger_adwords.market import PREMIUM, REGULAR, Market, sample_episode


class AdwordsEpisode:
    """One AdWords episode (``udagger.Episode`` interface).

    Args:
        market: The market.
        seed: Episode seed (defines the query sequence).
    """

    def __init__(self, market: Market, seed: int) -> None:
        """Sample the query sequence and start the simulator."""
        self.market = market
        self.seed = seed
        self.queries = sample_episode(market, np.random.default_rng(seed))
        self.sim = Simulator(market, self.queries)

    @property
    def done(self) -> bool:
        """Whether all queries have been processed."""
        return self.sim.done

    def observe(self) -> State:
        """Return the current state."""
        return self.sim.state()

    def features(self, state: State) -> np.ndarray:
        """Return the feature vector of a state."""
        return features(self.market, state)

    def mask(self, state: State) -> np.ndarray:
        """Return the feasibility mask (action 0 skips the query)."""
        return feasible_mask(self.market, state)

    def step(self, action: int) -> None:
        """Serve or skip the current query."""
        self.sim.step(action)

    def info(self, state: State) -> dict[str, float]:
        """Arrival time and class of the query, stored with the labels."""
        return {
            "epoch_time": state.time,
            "query_class": int(self.market.type_class[state.query]),
        }

    def stats(self) -> dict[str, float]:
        """Revenue (the objective) and service statistics."""
        cls = self.market.type_class[self.queries.types]
        return {
            "objective": float(self.sim.revenue),
            "n_regular": int(np.sum(cls == REGULAR)),
            "n_premium": int(np.sum(cls == PREMIUM)),
            "served_regular": int(self.sim.served[REGULAR]),
            "served_premium": int(self.sim.served[PREMIUM]),
            "budget_used": float(
                1.0 - self.sim.budgets.sum() / self.market.budgets.sum()
            ),
        }


@dataclass
class AdwordsProblem:
    """Factory of AdWords episodes (``udagger.Problem`` interface).

    Attributes:
        market: The market.
    """

    market: Market
    maximize: bool = True

    @property
    def n_actions(self) -> int:
        """Skip plus one action per advertiser."""
        return self.market.n_adv + 1

    def episode(self, seed: int) -> AdwordsEpisode:
        """Create the episode of a seed."""
        return AdwordsEpisode(self.market, seed)
