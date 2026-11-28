"""Episode simulator, state representation and scenario sampling.

At decision epoch ``k`` the decision maker observes the type of the arriving
query, its arrival time, the remaining budget of every advertiser and the
number of queries of each class observed so far. It either assigns the query
to an advertiser whose remaining budget covers the bid (collecting the bid as
revenue) or skips it. Action ``0`` is *skip* and action ``i + 1`` assigns the
query to advertiser ``i``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln
from scipy.stats import beta as beta_dist

from udagger_adwords.instance import PREMIUM, REGULAR, Episode, Market

SKIP = 0


@dataclass
class State:
    """Observable state at a decision epoch.

    Attributes:
        k: Index of the current query in the episode.
        query: Type of the current query.
        time: Arrival time of the current query.
        budgets: Remaining budget of each advertiser.
        observed: Number of queries of each class observed so far, including
            the current one.
    """

    k: int
    query: int
    time: float
    budgets: np.ndarray
    observed: np.ndarray


def feasible_mask(market: Market, state: State) -> np.ndarray:
    """Return the feasibility mask over actions.

    Args:
        market: The market.
        state: Current state.

    Returns:
        Boolean vector of size ``n_adv + 1``; entry 0 (skip) is always True.
    """
    bid = market.bids[:, state.query]
    mask = np.empty(market.n_adv + 1, dtype=bool)
    mask[0] = True
    mask[1:] = (bid > 0) & (bid <= state.budgets)
    return mask


class Simulator:
    """Step-by-step simulator of one episode.

    Args:
        market: The market.
        episode: The realised query sequence.
    """

    def __init__(self, market: Market, episode: Episode) -> None:
        """Initialise the simulator at the first query of the episode."""
        self.market = market
        self.episode = episode
        self.budgets = market.budgets.copy()
        self.k = 0
        self.observed = np.zeros(2, dtype=int)
        self.revenue = 0
        self.served = np.zeros(2, dtype=int)
        self.actions: list[int] = []
        if len(episode) > 0:
            self.observed[market.type_class[episode.types[0]]] += 1

    @property
    def done(self) -> bool:
        """Whether all queries have been processed."""
        return self.k >= len(self.episode)

    def state(self) -> State:
        """Return the current observable state."""
        return State(
            k=self.k,
            query=int(self.episode.types[self.k]),
            time=float(self.episode.times[self.k]),
            budgets=self.budgets.copy(),
            observed=self.observed.copy(),
        )

    def step(self, action: int) -> int:
        """Apply an action to the current query and move to the next one.

        Args:
            action: 0 to skip, ``i + 1`` to assign to advertiser ``i``.

        Returns:
            The revenue collected by the action.

        Raises:
            ValueError: If the action is infeasible.
        """
        q = int(self.episode.types[self.k])
        reward = 0
        if action != SKIP:
            i = action - 1
            bid = int(self.market.bids[i, q])
            if bid <= 0 or bid > self.budgets[i]:
                raise ValueError(f"infeasible action {action} for query {q}")
            self.budgets[i] -= bid
            reward = bid
            self.served[self.market.type_class[q]] += 1
        self.revenue += reward
        self.actions.append(action)
        self.k += 1
        if not self.done:
            nxt = int(self.episode.types[self.k])
            self.observed[self.market.type_class[nxt]] += 1
        return reward

    def true_future_counts(self) -> np.ndarray:
        """Return the realised counts of each type after the current query."""
        future = self.episode.types[self.k + 1 :]
        return np.bincount(future, minlength=self.market.n_types)


def remaining_count_posterior(
    market: Market, cls: int, time: float, m: int
) -> np.ndarray:
    """Posterior pmf of the number of future queries of a class.

    Given that ``m`` queries of class ``cls`` arrived in ``[0, time]``, the
    total number ``N`` of queries of the class has posterior proportional to
    ``prior(N) * Binomial(m; N, F(time))`` where ``F`` is the arrival-time
    cdf of the class.

    Args:
        market: The market.
        cls: Query class.
        time: Current time in [0, 1].
        m: Number of queries of the class observed so far.

    Returns:
        Probability vector over the number of remaining queries ``N - m``.
    """
    prior = market.count_pmf(cls)
    a, b = market.beta_params[cls]
    surv = 1.0 - beta_dist.cdf(time, a, b)
    support = np.arange(prior.size)
    valid = support >= m
    if surv <= 1e-12 or not valid.any():
        return np.array([1.0])
    n = support[valid]
    logp = np.log(prior[valid] + 1e-300)
    logp += gammaln(n + 1) - gammaln(n - m + 1) + (n - m) * np.log(surv)
    p = np.exp(logp - logp.max())
    return p / p.sum()


def sample_future_counts(
    market: Market, state: State, n_scenarios: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample future query-type counts from the posterior.

    Args:
        market: The market.
        state: Current state.
        n_scenarios: Number of scenarios.
        rng: Random generator.

    Returns:
        Integer matrix of shape (n_scenarios, n_types).
    """
    counts = np.zeros((n_scenarios, market.n_types), dtype=int)
    for cls in (REGULAR, PREMIUM):
        post = remaining_count_posterior(
            market, cls, state.time, int(state.observed[cls])
        )
        remaining = rng.choice(post.size, size=n_scenarios, p=post)
        idx = np.flatnonzero(market.type_class == cls)
        probs = market.type_probs[idx] / market.type_probs[idx].sum()
        for s in range(n_scenarios):
            counts[s, idx] = rng.multinomial(remaining[s], probs)
    return counts
