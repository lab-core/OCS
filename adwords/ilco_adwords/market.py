"""Market definition and episode generation for the stochastic AdWords problem.

A *market* is fixed across episodes: a set of advertisers with integral
budgets, a set of query types with integral bids, and the stochastic arrival
model. An *episode* is a random, time-ordered sequence of queries drawn from
the arrival model.

Query types are split into two classes. *Regular* queries are numerous, carry
low bids and arrive uniformly over the horizon. *Premium* queries are rarer,
carry high bids, are only of interest to a few advertisers and tend to arrive
late. The number of queries of each class is random, which is the source of
uncertainty that a good policy must anticipate: spending the budget of
premium-capable advertisers on early regular queries may leave late premium
queries unserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

REGULAR = 0
PREMIUM = 1


@dataclass(frozen=True)
class Market:
    """Static description of an AdWords market.

    Attributes:
        bids: Integer matrix of shape (n_adv, n_types); 0 means no bid.
        budgets: Integer vector of shape (n_adv,) with initial budgets.
        type_class: Vector of shape (n_types,) with the class of each type.
        type_probs: Vector of shape (n_types,) with the probability of each
            type within its class (sums to one per class).
        count_mean: Mean number of queries per class, shape (2,).
        count_std: Standard deviation of the number of queries per class.
        beta_params: Beta distribution parameters (a, b) of the arrival time
            of each class, shape (2, 2).
        max_count: Upper bound of the support of the count distributions.
    """

    bids: np.ndarray
    budgets: np.ndarray
    type_class: np.ndarray
    type_probs: np.ndarray
    count_mean: np.ndarray
    count_std: np.ndarray
    beta_params: np.ndarray
    max_count: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=int))

    @property
    def n_adv(self) -> int:
        """Number of advertisers."""
        return int(self.bids.shape[0])

    @property
    def n_types(self) -> int:
        """Number of query types."""
        return int(self.bids.shape[1])

    def count_pmf(self, cls: int) -> np.ndarray:
        """Return the prior pmf of the number of queries of a class.

        The pmf is a discretised normal distribution truncated to
        ``{0, ..., max_count[cls]}``.

        Args:
            cls: Query class (``REGULAR`` or ``PREMIUM``).

        Returns:
            Probability vector indexed by the number of queries.
        """
        support = np.arange(int(self.max_count[cls]) + 1)
        logp = -0.5 * ((support - self.count_mean[cls]) / self.count_std[cls]) ** 2
        pmf = np.exp(logp - logp.max())
        return pmf / pmf.sum()


@dataclass(frozen=True)
class Episode:
    """A realised sequence of queries.

    Attributes:
        types: Query type of each arrival, shape (K,).
        times: Arrival time in [0, 1] of each arrival, non-decreasing.
    """

    types: np.ndarray
    times: np.ndarray

    def __len__(self) -> int:
        """Return the number of queries of the episode."""
        return int(self.types.shape[0])


def make_market(
    n_adv: int = 8,
    n_premium_adv: int = 3,
    n_regular_types: int = 8,
    n_premium_types: int = 4,
    bidders_per_regular: int = 3,
    count_mean: tuple[float, float] = (120.0, 30.0),
    count_std: tuple[float, float] = (20.0, 10.0),
    premium_beta: tuple[float, float] = (4.0, 1.0),
    tightness: float = 1.2,
    premium_tightness: float = 0.7,
    seed: int = 0,
) -> Market:
    """Build a random but reproducible AdWords market.

    Premium-capable advertisers bid high on premium types (6 to 10) and
    moderately on the regular types they are interested in (3 to 5), so that a
    greedy highest-bid rule is attracted to them. Other advertisers only bid on
    regular types (1 to 3). Every regular type has ``bidders_per_regular``
    bidders, at least one of which is a regular advertiser. The budget of a
    premium-capable advertiser is ``premium_tightness`` times its share of the
    expected premium spend; the budget of a regular advertiser is
    ``tightness`` times its share of the expected regular spend (in both
    cases, queries are split evenly among the bidders of the class).

    Args:
        n_adv: Number of advertisers.
        n_premium_adv: Number of advertisers bidding on premium types.
        n_regular_types: Number of regular query types.
        n_premium_types: Number of premium query types.
        bidders_per_regular: Number of bidders for each regular type.
        count_mean: Mean number of (regular, premium) queries.
        count_std: Standard deviation of the number of (regular, premium)
            queries.
        premium_beta: Beta parameters of premium arrival times.
        tightness: Budget scaling of regular advertisers.
        premium_tightness: Budget scaling of premium-capable advertisers.
        seed: Random seed.

    Returns:
        The generated market.
    """
    rng = np.random.default_rng(seed)
    n_types = n_regular_types + n_premium_types
    bids = np.zeros((n_adv, n_types), dtype=int)
    premium_adv = np.arange(n_premium_adv)
    regular_adv = np.arange(n_premium_adv, n_adv)

    for j in range(n_regular_types):
        first = rng.choice(regular_adv)
        others = rng.choice(
            np.setdiff1d(np.arange(n_adv), [first]),
            size=bidders_per_regular - 1,
            replace=False,
        )
        for i in np.concatenate(([first], others)):
            low, high = (3, 5) if i in premium_adv else (1, 3)
            bids[i, j] = rng.integers(low, high + 1)
    for j in range(n_regular_types, n_types):
        k = rng.integers(1, n_premium_adv + 1)
        for i in rng.choice(premium_adv, size=k, replace=False):
            bids[i, j] = rng.integers(6, 11)

    type_class = np.array([REGULAR] * n_regular_types + [PREMIUM] * n_premium_types)
    type_probs = np.zeros(n_types)
    type_probs[:n_regular_types] = rng.dirichlet(np.full(n_regular_types, 5.0))
    type_probs[n_regular_types:] = rng.dirichlet(np.full(n_premium_types, 5.0))

    expected = type_probs * np.asarray(count_mean)[type_class]
    budgets = np.zeros(n_adv, dtype=int)
    for group, cls, scale in (
        (premium_adv, PREMIUM, premium_tightness),
        (regular_adv, REGULAR, tightness),
    ):
        sub = bids[group][:, type_class == cls]
        n_bidders = np.maximum((sub > 0).sum(axis=0), 1)
        share = (sub * expected[type_class == cls] / n_bidders).sum(axis=1)
        budgets[group] = np.maximum(np.round(scale * share), 1).astype(int)

    mean = np.asarray(count_mean, dtype=float)
    std = np.asarray(count_std, dtype=float)
    return Market(
        bids=bids,
        budgets=budgets,
        type_class=type_class,
        type_probs=type_probs,
        count_mean=mean,
        count_std=std,
        beta_params=np.array([[1.0, 1.0], list(premium_beta)]),
        max_count=np.ceil(mean + 6 * std).astype(int),
    )


def sample_episode(market: Market, rng: np.random.Generator) -> Episode:
    """Sample a random episode from the market arrival model.

    Args:
        market: The market.
        rng: Random generator.

    Returns:
        The sampled episode, ordered by arrival time.
    """
    types_list = []
    times_list = []
    for cls in (REGULAR, PREMIUM):
        pmf = market.count_pmf(cls)
        n = int(rng.choice(pmf.size, p=pmf))
        idx = np.flatnonzero(market.type_class == cls)
        probs = market.type_probs[idx] / market.type_probs[idx].sum()
        types_list.append(rng.choice(idx, size=n, p=probs))
        a, b = market.beta_params[cls]
        times_list.append(rng.beta(a, b, size=n))
    types = np.concatenate(types_list).astype(int)
    times = np.concatenate(times_list)
    order = np.argsort(times, kind="stable")
    return Episode(types=types[order], times=times[order])


def episode_seed(split: str, index: int, run_seed: int = 0) -> int:
    """Return the deterministic seed of an episode.

    Args:
        split: One of ``"train"``, ``"valid"`` or ``"test"``.
        index: Index of the episode within the split.
        run_seed: Seed of the training run (only used for ``"train"``).

    Returns:
        A seed that is unique across splits and runs.

    Raises:
        ValueError: If ``split`` is unknown.
    """
    offsets = {"test": 1_000_000, "valid": 2_000_000, "train": 3_000_000}
    if split not in offsets:
        raise ValueError(f"unknown split {split!r}")
    run_offset = 1_000_000 * run_seed if split == "train" else 0
    return offsets[split] + run_offset + index
