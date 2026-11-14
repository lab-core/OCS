"""Unit tests of the AdWords simulator, experts and solver."""

from __future__ import annotations

import numpy as np
import pytest

from ilco_adwords.env import (
    Simulator,
    feasible_mask,
    remaining_count_posterior,
    sample_future_counts,
)
from ilco_adwords.experts import Expert
from ilco_adwords.market import (
    PREMIUM,
    REGULAR,
    Episode,
    Market,
    make_market,
    sample_episode,
)
from ilco_adwords.policy import NumpyPolicy, features, train_policy
from ilco_adwords.rollout import run_episode
from ilco_adwords.solver import offline_optimum, solve_count_mip


@pytest.fixture(scope="module")
def market() -> Market:
    """Default market."""
    return make_market()


def test_episode_reproducible(market: Market) -> None:
    """The same seed yields the same episode, ordered in time."""
    a = sample_episode(market, np.random.default_rng(3))
    b = sample_episode(market, np.random.default_rng(3))
    assert np.array_equal(a.types, b.types)
    assert np.all(np.diff(a.times) >= 0)


def test_simulator_respects_budgets(market: Market) -> None:
    """Infeasible actions are rejected and budgets never become negative."""
    ep = sample_episode(market, np.random.default_rng(0))
    first = Simulator(market, ep)
    infeasible = np.flatnonzero(~feasible_mask(market, first.state()))
    if infeasible.size:
        with pytest.raises(ValueError):
            first.step(int(infeasible[0]))
    sim = Simulator(market, ep)
    while not sim.done:
        mask = feasible_mask(market, sim.state())
        sim.step(int(np.flatnonzero(mask)[-1]))
        assert np.all(sim.budgets >= 0)


def test_hand_made_instance() -> None:
    """Greedy is myopic, hindsight reserves budget for the premium query."""
    market = Market(
        bids=np.array([[5, 10], [2, 0]]),
        budgets=np.array([10, 10]),
        type_class=np.array([REGULAR, PREMIUM]),
        type_probs=np.array([1.0, 1.0]),
        count_mean=np.array([1.0, 1.0]),
        count_std=np.array([0.5, 0.5]),
        beta_params=np.array([[1.0, 1.0], [4.0, 1.0]]),
        max_count=np.array([4, 4]),
    )
    ep = Episode(types=np.array([0, 1]), times=np.array([0.1, 0.9]))
    assert offline_optimum(market, np.array([1, 1])) == pytest.approx(12.0)
    sim = Simulator(market, ep)
    state = sim.state()
    greedy = Expert("greedy")(market, state, np.random.default_rng(0))
    assert greedy.action == 1
    full = Expert("full")(
        market, state, np.random.default_rng(0), sim.true_future_counts()
    )
    assert full.action == 2


def test_hindsight_dominates_online(market: Market) -> None:
    """No online expert beats the offline optimum."""
    for seed in range(3):
        ep = sample_episode(market, np.random.default_rng(seed))
        opt = offline_optimum(market, np.bincount(ep.types, minlength=market.n_types))
        for kind, k in (("greedy", 1), ("msvv", 1), ("full", 1), ("saa", 3)):
            rev = run_episode(market, seed, Expert(kind, k)).revenue
            assert rev <= opt + 1e-6
            if kind == "full":
                assert rev == pytest.approx(opt)


def test_posterior_and_scenarios(market: Market) -> None:
    """Posterior is a distribution; late in the horizon few queries remain."""
    for cls in (REGULAR, PREMIUM):
        post = remaining_count_posterior(market, cls, 0.5, 10)
        assert post.sum() == pytest.approx(1.0)
    early = remaining_count_posterior(market, PREMIUM, 0.0, 0)
    late = remaining_count_posterior(market, PREMIUM, 0.999, 30)
    support_late = np.arange(late.size)
    support_early = np.arange(early.size)
    assert support_late @ late < support_early @ early
    sim = Simulator(market, sample_episode(market, np.random.default_rng(1)))
    counts = sample_future_counts(market, sim.state(), 4, np.random.default_rng(0))
    assert counts.shape == (4, market.n_types) and counts.min() >= 0


def test_saa_action_feasible(market: Market) -> None:
    """The first-stage decision of the SAA model is feasible."""
    sim = Simulator(market, sample_episode(market, np.random.default_rng(2)))
    state = sim.state()
    scen = sample_future_counts(market, state, 5, np.random.default_rng(0))
    res = solve_count_mip(market, state.budgets, state.query, scen)
    assert feasible_mask(market, state)[res.action]


def test_policy_learns_greedy(market: Market) -> None:
    """A policy cloned from greedy reproduces most greedy actions."""
    data = [run_episode(market, s, Expert("greedy"), collect=True) for s in range(30)]
    x = np.concatenate([d.x for d in data])
    y = np.concatenate([d.target for d in data])
    m = np.concatenate([d.mask for d in data])
    policy, stats = train_policy(x, y, m, max_epochs=40)
    assert isinstance(policy, NumpyPolicy)
    assert stats.val_acc > 0.9
    sim = Simulator(market, sample_episode(market, np.random.default_rng(99)))
    assert features(market, sim.state()).shape[0] == x.shape[1]
