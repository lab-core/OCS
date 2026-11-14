"""Tests of the problem-independent package on a toy problem."""

from __future__ import annotations

import numpy as np
import pytest

from udagger import (
    Label,
    LearnConfig,
    Runner,
    aggregate_votes,
    learn,
    merge,
    one_hot,
    run_episode,
    stack,
)


class ToyEpisode:
    """Ten epochs; the best action is the index of the largest feature.

    Action 0 is infeasible at odd epochs. The objective is the number of
    epochs where the best action was taken.
    """

    n_actions = 4

    def __init__(self, seed: int) -> None:
        """Draw the features of every epoch."""
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.x = rng.random((10, self.n_actions)).astype(np.float32)
        self.t = 0
        self.score = 0

    @property
    def done(self) -> bool:
        """Whether all epochs were played."""
        return self.t >= len(self.x)

    def observe(self) -> int:
        """The state is the epoch index."""
        return self.t

    def features(self, state: int) -> np.ndarray:
        """Features of the epoch."""
        return self.x[state]

    def mask(self, state: int) -> np.ndarray:
        """Action 0 is infeasible at odd epochs."""
        m = np.ones(self.n_actions, dtype=bool)
        m[0] = state % 2 == 0
        return m

    def best(self, state: int) -> int:
        """Best feasible action."""
        return int(np.argmax(np.where(self.mask(state), self.x[state], -1.0)))

    def step(self, action: int) -> None:
        """Score the action and move on."""
        assert self.mask(self.t)[action]
        self.score += int(action == self.best(self.t))
        self.t += 1

    def info(self, state: int) -> dict[str, float]:
        """Store the epoch with the labels."""
        return {"epoch": state}

    def stats(self) -> dict[str, float]:
        """Objective: number of best actions."""
        return {"objective": float(self.score)}


class ToyProblem:
    """Factory of toy episodes."""

    n_actions = ToyEpisode.n_actions
    maximize = True

    def episode(self, seed: int) -> ToyEpisode:
        """Create an episode."""
        return ToyEpisode(seed)


class OracleExpert:
    """Expert taking the best feasible action."""

    name = "oracle"

    def __call__(
        self, episode: ToyEpisode, state: int, rng: np.random.Generator
    ) -> Label:
        """Label a state with the best action."""
        a = episode.best(state)
        return Label(a, one_hot(a, episode.n_actions), solve_time=0.001)


def test_aggregate_votes() -> None:
    """Votes become a distribution; majority or sampled action."""
    action, dist = aggregate_votes([1, 1, 3], 4)
    assert action == 1
    np.testing.assert_allclose(dist, [0, 2 / 3, 0, 1 / 3])
    sampled, _ = aggregate_votes([2, 2], 4, np.random.default_rng(0), sample=True)
    assert sampled == 2
    with pytest.raises(ValueError):
        aggregate_votes([], 4)
    with pytest.raises(ValueError):
        aggregate_votes([1], 4, sample=True)


def test_rollout_collects_labels() -> None:
    """The expert alone is optimal; labels and info are stacked by episode."""
    problem = ToyProblem()
    results = [run_episode(problem, s, OracleExpert(), collect=True) for s in (1, 2)]
    assert all(r.objective == 10 for r in results)
    assert results[0].solve_time == pytest.approx(0.01)
    data = stack(results)
    assert data["x"].shape == (20, 4) and data["target"].shape == (20, 4)
    assert list(np.unique(data["episode"])) == [1, 2]
    np.testing.assert_array_equal(data["epoch"][:10], np.arange(10))
    assert merge({}, data) is data
    assert merge(data, data)["x"].shape[0] == 40
    with pytest.raises(ValueError):
        run_episode(problem, 1)


@pytest.mark.parametrize("mode", ["oneshot", "dagger"])
def test_learn_imitates_expert(mode: str) -> None:
    """Both modes learn a near-optimal policy, chosen on validation."""
    runner = Runner(ToyProblem())
    cfg = LearnConfig(mode=mode, epochs=40, per_iteration=10, retrain_every=2)
    res = learn(
        runner,
        OracleExpert(),
        cfg,
        train_seeds=list(range(100, 140)),
        valid_seeds=list(range(10)),
        log=lambda line: None,
    )
    best = max(r["val_objective"] for r in res.curve)
    assert res.best["val_objective"] == best
    assert best >= 7.0
    assert res.data["x"].shape[0] == 400
    with pytest.raises(ValueError):
        learn(
            runner,
            OracleExpert(),
            LearnConfig(mode="dagger", per_iteration=3),
            train_seeds=list(range(10)),
            valid_seeds=[0],
        )


def test_minimisation_selects_lowest() -> None:
    """With ``maximize = False`` the lowest validation objective wins."""
    problem = ToyProblem()
    problem.maximize = False
    res = learn(
        Runner(problem),
        OracleExpert(),
        LearnConfig(mode="oneshot", sizes=[2], epochs=30),
        train_seeds=list(range(100, 130)),
        valid_seeds=list(range(5)),
        log=lambda line: None,
    )
    assert res.best["val_objective"] == min(r["val_objective"] for r in res.curve)
