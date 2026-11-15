"""Tests of the physician-to-patient example (need Pyomo and Gurobi).

The features and the expert decisions of ``udagger_ppa`` were checked to be
identical to those of ``IO_learning`` in the original scripts (git tag
``ppa``) on the same states and seeds, for the full-information,
deterministic, aggregated and two-stage experts.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("pyomo")
pytest.importorskip("gurobipy")

# Imported after the skips above (these modules need Pyomo and Gurobi).
from udagger import run_episode, stack  # noqa: E402

from udagger_ppa.experts import Expert  # noqa: E402
from udagger_ppa.features import feature_names, features  # noqa: E402
from udagger_ppa.instance import Instance, split_seeds  # noqa: E402
from udagger_ppa.problem import PPAProblem  # noqa: E402
from udagger_ppa.solver import offline_optimum  # noqa: E402


def _greedy_states(seed: int, n: int) -> tuple[PPAProblem, Any, list[dict]]:
    problem = PPAProblem(Instance())
    episode = problem.episode(seed)
    greedy = Expert("greedy")
    rng = np.random.default_rng(0)
    states = []
    while not episode.done and len(states) < n:
        state = episode.observe()
        states.append(state)
        episode.step(greedy(episode, state, rng).action)
    return problem, episode, states


def test_features_and_mask() -> None:
    """Features follow their names; the mask matches the blocked flags."""
    problem, episode, states = _greedy_states(5001, 200)
    n_phys = episode.sim.P
    names = feature_names(n_phys)
    for state in states:
        x = features(state, episode.sim)
        assert x.shape == (len(names),) and x.dtype == np.float32
        blocked = x[-n_phys - 1 : -1]
        np.testing.assert_array_equal(
            episode.mask(state), np.append(blocked == 0, True)
        )
        assert x[names.index("mask_reject")] == 0  # rejection is always feasible


@pytest.mark.parametrize(
    ("kind", "k"), [("full", 1), ("det", 1), ("agg", 3), ("saa", 2)]
)
def test_expert_labels_are_feasible(kind: str, k: int) -> None:
    """Expert labels are distributions over feasible actions."""
    problem, episode, states = _greedy_states(5002, 4)
    expert = Expert(kind, n_scenarios=k, time_limit=30)
    rng = np.random.default_rng(0)
    for state in states:
        label = expert(episode, state, rng)
        mask = episode.mask(state)
        assert mask[label.action]
        np.testing.assert_allclose(label.dist.sum(), 1.0)
        assert np.all(label.dist[~mask] == 0)


def test_rollout_hindsight_and_seeds() -> None:
    """Greedy labels are feasible; full information is optimal online."""
    problem = PPAProblem(Instance())
    res = run_episode(problem, 5001, Expert("greedy"), collect=True)
    assert res.objective == run_episode(problem, 5001, Expert("greedy")).objective
    data = stack([res])
    rows = np.arange(len(data["mask"]))
    assert np.all(data["mask"][rows, data["expert_action"]])
    np.testing.assert_allclose(data["target"].sum(axis=1), 1.0)
    full = Expert("full", mip_gap=0.0, time_limit=60)
    optimum, _ = offline_optimum(Instance(), 5003, time_limit=60)
    assert run_episode(problem, 5003, full).objective == pytest.approx(optimum)
    assert split_seeds("test", 2) == [5001, 5002]
    with pytest.raises(ValueError):
        split_seeds("valid", 1001)


def test_old_episode_cannot_use_the_shared_simulator() -> None:
    """An episode fails loudly once a newer episode reuses the simulator."""
    problem = PPAProblem(Instance())
    old = problem.episode(5001)
    problem.episode(5002)
    with pytest.raises(RuntimeError):
        old.step(0)
