"""Tests of the PPA adapter to ``udagger`` (need Pyomo and Gurobi)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyomo")
pytest.importorskip("gurobipy")

from udagger import run_episode, stack  # noqa: E402  (after the skips)

from udagger_ppa.problem import (  # noqa: E402  (after the skips)
    PPAExpert,
    PPAGreedy,
    PPAProblem,
    config_from_args,
)


@pytest.fixture(scope="module")
def problem() -> PPAProblem:
    """Default instance of the original study."""
    return PPAProblem(config_from_args([]))


def test_greedy_rollout_and_labels(problem: PPAProblem) -> None:
    """Greedy labels are feasible one-hot targets; episodes are reproducible."""
    res = run_episode(problem, 5001, PPAGreedy(), collect=True)
    again = run_episode(problem, 5001, PPAGreedy())
    assert res.objective == again.objective > 0
    data = stack([res])
    assert np.all(data["mask"][np.arange(len(data["mask"])), data["expert_action"]])
    np.testing.assert_allclose(data["target"].sum(axis=1), 1.0)
    assert data["x"].shape[1] == len(problem.ctx.learner.feature_names)


def test_full_information_beats_greedy(problem: PPAProblem) -> None:
    """The full-information expert is at least as good as greedy online."""
    full = PPAExpert(config_from_args(["--stoch", "0", "--new", "0", "--mipgap", "0"]))
    assert full.name == "full"
    cost_full = run_episode(problem, 5002, full).objective
    cost_greedy = run_episode(problem, 5002, PPAGreedy()).objective
    assert cost_full <= cost_greedy
