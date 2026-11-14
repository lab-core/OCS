"""Tests of the PPA example, including equivalence with the original code.

The original scripts (``legacy/``) build features and expert decisions with
``IO_learning``; the package must give the same values on the same states
and seeds. These tests need Pyomo and Gurobi.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

pytest.importorskip("pyomo")
pytest.importorskip("gurobipy")

# Imported after the skips above (these modules need Pyomo and Gurobi).
from udagger import run_episode, stack  # noqa: E402

from udagger_ppa.experts import EXPERT_SEED_OFFSET, Expert  # noqa: E402
from udagger_ppa.features import features  # noqa: E402
from udagger_ppa.instance import Instance, split_seeds  # noqa: E402
from udagger_ppa.problem import PPAProblem  # noqa: E402
from udagger_ppa.solver import offline_optimum  # noqa: E402

LEGACY = str(Path(__file__).resolve().parents[1] / "legacy")


@pytest.fixture(scope="module")
def legacy() -> Iterator[tuple[ModuleType, ModuleType]]:
    """Modules of the original scripts."""
    sys.path.insert(0, LEGACY)
    import Learn_Module
    import utility

    yield Learn_Module, utility
    sys.path.remove(LEGACY)


def _legacy_config(utility: ModuleType, options: list[str]) -> Any:
    saved = sys.argv
    try:
        sys.argv = [saved[0], *options]
        return utility.build_config(utility.parse_args())
    finally:
        sys.argv = saved


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


def test_features_match_original(
    legacy: tuple[ModuleType, ModuleType], tmp_path: Path
) -> None:
    """Features and masks equal IO_learning.get_feature_vector."""
    learn_module, utility = legacy
    problem, episode, states = _greedy_states(5001, 200)
    config = _legacy_config(utility, [])
    learner = learn_module.IO_learning(
        episode.sim, asdict(config), folder=str(tmp_path)
    )
    assert len(learner.feature_names) == len(features(states[0], episode.sim))
    for state in states:
        old = np.asarray(learner.get_feature_vector(state), np.float32)
        np.testing.assert_array_equal(features(state, episode.sim), old)
        n_phys = episode.sim.P
        np.testing.assert_array_equal(
            episode.mask(state), np.append(old[-n_phys - 1 : -1] == 0, True)
        )


@pytest.mark.parametrize(
    ("kind", "k", "options"),
    [
        ("full", 1, ["--stoch", "0", "--new", "0"]),
        ("det", 1, ["--stoch", "0", "--new", "1", "--scenarios", "1"]),
        ("agg", 3, ["--stoch", "0", "--new", "1", "--scenarios", "3"]),
        ("saa", 2, ["--stoch", "1", "--new", "1", "--scenarios", "2"]),
    ],
)
def test_experts_match_original(
    legacy: tuple[ModuleType, ModuleType],
    kind: str,
    k: int,
    options: list[str],
    tmp_path: Path,
) -> None:
    """Expert votes equal IO_learning.get_expert_action with the same seeds."""
    learn_module, utility = legacy
    problem, episode, states = _greedy_states(5002, 6)
    config = _legacy_config(utility, [*options, "--mipgap", "0.02"])
    learner = learn_module.IO_learning(
        episode.sim, asdict(config), folder=str(tmp_path)
    )
    learner.set_scenarios(EXPERT_SEED_OFFSET + 1000 * episode.seed, asdict(config))
    expert = Expert(kind, n_scenarios=k, mip_gap=0.02, time_limit=30)
    rng = np.random.default_rng(0)
    for state in states:
        votes = learner.get_expert_action(state, "test")[0]
        label = expert(episode, state, rng)
        old = np.bincount(np.asarray(votes, int), minlength=len(label.dist))
        np.testing.assert_allclose(label.dist, old / old.sum())


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
