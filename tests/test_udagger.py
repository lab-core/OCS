"""Tests of the problem-independent package on a toy problem."""

from __future__ import annotations

from pathlib import Path

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


def test_learn_rejects_empty_inputs() -> None:
    """Missing episodes or labels raise a clear error."""
    runner = Runner(ToyProblem())
    with pytest.raises(ValueError, match="training and validation"):
        learn(runner, OracleExpert(), LearnConfig(), [], [0])

    class Silent(ToyProblem):
        def episode(self, seed: int) -> ToyEpisode:
            ep = ToyEpisode(seed)
            ep.t = len(ep.x)  # already over: nothing to label
            return ep

    with pytest.raises(ValueError, match="no state was labelled"):
        learn(Runner(Silent()), OracleExpert(), LearnConfig(mode="oneshot"), [1], [2])


def _toy_problem(args: object) -> ToyProblem:
    return ToyProblem()


def _toy_expert(args: object) -> OracleExpert:
    return OracleExpert()


def _toy_seeds(split: str, n: int, run_seed: int) -> list[int]:
    base = {"train": 1000 * (run_seed + 1), "valid": 100, "test": 0}[split]
    return list(range(base, base + n))


def _toy_hindsight(args: object, seed: int) -> float:
    return 10.0  # the oracle takes the best action at every epoch


def test_cli_and_analysis(tmp_path: Path) -> None:
    """Generic commands write the shared files, which the analysis reads."""
    from udagger.analysis import Study, standard_outputs
    from udagger.cli import Application, main

    app = Application(
        name="toy",
        objective="score",
        expert_kinds=("oracle",),
        add_arguments=lambda parser: None,
        make_problem=_toy_problem,
        make_expert=_toy_expert,
        seeds=_toy_seeds,
        hindsight=_toy_hindsight,
    )
    out = ["--out", str(tmp_path), "--n-valid", "5", "--n-test", "5"]
    main(app, "", [*out, "hindsight"])
    main(app, "", [*out, "expert", "--expert", "oracle"])
    for mode in ("oneshot", "dagger"):
        learn = ["learn", "--expert", "oracle", "--mode", mode, "--episodes", "20"]
        main(app, "", [*out, *learn, "--epochs", "20", "--save-data"])
    run = tmp_path / "oracle_gap0.02_dagger_s0"
    assert not (run / "checkpoint").exists()  # removed once the run is complete
    assert {p.name for p in run.iterdir()} >= {
        "test.csv",
        "test_mode.csv",
        "test_sample.csv",
        "test.json",
        "policy.npz",
        "labels.npz",
    }
    study = Study(
        prefix="toy",
        objective="score",
        maximize=True,
        order=["oracle"],
        labels={"oracle": "Oracle"},
        short={"oracle": "Oracle"},
        colors={"oracle": "k"},
        comparisons=((("oracle", "oneshot"), ("oracle", "dagger")),),
        curve_experts=("oracle",),
    )
    # A variant under --tag must not be averaged with the main runs.
    variant = ["learn", "--expert", "oracle", "--mode", "dagger", "--episodes", "20"]
    main(app, "", [*out, *variant, "--epochs", "1", "--tag", "_v"])
    from udagger.analysis import episode_regret, load

    _, runs, _ = load(tmp_path, study)
    assert sorted(runs["tag"]) == ["", "", "_v"]
    assert len(episode_regret(study, tmp_path, "oracle", "dagger", 0.02)) == 5
    table = standard_outputs(study, tmp_path, tmp_path / "paper")
    assert table.loc[0, "online"] == pytest.approx(0.0)  # the expert is optimal
    assert 0.0 <= table.loc[0, "oneshot"] <= 100.0
    assert (tmp_path / "paper" / "tables" / "toy_tests.tex").exists()
    assert (tmp_path / "paper" / "figs" / "toy_curves.pdf").exists()


def test_grid_lines() -> None:
    """Grid lines follow the command-line syntax."""
    from udagger.grid import expert_lines, gap_lines, learn_lines

    lines = learn_lines([("saa", 5)], range(2), 40, save_seed=0)
    assert lines[0] == (
        "learn --expert saa --scenarios 5 --seed 0 --mode oneshot --episodes 40"
        " --save-data"
    )
    assert len(lines) == 4
    assert gap_lines(("saa", 30), (0.1,), [3], 40) == [
        "learn --expert saa --scenarios 30 --seed 3 --gap 0.1 --mode dagger"
        " --episodes 40"
    ]
    assert expert_lines([("det", 1)], ("saa", 30), (0.002,), prefix="--x 1 ") == [
        "--x 1 expert --expert det --scenarios 1",
        "--x 1 expert --expert saa --scenarios 30 --gap 0.002",
    ]


def test_in_process_runners_are_independent() -> None:
    """Two runners with workers=1 each run their own problem."""

    class Constant(ToyProblem):
        def __init__(self, value: float) -> None:
            self.value = value

        def episode(self, seed: int) -> ToyEpisode:
            ep = ToyEpisode(seed)
            ep.stats = lambda: {"objective": self.value}
            return ep

    first, second = Runner(Constant(1.0)), Runner(Constant(2.0))
    assert first.run([0], expert=OracleExpert())[0].objective == 1.0
    assert second.run([0], expert=OracleExpert())[0].objective == 2.0


def test_learn_refuses_to_overwrite_other_options(tmp_path: Path) -> None:
    """A run folder is not overwritten by a run with other options."""
    from udagger.cli import Application, main

    app = Application(
        name="toy",
        objective="score",
        expert_kinds=("oracle",),
        add_arguments=lambda parser: None,
        make_problem=_toy_problem,
        make_expert=_toy_expert,
        seeds=_toy_seeds,
        hindsight=_toy_hindsight,
    )
    base = ["--out", str(tmp_path), "--n-valid", "2", "--n-test", "2", "learn"]
    base += ["--expert", "oracle", "--episodes", "5"]
    main(app, "", [*base, "--epochs", "2"])
    main(app, "", [*base, "--epochs", "2"])  # same options: allowed
    with pytest.raises(SystemExit):
        main(app, "", [*base, "--epochs", "3"])
    main(app, "", [*base, "--epochs", "3", "--tag", "_e3"])
    assert (tmp_path / "oracle_gap0.02_oneshot_s0_e3" / "test.json").exists()


def test_invalid_sizes_are_ignored() -> None:
    """Non-positive one-shot sizes are ignored, not used as slices."""
    res = learn(
        Runner(ToyProblem()),
        OracleExpert(),
        LearnConfig(mode="oneshot", sizes=[0, -3, 5], epochs=2),
        train_seeds=list(range(100, 110)),
        valid_seeds=[0],
        log=lambda line: None,
    )
    assert [r["episodes"] for r in res.curve] == [5, 10]


def _toy_app(hindsight: object = _toy_hindsight) -> object:
    """Application of the toy problem for the generic command line."""
    from udagger.cli import Application

    return Application(
        name="toy",
        objective="score",
        expert_kinds=("oracle",),
        add_arguments=lambda parser: None,
        make_problem=_toy_problem,
        make_expert=_toy_expert,
        seeds=_toy_seeds,
        hindsight=hindsight,
    )


def _failing_hindsight(args: object, seed: int) -> float:
    if seed == 1:
        raise RuntimeError("no incumbent")
    return 10.0


def test_reruns_variants_and_failures(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """Completed runs are skipped, variants need a tag, failures are recorded."""
    from udagger.analysis import Study, standard_outputs
    from udagger.cli import main

    app = _toy_app()
    out = ["--out", str(tmp_path), "--n-valid", "2", "--n-test", "3"]
    expert = ["expert", "--expert", "oracle"]
    main(app, "", [*out, *expert])
    main(app, "", [*out, *expert])
    assert "skipped" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(app, "", [*out, *expert, "--time-limit", "5"])
    main(app, "", [*out, *expert, "--time-limit", "5", "--tag", "_t5"])
    assert (tmp_path / "expert_oracle_gap0.02_t5.csv").exists()

    main(_toy_app(_failing_hindsight), "", [*out, "hindsight"])
    import pandas as pd

    test = pd.read_csv(tmp_path / "hindsight_test.csv")
    assert test["hindsight"].isna().sum() == 1  # seed 1 failed, others kept

    # The analysis runs before any learning run exists.
    study = Study(
        prefix="toy",
        objective="score",
        maximize=True,
        order=["oracle"],
        labels={"oracle": "Oracle"},
        short={"oracle": "Oracle"},
        colors={"oracle": "k"},
    )
    table = standard_outputs(study, tmp_path, tmp_path / "paper")
    assert table.empty


class JobEnded(Exception):
    """Simulated end of a cluster job (time limit)."""


def test_resume_ignores_options_used_after_learning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """A run interrupted and resubmitted with another --n-test resumes."""
    from udagger.cli import main
    from udagger.rollout import Runner as RunnerClass

    app = _toy_app()
    base = ["--out", str(tmp_path), "--n-valid", "2"]
    learn_args = ["learn", "--expert", "oracle", "--mode", "dagger"]
    learn_args += ["--episodes", "20", "--epochs", "2", "--checkpoint-every", "1"]
    original = RunnerClass.run
    calls = {"collect": 0}

    def crash_after_one(self: RunnerClass, seeds: list[int], **kw: object) -> list:
        if kw.get("collect"):
            calls["collect"] += 1
            if calls["collect"] > 1:
                raise JobEnded
        return original(self, seeds, **kw)

    monkeypatch.setattr(RunnerClass, "run", crash_after_one)
    with pytest.raises(JobEnded):
        main(app, "", [*base, "--n-test", "2", *learn_args])
    run = tmp_path / "oracle_gap0.02_dagger_s0"
    assert (run / "checkpoint" / "state.json").exists()
    monkeypatch.setattr(RunnerClass, "run", original)
    main(app, "", [*base, "--n-test", "4", *learn_args])
    assert '"resumed"' in capsys.readouterr().out
    import pandas as pd

    assert len(pd.read_csv(run / "test.csv")) == 4
    assert not (run / "checkpoint").exists()


class BadExpert(OracleExpert):
    """Expert that labels an infeasible action at odd epochs."""

    name = "bad"

    def __call__(
        self, episode: ToyEpisode, state: int, rng: np.random.Generator
    ) -> Label:
        """Put half of the target on action 0, which is infeasible at odd epochs."""
        label = super().__call__(episode, state, rng)
        dist = 0.5 * label.dist
        dist[0] += 0.5
        return Label(label.action, dist)


def test_infeasible_expert_targets_and_actions() -> None:
    """Mass on infeasible actions is dropped; infeasible actions raise."""
    res = run_episode(ToyProblem(), 3, BadExpert(), collect=True)
    data = stack([res])
    odd = data["epoch"] % 2 == 1
    assert np.all(data["target"][odd, 0] == 0)
    np.testing.assert_allclose(data["target"].sum(axis=1), 1.0)

    class Infeasible(OracleExpert):
        def __call__(
            self, episode: ToyEpisode, state: int, rng: np.random.Generator
        ) -> Label:
            return Label(0, one_hot(0, episode.n_actions))

    with pytest.raises(ValueError, match="infeasible"):
        run_episode(ToyProblem(), 3, Infeasible())
