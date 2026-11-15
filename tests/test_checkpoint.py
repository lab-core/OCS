"""Interrupted and resumed runs give the same results as uninterrupted ones."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from test_udagger import OracleExpert, ToyProblem

from udagger import LearnConfig, LearnResult, Runner, learn
from udagger import checkpoint as ckpt


class Crash(RuntimeError):
    """Simulated end of a job."""


class CrashingRunner(Runner):
    """Runner that stops after a number of collecting calls."""

    def __init__(self, calls: int) -> None:
        """Run in-process and crash on collecting call number ``calls``."""
        super().__init__(ToyProblem())
        self.calls = calls

    def run(self, seeds: list[int], **kwargs: object) -> list:
        """Count the collecting calls (validation calls do not collect)."""
        if kwargs.get("collect"):
            self.calls -= 1
            if self.calls < 0:
                raise Crash
        return super().run(seeds, **kwargs)


CONFIGS = {
    "dagger": LearnConfig(mode="dagger", per_iteration=5, retrain_every=2, epochs=3),
    "oneshot": LearnConfig(mode="oneshot", per_iteration=5, sizes=[10], epochs=3),
}


def _learn(runner: Runner, cfg: LearnConfig, folder: Path | None) -> LearnResult:
    return learn(
        runner,
        OracleExpert(),
        cfg,
        train_seeds=list(range(100, 140)),
        valid_seeds=list(range(5)),
        log=lambda line: None,
        checkpoint_dir=folder,
        checkpoint_every=2,
    )


def _same(a: LearnResult, b: LearnResult) -> None:
    keys = ("iteration", "episodes", "samples", "val_objective", "val_loss")
    assert [{k: r[k] for k in keys} for r in a.curve] == [
        {k: r[k] for k in keys} for r in b.curve
    ]
    assert a.best["iteration"] == b.best["iteration"]
    assert a.label_seconds == pytest.approx(b.label_seconds)
    for (wa, ba), (wb, bb) in zip(a.policy.weights, b.policy.weights, strict=True):
        np.testing.assert_array_equal(wa, wb)
        np.testing.assert_array_equal(ba, bb)
    assert a.data.keys() == b.data.keys()
    for k in a.data:
        np.testing.assert_array_equal(a.data[k], b.data[k])


# (mode, labelling calls before the crash, whether a checkpoint exists then):
# DAgger saves every 2 iterations (8 in total), one-shot after each of its 4
# labelling blocks of 10 episodes.
CASES = [
    ("dagger", 1, False),
    ("dagger", 3, True),
    ("dagger", 5, True),
    ("oneshot", 1, True),
    ("oneshot", 3, True),
]


@pytest.mark.parametrize(("mode", "crash_after", "saved"), CASES)
def test_resume_gives_same_result(
    tmp_path: Path, mode: str, crash_after: int, saved: bool
) -> None:
    """Crash after some labelling calls, resume, compare with no crash."""
    cfg = CONFIGS[mode]
    reference = _learn(Runner(ToyProblem()), cfg, None)
    folder = tmp_path / "ckpt"
    with pytest.raises(Crash):
        _learn(CrashingRunner(crash_after), cfg, folder)
    assert (folder / "state.json").exists() == saved
    resumed = _learn(Runner(ToyProblem()), cfg, folder)
    _same(reference, resumed)


def test_checkpoint_of_another_run_is_refused(tmp_path: Path) -> None:
    """A checkpoint is only resumed with the same settings."""
    folder = tmp_path / "ckpt"
    with pytest.raises(Crash):
        _learn(CrashingRunner(3), CONFIGS["dagger"], folder)
    other = LearnConfig(mode="dagger", per_iteration=5, retrain_every=2, epochs=4)
    with pytest.raises(ValueError, match="another run"):
        _learn(Runner(ToyProblem()), other, folder)


def test_interrupted_save_keeps_previous_checkpoint(tmp_path: Path) -> None:
    """Files of two different saves are not mixed."""
    folder = tmp_path / "ckpt"
    fingerprint = {"run": 1}
    ckpt.save(folder, fingerprint, {"next": 2}, {"x": np.zeros((2, 3))})
    assert ckpt.load(folder, fingerprint)["state"]["next"] == 2
    meta = json.loads((folder / "state.json").read_text())
    meta["token"] = "another-save"
    (folder / "state.json").write_text(json.dumps(meta))
    assert ckpt.load(folder, fingerprint) is None
    ckpt.remove(folder)
    assert not folder.exists()
