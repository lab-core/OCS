"""Regression test: results identical to the code before the udagger refactor.

The reference values were produced by the AdWords code before the learning
code was moved to the ``udagger`` package, with the options below. The same
commands must give the same per-episode test revenues, for both action rules
of the learned policy, for the hindsight optimum and for an expert used
online. The values depend on PyTorch and Gurobi giving the same
floating-point results; they were recorded on macOS (arm64).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from udagger.cli import main

from udagger_adwords.cli import APP

COMMON = ["--workers", "1", "--n-valid", "10", "--n-test", "10"]
LEARN = ["--episodes", "20", "--per-iteration", "5", "--retrain-every", "2"]
LEARN += ["--epochs", "3"]

REFERENCE = {
    ("det", "dagger"): {
        "sample": [350, 419, 412, 421, 370, 445, 388, 434, 435, 425],
        "mode": [364, 432, 404, 437, 404, 458, 410, 441, 455, 425],
    },
    ("agg --scenarios 3", "oneshot"): {
        "sample": [343, 400, 393, 417, 361, 432, 379, 428, 421, 412],
        "mode": [373, 425, 411, 422, 395, 437, 404, 417, 437, 419],
    },
    ("saa --scenarios 3", "dagger"): {
        "sample": [343, 416, 388, 442, 399, 456, 393, 454, 450, 430],
        "mode": [373, 439, 412, 444, 406, 466, 413, 461, 464, 443],
    },
    ("full", "oneshot"): {
        "sample": [317, 402, 377, 428, 370, 447, 389, 440, 442, 409],
        "mode": [360, 436, 411, 443, 402, 465, 410, 460, 464, 443],
    },
}
HINDSIGHT = [419, 447, 441, 463, 412, 468, 419, 468, 468, 461]
EXPERT_AGG3 = [389, 421, 407, 441, 397, 455, 402, 442, 454, 436]


def _revenues(path: Path) -> list[float]:
    return pd.read_csv(path)["revenue"].tolist()


def test_hindsight_and_expert(tmp_path: Path) -> None:
    """Hindsight optimum and an aggregated expert online are unchanged."""
    main(APP, "", ["--out", str(tmp_path), *COMMON, "hindsight"])
    hindsight = pd.read_csv(tmp_path / "hindsight_test.csv")["hindsight"]
    np.testing.assert_allclose(hindsight, HINDSIGHT)
    expert = ["expert", "--expert", "agg", "--scenarios", "3"]
    main(APP, "", ["--out", str(tmp_path), *COMMON, *expert])
    np.testing.assert_allclose(
        _revenues(tmp_path / "expert_agg3_gap0.02.csv"), EXPERT_AGG3
    )


@pytest.mark.parametrize(("expert", "mode"), list(REFERENCE))
def test_learning_runs(tmp_path: Path, expert: str, mode: str) -> None:
    """Learned policies give the same test revenues as before the refactor."""
    args = ["--out", str(tmp_path), *COMMON, "learn", "--expert", *expert.split()]
    main(APP, "", [*args, "--mode", mode, *LEARN])
    (run,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    for rule, expected in REFERENCE[(expert, mode)].items():
        np.testing.assert_allclose(_revenues(run / f"test_{rule}.csv"), expected)
