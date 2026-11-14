"""Calibrate market parameters by comparing simple experts online."""

from __future__ import annotations

import itertools
import multiprocessing as mp
import sys

import numpy as np
from udagger import run_episode

from udagger_adwords.experts import Expert
from udagger_adwords.market import make_market, sample_episode
from udagger_adwords.problem import AdwordsProblem
from udagger_adwords.solver import offline_optimum

# Episodes used only for calibration (test, valid and train start at 1e6, 2e6
# and 3e6 + 1e6 * run, see market.episode_seed).
CALIBRATION_OFFSET = 900_000


def task(args: tuple) -> tuple:
    """Evaluate all experts of a configuration on one episode."""
    cfg, seed = args
    m = make_market(**cfg)
    ep = sample_episode(m, np.random.default_rng(seed))
    opt = offline_optimum(m, np.bincount(ep.types, minlength=m.n_types))
    out = [opt]
    for kind, k in (("greedy", 1), ("msvv", 1), ("det", 1), ("agg", 10), ("saa", 10)):
        out.append(run_episode(AdwordsProblem(m), seed, Expert(kind, k)).objective)
    return out


def main() -> None:
    """Run the calibration grid and select the market parameters.

    Each configuration is scored on its own calibration episodes by the
    margin of the two-stage expert over the best myopic rule (greedy or
    MSVV), in ratio to the hindsight optimum, so that anticipation matters;
    ties are broken by its margin over the deterministic expert.

    Usage: ``python scripts/calibrate.py [N_EPISODES] [WORKERS]``.
    """
    n_ep = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    grid = [
        {"tightness": t, "premium_tightness": p, "seed": 0}
        for t, p in itertools.product((0.8, 1.0, 1.2), (0.7, 1.0))
    ]
    names = ("gr", "msvv", "det", "agg10", "saa10")
    seeds = [CALIBRATION_OFFSET + i for i in range(n_ep)]
    best = None
    with mp.get_context("spawn").Pool(workers) as pool:
        for cfg in grid:
            res = np.array(pool.map(task, [(cfg, s) for s in seeds]))
            ratio = dict(zip(names, res[:, 1:].sum(0) / res[:, 0].sum()))
            score = (
                ratio["saa10"] - max(ratio["gr"], ratio["msvv"]),
                ratio["saa10"] - ratio["det"],
            )
            print(
                cfg,
                "opt %.1f" % res[:, 0].mean(),
                " ".join(f"{n}={r:.3f}" for n, r in ratio.items()),
                "score=(%.3f, %.3f)" % score,
                flush=True,
            )
            if best is None or score > best[0]:
                best = (score, cfg)
    print("selected:", best[1], "score=(%.3f, %.3f)" % best[0])


if __name__ == "__main__":
    main()
