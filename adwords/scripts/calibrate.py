"""Calibrate market parameters by comparing simple experts online."""

from __future__ import annotations

import itertools
import multiprocessing as mp
import sys

import numpy as np

from ilco_adwords.experts import Expert
from ilco_adwords.market import make_market, sample_episode
from ilco_adwords.rollout import run_episode
from ilco_adwords.solver import offline_optimum


def task(args: tuple) -> tuple:
    """Evaluate all experts of a configuration on one episode."""
    cfg, seed = args
    m = make_market(**cfg)
    ep = sample_episode(m, np.random.default_rng(seed))
    opt = offline_optimum(m, np.bincount(ep.types, minlength=m.n_types))
    out = [opt]
    for kind, k in (("greedy", 1), ("msvv", 1), ("det", 1), ("agg", 10), ("saa", 10)):
        out.append(run_episode(m, seed, Expert(kind, k)).revenue)
    return out


def main() -> None:
    """Run the calibration grid and print ratios to the hindsight optimum."""
    n_ep = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    grid = [
        {"tightness": t, "premium_tightness": p, "seed": s}
        for t, p, s in itertools.product((0.8, 1.0, 1.2), (0.7, 1.0), (0,))
    ]
    with mp.get_context("spawn").Pool(
        int(sys.argv[2]) if len(sys.argv) > 2 else 8
    ) as pool:
        for cfg in grid:
            res = np.array(pool.map(task, [(cfg, 1_000_000 + i) for i in range(n_ep)]))
            ratio = res[:, 1:].sum(0) / res[:, 0].sum()
            print(
                cfg,
                "opt %.1f" % res[:, 0].mean(),
                " ".join(
                    f"{n}={r:.3f}"
                    for n, r in zip(("gr", "msvv", "det", "agg10", "saa10"), ratio)
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
