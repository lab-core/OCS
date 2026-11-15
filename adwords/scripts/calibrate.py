"""Calibrate market parameters by comparing simple experts online."""

from __future__ import annotations

import itertools
import multiprocessing as mp
import sys

import numpy as np
from udagger import run_episode

from udagger_adwords.experts import Expert
from udagger_adwords.instance import make_market, sample_episode
from udagger_adwords.problem import AdwordsProblem
from udagger_adwords.solver import offline_optimum

# Episodes used only for calibration (test, valid and train start at 1e6, 2e6
# and 3e6 + 1e6 * run, see market.episode_seed).
CALIBRATION_OFFSET = 900_000
# Expert settings: "exact" (optimal MIPs, 10 s, majority vote) was used to
# choose the budgets; "protocol" is the setting of the study (2% gap, 30 s,
# sampled votes), to check that the choice does not depend on it.
SETTINGS = {
    "exact": {"mip_gap": 0.0, "time_limit": 10.0, "sample_vote": False},
    "protocol": {"mip_gap": 0.02, "time_limit": 30.0, "sample_vote": True},
}


def task(args: tuple) -> tuple:
    """Evaluate all experts of a configuration on one episode."""
    cfg, seed, settings = args
    m = make_market(**cfg)
    ep = sample_episode(m, np.random.default_rng(seed))
    opt = offline_optimum(m, np.bincount(ep.types, minlength=m.n_types))
    out = [opt]
    for kind, k in (("greedy", 1), ("msvv", 1), ("det", 1), ("agg", 10), ("saa", 10)):
        expert = Expert(kind, k, **SETTINGS[settings])
        out.append(run_episode(AdwordsProblem(m), seed, expert).objective)
    return out


def main() -> None:
    """Run the calibration grid and select the market parameters.

    Each configuration is scored on its own calibration episodes by the
    margin of the two-stage expert over the best myopic rule (greedy or
    MSVV), in ratio to the hindsight optimum, so that anticipation matters;
    ties are broken by its margin over the deterministic expert.

    Usage: ``python scripts/calibrate.py [N_EPISODES] [WORKERS] [SETTINGS]``
    with SETTINGS ``exact`` (default) or ``protocol``.
    """
    n_ep = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    settings = sys.argv[3] if len(sys.argv) > 3 else "exact"
    grid = [
        {"tightness": t, "premium_tightness": p, "seed": 0}
        for t, p in itertools.product((0.8, 1.0, 1.2), (0.7, 1.0))
    ]
    names = ("gr", "msvv", "det", "agg10", "saa10")
    seeds = [CALIBRATION_OFFSET + i for i in range(n_ep)]
    best = None
    with mp.get_context("spawn").Pool(workers) as pool:
        for cfg in grid:
            res = np.array(pool.map(task, [(cfg, s, settings) for s in seeds]))
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
