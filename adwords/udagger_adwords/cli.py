"""Command line of the AdWords experiments (generic ``udagger`` commands).

Usage: ``python -m udagger_adwords.cli [options] {hindsight,expert,learn} ...``.
The sub-commands, options and output files are those of :mod:`udagger.cli`;
this module only adds the market options. The defaults reproduce the protocol
of the physician study: 4,000 training episodes, DAgger with 10 episodes per
iteration and a retraining after every iteration, 100 training epochs,
actions sampled from the learned policy, and expert MIPs stopped at a 2% gap
or after 30 seconds.
"""

from __future__ import annotations

import argparse

import numpy as np
from udagger.cli import Application, main

from udagger_adwords.experts import EXPERT_KINDS, Expert
from udagger_adwords.instance import Market, make_market, sample_episode, split_seeds
from udagger_adwords.problem import AdwordsProblem
from udagger_adwords.solver import offline_optimum


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Options of the market (budgets calibrated in ``scripts/calibrate.py``)."""
    parser.add_argument("--market-seed", type=int, default=0)
    parser.add_argument("--tightness", type=float, default=1.2)
    parser.add_argument("--premium-tightness", type=float, default=0.7)
    parser.add_argument("--regular-std", type=float, default=20.0)
    parser.add_argument(
        "--premium-std",
        type=float,
        default=10.0,
        help="standard deviation of the number of premium queries",
    )


def market(args: argparse.Namespace) -> Market:
    """Market defined by the options."""
    return make_market(
        seed=args.market_seed,
        tightness=args.tightness,
        premium_tightness=args.premium_tightness,
        count_std=(args.regular_std, args.premium_std),
    )


def make_problem(args: argparse.Namespace) -> AdwordsProblem:
    """AdWords problem of the options."""
    return AdwordsProblem(market(args))


def make_expert(args: argparse.Namespace) -> Expert:
    """Expert of the options."""
    return Expert(
        args.expert,
        n_scenarios=args.scenarios,
        mip_gap=args.gap,
        time_limit=args.time_limit,
        sample_vote=args.sample_vote,
    )


def seeds(split: str, n: int, run_seed: int) -> list[int]:
    """Episode seeds of a split."""
    return split_seeds(split, n, run_seed=run_seed)


def hindsight(args: argparse.Namespace, seed: int) -> float:
    """Offline optimum (revenue) of an episode."""
    mkt = market(args)
    episode = sample_episode(mkt, np.random.default_rng(seed))
    return offline_optimum(mkt, np.bincount(episode.types, minlength=mkt.n_types))


APP = Application(
    name="adwords",
    objective="revenue",
    expert_kinds=EXPERT_KINDS,
    add_arguments=add_arguments,
    make_problem=make_problem,
    make_expert=make_expert,
    seeds=seeds,
    hindsight=hindsight,
)


if __name__ == "__main__":
    # Import the application by module name so that its functions can be sent
    # to worker processes (not as attributes of ``__main__``).
    from udagger_adwords.cli import APP as _APP

    main(_APP, __doc__.splitlines()[0])
