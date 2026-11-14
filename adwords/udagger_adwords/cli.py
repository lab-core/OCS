"""Command-line entry points for the AdWords experiments.

Sub-commands:

* ``hindsight``: offline optimum of validation/test episodes.
* ``expert``: online performance of an expert used directly as a policy.
* ``learn``: imitation learning (one-shot or DAgger) from an expert.

All results are written as CSV/JSON files under ``--out``. The defaults
reproduce the protocol of the physician-to-patient study (``ppa/``): 4,000
training episodes, DAgger with 10 episodes per iteration and a retraining
after every iteration, 100 training epochs, actions sampled from the learned
policy, and expert MIPs stopped at a 2% gap or after 30 seconds.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from udagger import Runner
from udagger.experiment import add_learn_arguments, evaluate_expert, run_learning

from udagger_adwords.experts import EXPERT_KINDS, Expert
from udagger_adwords.market import Market, make_market, sample_episode, split_seeds
from udagger_adwords.problem import AdwordsProblem
from udagger_adwords.solver import offline_optimum


def _market(args: argparse.Namespace) -> Market:
    return make_market(
        seed=args.market_seed,
        tightness=args.tightness,
        premium_tightness=args.premium_tightness,
        count_std=(args.regular_std, args.premium_std),
    )


def _hindsight_task(args: tuple[Market, int]) -> tuple[int, float]:
    market, seed = args
    ep = sample_episode(market, np.random.default_rng(seed))
    counts = np.bincount(ep.types, minlength=market.n_types)
    return seed, offline_optimum(market, counts)


def cmd_hindsight(args: argparse.Namespace) -> None:
    """Compute the offline optimum of the validation and test episodes."""
    import multiprocessing as mp

    market = _market(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for split, n in (("valid", args.n_valid), ("test", args.n_test)):
        seeds = split_seeds(split, n)
        with mp.get_context("spawn").Pool(args.workers) as pool:
            rows = pool.map(_hindsight_task, [(market, s) for s in seeds])
        text = "seed,hindsight\n" + "".join(f"{s},{v}\n" for s, v in rows)
        (out / f"hindsight_{split}.csv").write_text(text)
        print(split, "mean hindsight", np.mean([v for _, v in rows]))


def _expert(args: argparse.Namespace) -> Expert:
    return Expert(
        args.expert,
        n_scenarios=args.scenarios,
        mip_gap=args.gap,
        time_limit=args.time_limit,
        sample_vote=args.sample_vote,
    )


def cmd_expert(args: argparse.Namespace) -> None:
    """Evaluate an expert used directly as an online policy."""
    expert = _expert(args)
    runner = Runner(AdwordsProblem(_market(args)), args.workers)
    summary = evaluate_expert(
        runner,
        expert,
        split_seeds("test", args.n_test),
        Path(args.out),
        tag=f"{expert.name}_gap{args.gap:g}",
        objective="revenue",
        meta={"expert": expert.name, "gap": args.gap},
    )
    runner.close()
    print(json.dumps(summary))


def cmd_learn(args: argparse.Namespace) -> None:
    """Learn a policy from an expert with one-shot cloning or DAgger."""
    expert = _expert(args)
    runner = Runner(AdwordsProblem(_market(args)), args.workers)
    run_learning(
        runner,
        expert,
        args,
        train_seeds=split_seeds("train", args.episodes, run_seed=args.seed),
        valid_seeds=split_seeds("valid", args.n_valid),
        test_seeds=split_seeds("test", args.n_test),
        out=Path(args.out) / f"{expert.name}_gap{args.gap:g}_{args.mode}_s{args.seed}",
        objective="revenue",
        meta={
            "expert": expert.name,
            "kind": expert.kind,
            "scenarios": expert.n_scenarios,
            "gap": args.gap,
        },
    )
    runner.close()


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser of the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.environ.get("ADWORDS_OUT", "results"))
    parser.add_argument(
        "--out-suffix", default="", help="appended to --out (e.g. per variant)"
    )
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
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--n-valid", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=1000)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("hindsight").set_defaults(func=cmd_hindsight)

    def expert_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--expert", choices=EXPERT_KINDS, required=True)
        p.add_argument("--scenarios", type=int, default=1)
        p.add_argument("--gap", type=float, default=0.02)
        p.add_argument("--time-limit", type=float, default=30.0)
        p.add_argument(
            "--sample-vote",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="aggregated expert acts by sampling its votes (else majority)",
        )

    p_exp = sub.add_parser("expert")
    expert_args(p_exp)
    p_exp.set_defaults(func=cmd_expert)

    p_learn = sub.add_parser("learn")
    expert_args(p_learn)
    add_learn_arguments(p_learn)
    p_learn.set_defaults(func=cmd_learn)
    return parser


def main() -> None:
    """Parse the command line and run the selected sub-command."""
    args = build_parser().parse_args()
    args.out = args.out + args.out_suffix
    args.func(args)


if __name__ == "__main__":
    main()
