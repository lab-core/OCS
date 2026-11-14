"""Command-line entry points for the physician-to-patient experiments.

Sub-commands (as in the AdWords example):

* ``hindsight``: offline optimum of the validation and test episodes;
* ``expert``: online performance of an expert used directly as a policy;
* ``learn``: imitation learning (one-shot or DAgger) from an expert.

All results are written as CSV/JSON files under ``--out``. The defaults are
the protocol of the original study: 4,000 training episodes, DAgger with 10
episodes per iteration and a retraining after every iteration, 100 training
epochs, actions sampled from the learned policy, and expert MIPs stopped at a
2% gap or after 30 seconds.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from dataclasses import fields
from pathlib import Path

from udagger import Runner
from udagger.experiment import add_learn_arguments, evaluate_expert, run_learning

from udagger_ppa.experts import EXPERT_KINDS, Expert
from udagger_ppa.instance import Instance, split_seeds
from udagger_ppa.problem import PPAProblem, offline_optimum


def _instance(args: argparse.Namespace) -> Instance:
    return Instance(**{f.name: getattr(args, f.name) for f in fields(Instance)})


def _expert(args: argparse.Namespace) -> Expert:
    return Expert(
        args.expert,
        instance=_instance(args),
        n_scenarios=args.scenarios,
        mip_gap=args.gap,
        time_limit=args.time_limit,
        sample_vote=args.sample_vote,
    )


def _hindsight_task(args: tuple[Instance, int, float]) -> tuple[int, float]:
    instance, seed, time_limit = args
    return seed, offline_optimum(instance, seed, time_limit)[0]


def cmd_hindsight(args: argparse.Namespace) -> None:
    """Compute the offline optimum of the validation and test episodes."""
    instance = _instance(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for split, n in (("valid", args.n_valid), ("test", args.n_test)):
        tasks = [(instance, s, args.time_limit) for s in split_seeds(split, n)]
        with mp.get_context("spawn").Pool(args.workers) as pool:
            rows = pool.map(_hindsight_task, tasks)
        text = "seed,hindsight\n" + "".join(f"{s},{v}\n" for s, v in rows)
        (out / f"hindsight_{split}.csv").write_text(text)
        print(split, "mean hindsight", sum(v for _, v in rows) / max(1, len(rows)))


def cmd_expert(args: argparse.Namespace) -> None:
    """Evaluate an expert used directly as an online policy."""
    expert = _expert(args)
    runner = Runner(PPAProblem(_instance(args)), args.workers)
    summary = evaluate_expert(
        runner,
        expert,
        split_seeds("test", args.n_test),
        Path(args.out),
        tag=f"{expert.name}_gap{args.gap:g}",
        objective="cost",
        meta={"expert": expert.name, "gap": args.gap},
    )
    runner.close()
    print(json.dumps(summary))


def cmd_learn(args: argparse.Namespace) -> None:
    """Learn a policy from an expert with one-shot cloning or DAgger."""
    expert = _expert(args)
    runner = Runner(PPAProblem(_instance(args)), args.workers)
    run_learning(
        runner,
        expert,
        args,
        train_seeds=split_seeds("train", args.episodes, run_seed=args.seed),
        valid_seeds=split_seeds("valid", args.n_valid),
        test_seeds=split_seeds("test", args.n_test),
        out=Path(args.out) / f"{expert.name}_gap{args.gap:g}_{args.mode}_s{args.seed}",
        objective="cost",
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
    parser.add_argument("--out", default=os.environ.get("PPA_OUT", "results"))
    parser.add_argument(
        "--out-suffix", default="", help="appended to --out (e.g. per variant)"
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--n-valid", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=1000)
    for f in fields(Instance):
        kind = int if f.name == "release_cap" else type(f.default)
        parser.add_argument(
            f"--{f.name.replace('_', '-')}", type=kind, default=f.default
        )
    sub = parser.add_subparsers(dest="command", required=True)

    p_hind = sub.add_parser("hindsight")
    p_hind.add_argument("--time-limit", type=float, default=300.0)
    p_hind.set_defaults(func=cmd_hindsight)

    def expert_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--expert", choices=EXPERT_KINDS, required=True)
        p.add_argument("--scenarios", type=int, default=1)
        p.add_argument("--gap", type=float, default=0.02)
        p.add_argument("--time-limit", type=float, default=30.0)
        p.add_argument(
            "--sample-vote",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="act by sampling the votes of the solved models (else majority)",
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
