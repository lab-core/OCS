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
import time
from pathlib import Path

import numpy as np
import torch
from udagger import EpisodeResult, LearnConfig, NumpyPolicy, Runner, learn

from ilco_adwords.experts import EXPERT_KINDS, Expert
from ilco_adwords.market import Market, make_market, sample_episode, split_seeds
from ilco_adwords.problem import AdwordsProblem
from ilco_adwords.solver import offline_optimum


def _market(args: argparse.Namespace) -> Market:
    return make_market(
        seed=args.market_seed,
        tightness=args.tightness,
        premium_tightness=args.premium_tightness,
        count_std=(args.regular_std, args.premium_std),
    )


def _write_episodes(path: Path, results: list[EpisodeResult], extra: dict) -> None:
    lines = [
        "seed,revenue,n_regular,n_premium,served_regular,served_premium,"
        "budget_used,solve_time,max_gap"
    ]
    for r in results:
        st = r.stats
        lines.append(
            f"{r.seed},{st['objective']},{st['n_regular']},{st['n_premium']},"
            f"{st['served_regular']},{st['served_premium']},{st['budget_used']:.5f},"
            f"{r.solve_time:.4f},{r.max_gap:.5f}"
        )
    path.write_text("\n".join(lines) + "\n")
    path.with_suffix(".json").write_text(json.dumps(extra, indent=2))


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
    market = _market(args)
    expert = _expert(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runner = Runner(AdwordsProblem(market), args.workers)
    start = time.perf_counter()
    results = runner.run(split_seeds("test", args.n_test), expert=expert)
    runner.close()
    wall = time.perf_counter() - start
    rev = np.array([r.objective for r in results])
    tag = f"{expert.name}_gap{args.gap:g}"
    summary = {
        "expert": expert.name,
        "gap": args.gap,
        "mean_revenue": float(rev.mean()),
        "wall_seconds": wall,
        "solve_seconds": float(sum(r.solve_time for r in results)),
        "decisions": int(
            sum(r.stats["n_regular"] + r.stats["n_premium"] for r in results)
        ),
    }
    _write_episodes(out / f"expert_{tag}.csv", results, summary)
    print(json.dumps(summary))


def _evaluate(runner: Runner, policy: NumpyPolicy, split: str, n: int) -> list:
    return runner.run(split_seeds(split, n), policy=policy)


def _mean(results: list[EpisodeResult]) -> float:
    return float(np.mean([r.objective for r in results]))


def cmd_learn(args: argparse.Namespace) -> None:
    """Learn a policy from an expert with one-shot cloning or DAgger."""
    market = _market(args)
    expert = _expert(args)
    tag = f"{expert.name}_gap{args.gap:g}_{args.mode}_s{args.seed}"
    out = Path(args.out) / tag
    out.mkdir(parents=True, exist_ok=True)
    runner = Runner(AdwordsProblem(market), args.workers)
    # The pool workers are idle while the network trains.
    torch.set_num_threads(max(1, args.workers))
    start = time.perf_counter()
    cfg = LearnConfig(
        mode=args.mode,
        seed=args.seed,
        sizes=args.sizes,
        per_iteration=args.per_iteration,
        retrain_every=args.retrain_every,
        beta_decay=args.beta_decay,
        warm_start=args.warm_start,
        policy_action=args.policy_action,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
    )
    res = learn(
        runner,
        expert,
        cfg,
        train_seeds=split_seeds("train", args.episodes, run_seed=args.seed),
        valid_seeds=split_seeds("valid", args.n_valid),
        log=lambda line: print(line, flush=True),
    )
    test = _evaluate(runner, res.policy, "test", args.n_test)
    # The same weights with the other action rule (mode vs. sampling).
    other = res.policy.with_sampling(not res.policy.sample)
    test_other = _evaluate(runner, other, "test", args.n_test)
    runner.close()
    by_rule = {
        "sample" if res.policy.sample else "mode": test,
        "sample" if other.sample else "mode": test_other,
    }
    summary = {
        "expert": expert.name,
        "kind": expert.kind,
        "scenarios": expert.n_scenarios,
        "gap": args.gap,
        "mode": args.mode,
        "seed": args.seed,
        "episodes": args.episodes,
        "best": res.best,
        "policy_action": args.policy_action,
        "test_revenue": _mean(test),
        "test_revenue_mode": _mean(by_rule["mode"]),
        "test_revenue_sample": _mean(by_rule["sample"]),
        "label_seconds": res.label_seconds,
        "wall_seconds": time.perf_counter() - start,
        "curve": res.curve,
    }
    _write_episodes(out / "test.csv", test, summary)
    # Per-episode results of both action rules, for paired comparisons.
    for rule, results in by_rule.items():
        _write_episodes(out / f"test_{rule}.csv", results, summary)
    np.savez_compressed(
        out / "policy.npz", *[a for wb in res.policy.weights for a in wb]
    )
    if args.save_data:
        np.savez_compressed(
            out / "labels.npz",
            **{
                k: res.data[k]
                for k in ("epoch_time", "expert_action", "query_class", "episode")
            },
        )
    print(json.dumps({k: v for k, v in summary.items() if k != "curve"}))


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
    p_learn.add_argument("--mode", choices=("oneshot", "dagger"), default="oneshot")
    p_learn.add_argument("--seed", type=int, default=0)
    p_learn.add_argument("--episodes", type=int, default=4000)
    p_learn.add_argument(
        "--sizes",
        type=int,
        nargs="*",
        default=[],
        help="one-shot: also train on these sizes and keep the best on validation",
    )
    p_learn.add_argument("--per-iteration", type=int, default=10)
    p_learn.add_argument("--retrain-every", type=int, default=1)
    p_learn.add_argument("--beta-decay", type=float, default=0.8)
    p_learn.add_argument(
        "--policy-action", choices=("sample", "mode"), default="sample"
    )
    p_learn.add_argument(
        "--warm-start", action=argparse.BooleanOptionalAction, default=True
    )
    p_learn.add_argument("--epochs", type=int, default=100)
    p_learn.add_argument("--batch-size", type=int, default=256)
    p_learn.add_argument("--patience", type=int, default=None)
    p_learn.add_argument("--save-data", action="store_true")
    p_learn.set_defaults(func=cmd_learn)
    return parser


def main() -> None:
    """Parse the command line and run the selected sub-command."""
    args = build_parser().parse_args()
    args.out = args.out + args.out_suffix
    args.func(args)


if __name__ == "__main__":
    main()
