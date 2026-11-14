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
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ilco_adwords.experts import EXPERT_KINDS, Expert
from ilco_adwords.market import Market, make_market, sample_episode
from ilco_adwords.policy import NumpyPolicy, train_policy
from ilco_adwords.rollout import EpisodeResult, Runner, split_seeds
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
        lines.append(
            f"{r.seed},{r.revenue},{r.n_queries[0]},{r.n_queries[1]},"
            f"{r.served[0]},{r.served[1]},{r.budget_used:.5f},"
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
    runner = Runner(market, args.workers)
    start = time.perf_counter()
    results = runner.run(split_seeds("test", args.n_test), expert=expert)
    runner.close()
    wall = time.perf_counter() - start
    rev = np.array([r.revenue for r in results])
    tag = f"{expert.name}_gap{args.gap:g}"
    summary = {
        "expert": expert.name,
        "gap": args.gap,
        "mean_revenue": float(rev.mean()),
        "wall_seconds": wall,
        "solve_seconds": float(sum(r.solve_time for r in results)),
        "decisions": int(sum(r.n_queries.sum() for r in results)),
    }
    _write_episodes(out / f"expert_{tag}.csv", results, summary)
    print(json.dumps(summary))


def _stack(results: list[EpisodeResult]) -> dict[str, np.ndarray]:
    keep = [r for r in results if r.x is not None]
    data = {
        k: np.concatenate([getattr(r, k) for r in keep])
        for k in ("x", "target", "mask", "epoch_time", "expert_action", "query_class")
    }
    data["episode"] = np.concatenate(
        [np.full(r.n_labels, r.seed) for r in keep]
    ).astype(np.int64)
    return data


def _merge(a: dict[str, np.ndarray] | None, b: dict[str, np.ndarray]) -> dict:
    if a is None:
        return b
    return {k: np.concatenate([a[k], b[k]]) for k in a}


def _evaluate(runner: Runner, policy: NumpyPolicy, split: str, n: int) -> list:
    return runner.run(split_seeds(split, n), policy=policy)


def _train(
    args: argparse.Namespace,
    data: dict[str, np.ndarray],
    seed: int,
    init: NumpyPolicy | None = None,
) -> tuple[NumpyPolicy, dict]:
    policy, stats = train_policy(
        data["x"],
        data["target"],
        data["mask"],
        seed=seed,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        patience=args.patience,
        init=init,
    )
    return policy.with_sampling(args.policy_action == "sample"), asdict(stats)


def cmd_learn(args: argparse.Namespace) -> None:
    """Learn a policy from an expert with one-shot cloning or DAgger."""
    market = _market(args)
    expert = _expert(args)
    tag = f"{expert.name}_gap{args.gap:g}_{args.mode}_s{args.seed}"
    out = Path(args.out) / tag
    out.mkdir(parents=True, exist_ok=True)
    runner = Runner(market, args.workers)
    # The pool workers are idle while the network trains.
    torch.set_num_threads(max(1, args.workers))
    start = time.perf_counter()
    label_seconds = 0.0
    curve = []
    data = None
    best = {"val": -np.inf}

    def checkpoint(policy: NumpyPolicy, it: int, episodes: int, stats: dict) -> None:
        nonlocal best
        val = _evaluate(runner, policy, "valid", args.n_valid)
        val_rev = float(np.mean([r.revenue for r in val]))
        row = {
            "iteration": it,
            "episodes": episodes,
            "samples": int(data["x"].shape[0]),
            "label_seconds": label_seconds,
            "wall_seconds": time.perf_counter() - start,
            "val_revenue": val_rev,
            **stats,
        }
        curve.append(row)
        print(json.dumps(row), flush=True)
        if val_rev > best["val"]:
            best = {"val": val_rev, "policy": policy, "row": row}

    train_seeds = split_seeds("train", args.episodes, run_seed=args.seed)
    if args.mode == "oneshot":
        results = runner.run(train_seeds, expert=expert, collect=True)
        label_seconds = sum(r.solve_time for r in results)
        data_all = _stack(results)
        sizes = sorted(
            {int(s) for s in args.sizes if s <= args.episodes} | {args.episodes}
        )
        for size in sizes:
            keep = np.isin(data_all["episode"], train_seeds[:size])
            data = {k: v[keep] for k, v in data_all.items()}
            policy, stats = _train(args, data, args.seed)
            checkpoint(policy, 0, size, stats)
        data = data_all
    else:
        per_it = args.per_iteration
        if args.episodes % per_it:
            raise ValueError("--episodes must be a multiple of --per-iteration")
        n_it = args.episodes // per_it
        policy = None
        for it in range(n_it):
            beta = 1.0 if it == 0 else args.beta_decay**it
            seeds = train_seeds[it * per_it : (it + 1) * per_it]
            results = runner.run(
                seeds, expert=expert, policy=policy, beta=beta, collect=True
            )
            label_seconds += sum(r.solve_time for r in results)
            data = _merge(data, _stack(results))
            last = it == n_it - 1
            if (it + 1) % args.retrain_every == 0 or it == 0 or last:
                init = policy if args.warm_start else None
                train_seed = args.seed * 100_000 + it
                policy, stats = _train(args, data, train_seed, init)
                checkpoint(policy, it + 1, (it + 1) * per_it, stats)

    test = _evaluate(runner, best["policy"], "test", args.n_test)
    # The same weights with the other action rule (mode vs. sampling).
    other = best["policy"].with_sampling(not best["policy"].sample)
    test_other = _evaluate(runner, other, "test", args.n_test)
    runner.close()
    summary = {
        "expert": expert.name,
        "kind": expert.kind,
        "scenarios": expert.n_scenarios,
        "gap": args.gap,
        "mode": args.mode,
        "seed": args.seed,
        "episodes": args.episodes,
        "best": best["row"],
        "policy_action": args.policy_action,
        "test_revenue": float(np.mean([r.revenue for r in test])),
        "test_revenue_mode": float(
            np.mean([r.revenue for r in (test if other.sample else test_other)])
        ),
        "test_revenue_sample": float(
            np.mean([r.revenue for r in (test_other if other.sample else test)])
        ),
        "label_seconds": label_seconds,
        "wall_seconds": time.perf_counter() - start,
        "curve": curve,
    }
    _write_episodes(out / "test.csv", test, summary)
    # Per-episode results of both action rules, for paired comparisons.
    rule_other = "sample" if other.sample else "mode"
    _write_episodes(out / f"test_{args.policy_action}.csv", test, summary)
    _write_episodes(out / f"test_{rule_other}.csv", test_other, summary)
    np.savez_compressed(
        out / "policy.npz", *[a for wb in best["policy"].weights for a in wb]
    )
    if args.save_data:
        np.savez_compressed(
            out / "labels.npz",
            epoch_time=data["epoch_time"],
            expert_action=data["expert_action"],
            query_class=data["query_class"],
            episode=data["episode"],
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
