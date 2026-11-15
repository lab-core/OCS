"""Command-line building blocks shared by the applications.

An application's command line adds its own options (instance, expert) and
calls :func:`add_learn_arguments`, :func:`run_learning`,
:func:`evaluate_expert` and :func:`write_episodes`, so that both examples
expose the same ``learn`` and ``expert`` commands and write the same files:

* ``expert_<name>_gap<g>.csv/.json``: an expert used online on the test set;
* ``<name>_gap<g>_<mode>_s<seed>/``: a learning run, with ``test.csv`` (the
  configured action rule), ``test_sample.csv``, ``test_mode.csv``,
  ``test.json`` (summary and learning curve), ``policy.npz`` and, with
  ``--save-data``, ``labels.npz`` (expert actions and ``Episode.info``).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from udagger import checkpoint as ckpt
from udagger.learner import LearnConfig, learn, mean_objective
from udagger.problem import Expert
from udagger.rollout import EpisodeResult, Runner


def add_learn_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the options of a learning run (defaults: physician-study protocol)."""
    parser.add_argument("--mode", choices=("oneshot", "dagger"), default="oneshot")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=4000)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="*",
        default=[],
        help="one-shot: also train on these sizes and keep the best on validation",
    )
    parser.add_argument("--per-iteration", type=int, default=10)
    parser.add_argument("--retrain-every", type=int, default=1)
    parser.add_argument("--beta-decay", type=float, default=0.8)
    parser.add_argument("--policy-action", choices=("sample", "mode"), default="sample")
    parser.add_argument(
        "--warm-start", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="iterations between checkpoints (0: none); a rerun resumes",
    )


def learn_config(args: argparse.Namespace) -> LearnConfig:
    """Build the learning settings from parsed options."""
    return LearnConfig(
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


def write_episodes(
    path: Path, results: list[EpisodeResult], objective: str, summary: dict
) -> None:
    """Write one CSV row per episode and the summary as JSON next to it.

    Args:
        path: CSV file to write.
        results: Episode results.
        objective: Column name of the objective (e.g. ``revenue``).
        summary: Written to the same path with a ``.json`` suffix.
    """
    keys = [k for k in (results[0].stats if results else {}) if k != "objective"]
    lines = [",".join(["seed", objective, *keys, "solve_time", "max_gap"])]
    for r in results:
        values = [r.stats["objective"], *(r.stats[k] for k in keys)]
        cells = [str(r.seed), *map(str, values), f"{r.solve_time:.4f}"]
        lines.append(",".join([*cells, f"{r.max_gap:.5f}"]))
    path.write_text("\n".join(lines) + "\n")
    path.with_suffix(".json").write_text(json.dumps(summary, indent=2))


def evaluate_expert(
    runner: Runner,
    expert: Expert,
    test_seeds: list[int],
    out: Path,
    tag: str,
    objective: str,
    meta: dict,
) -> dict:
    """Use an expert online on the test episodes and write the results.

    Returns:
        The summary written next to ``expert_<tag>.csv``.
    """
    start = time.perf_counter()
    results = runner.run(test_seeds, expert=expert)
    summary = {
        **meta,
        f"mean_{objective}": mean_objective(results),
        "wall_seconds": time.perf_counter() - start,
        "solve_seconds": float(sum(r.solve_time for r in results)),
        "decisions": int(sum(r.n_decisions for r in results)),
    }
    out.mkdir(parents=True, exist_ok=True)
    write_episodes(out / f"expert_{tag}.csv", results, objective, summary)
    return summary


def run_learning(
    runner: Runner,
    expert: Expert,
    args: argparse.Namespace,
    train_seeds: list[int],
    valid_seeds: list[int],
    test_seeds: list[int],
    out: Path,
    objective: str,
    meta: dict,
    context: dict | None = None,
) -> dict:
    """Learn from an expert, test both action rules and write the results.

    Args:
        runner: Runner of the problem.
        expert: Expert providing the labels.
        args: Options of :func:`add_learn_arguments`.
        train_seeds: Training episodes (their number is the budget).
        valid_seeds: Validation episodes (policy selection).
        test_seeds: Test episodes.
        out: Folder of the run.
        objective: Column name of the objective in the CSV files.
        meta: Extra entries of the summary (expert, gap, ...).
        context: Settings that a checkpoint must match to be resumed
            (default: ``meta``).

    Returns:
        The summary written to ``test.json``.
    """
    out.mkdir(parents=True, exist_ok=True)
    # The pool workers are idle while the network trains.
    torch.set_num_threads(max(1, runner.workers))
    every = getattr(args, "checkpoint_every", 0)
    checkpoint_dir = out / "checkpoint" if every > 0 else None
    res = learn(
        runner,
        expert,
        learn_config(args),
        train_seeds=train_seeds,
        valid_seeds=valid_seeds,
        log=lambda line: print(line, flush=True),
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=max(1, every),
        context=json.loads(
            json.dumps(meta if context is None else context, default=str)
        ),
    )
    learned = time.perf_counter()
    by_rule: dict[str, list[EpisodeResult]] = {}
    for policy in (res.policy, res.policy.with_sampling(not res.policy.sample)):
        by_rule["sample" if policy.sample else "mode"] = runner.run(
            test_seeds, policy=policy
        )
    summary = {
        **meta,
        "mode": args.mode,
        "seed": args.seed,
        "episodes": len(train_seeds),
        "best": res.best,
        "policy_action": args.policy_action,
        f"test_{objective}": mean_objective(by_rule[args.policy_action]),
        f"test_{objective}_mode": mean_objective(by_rule["mode"]),
        f"test_{objective}_sample": mean_objective(by_rule["sample"]),
        "label_seconds": res.label_seconds,
        "wall_seconds": res.wall_seconds + time.perf_counter() - learned,
        "curve": res.curve,
    }
    write_episodes(out / "test.csv", by_rule[args.policy_action], objective, summary)
    for rule, results in by_rule.items():
        write_episodes(out / f"test_{rule}.csv", results, objective, summary)
    weights = [a for wb in res.policy.weights for a in wb]
    np.savez_compressed(out / "policy.npz", *weights)
    if args.save_data:
        skip = ("x", "target", "mask")
        np.savez_compressed(
            out / "labels.npz", **{k: v for k, v in res.data.items() if k not in skip}
        )
    if checkpoint_dir is not None:
        ckpt.remove(checkpoint_dir)  # the run is complete
    print(json.dumps({k: v for k, v in summary.items() if k != "curve"}))
    return summary
