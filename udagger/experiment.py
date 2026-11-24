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

While a DAgger run is in progress, its folder also holds ``partial.json``
(the summary so far: best policy, curve and ``milestones``) and
``policy_partial.npz`` (the best policy so far), updated at every checkpoint,
so that a run stopped by a time limit still leaves usable results. Every
``--test-every`` episodes, the best policy so far is tested and added to
``milestones``: the test results that a run with that smaller budget would
give. Both files are removed when the run completes, and ``test.json`` keeps
the milestones.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from udagger import checkpoint as ckpt
from udagger.learner import LearnConfig, learn, mean_objective
from udagger.policy import NumpyPolicy
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
    parser.add_argument(
        "--test-every",
        type=int,
        default=500,
        help="DAgger: test the best policy so far every this many episodes "
        "(0: never); written to partial.json and test.json",
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


def _test_rules(
    runner: Runner, policy: NumpyPolicy, test_seeds: list[int]
) -> dict[str, list[EpisodeResult]]:
    """Test a policy with both action rules (sampling and most likely action)."""
    by_rule: dict[str, list[EpisodeResult]] = {}
    for p in (policy, policy.with_sampling(not policy.sample)):
        by_rule["sample" if p.sample else "mode"] = runner.run(test_seeds, policy=p)
    return by_rule


def _write_atomic(path: Path, text: str) -> None:
    """Write a file so that a job killed meanwhile leaves the previous one."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _save_policy(path: Path, policy: NumpyPolicy) -> None:
    """Save the weights of a policy (``arr_0``, ``arr_1``, ...: w, b per layer)."""
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, *[a for wb in policy.weights for a in wb])
    os.replace(tmp, path)


def _previous_milestones(path: Path, context: dict) -> list[dict]:
    """Milestones of an interrupted run with the same settings (for a resume)."""
    if not path.exists():
        return []
    previous = json.loads(path.read_text())
    if previous.get("context") != context:
        return []
    return list(previous.get("milestones", []))


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
    fingerprint = json.loads(
        json.dumps(meta if context is None else context, default=str)
    )
    test_every = max(0, getattr(args, "test_every", 0))
    partial_json = out / "partial.json"
    partial_policy = out / "policy_partial.npz"
    milestones = _previous_milestones(partial_json, fingerprint)

    def progress(best: NumpyPolicy, row: dict, curve: list[dict], done: int) -> None:
        """Write the results so far; test the best policy at each milestone."""
        tested = max((m["episodes"] for m in milestones), default=0)
        if test_every and done // test_every > tested // test_every:
            rules = _test_rules(runner, best, test_seeds)
            milestone = {
                "episodes": done,
                "best": row,
                f"test_{objective}": mean_objective(rules[args.policy_action]),
                f"test_{objective}_mode": mean_objective(rules["mode"]),
                f"test_{objective}_sample": mean_objective(rules["sample"]),
            }
            milestones.append(milestone)
            print(json.dumps({"milestone": milestone}), flush=True)
        _save_policy(partial_policy, best)
        partial = {
            **meta,
            "mode": args.mode,
            "seed": args.seed,
            "episodes": len(train_seeds),
            "episodes_done": done,
            "best": row,
            "policy_action": args.policy_action,
            "milestones": milestones,
            "curve": curve,
            "context": fingerprint,
        }
        _write_atomic(partial_json, json.dumps(partial, indent=2, default=str))

    res = learn(
        runner,
        expert,
        learn_config(args),
        train_seeds=train_seeds,
        valid_seeds=valid_seeds,
        log=lambda line: print(line, flush=True),
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=max(1, every),
        context=fingerprint,
        on_progress=progress if args.mode == "dagger" else None,
    )
    learned = time.perf_counter()
    by_rule = _test_rules(runner, res.policy, test_seeds)
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
    if milestones:
        summary["milestones"] = milestones
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
    partial_json.unlink(missing_ok=True)
    partial_policy.unlink(missing_ok=True)
    print(json.dumps({k: v for k, v in summary.items() if k != "curve"}))
    return summary
