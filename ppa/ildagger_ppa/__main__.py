"""Learn a PPA policy with ``ildagger`` (one-shot cloning or DAgger).

Usage (from ``ppa/``, with the repository root on ``PYTHONPATH``)::

    python -m ildagger_ppa --mode dagger --episodes 4000 --workers 10 \\
        --out results/saa5 -- --stoch 1 --new 1 --scenarios 5

Options after ``--`` are those of the original scripts (``utility.py``) and
define the instance and the expert (``--stoch``, ``--new``, ``--scenarios``,
``--mipgap``, ``--time_limit``); ``--expert greedy`` uses the myopic rule.
Episode seeds follow the original evaluation: ``5001 + n`` for test and
``6001 + n`` for validation; training episodes use ``10**6 * (seed + 1) + i``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from ildagger import EpisodeResult, LearnConfig, Runner, learn

from ildagger_ppa.problem import (
    PPAExpert,
    PPAGreedy,
    PPAProblem,
    config_from_args,
)

TEST_SEED = 5001
VALID_SEED = 6001


def _write(path: Path, results: list[EpisodeResult]) -> None:
    keys = list(results[0].stats) if results else []
    lines = [",".join(["seed", *keys, "solve_time"])]
    for r in results:
        values = [str(r.stats[k]) for k in keys]
        lines.append(",".join([str(r.seed), *values, f"{r.solve_time:.4f}"]))
    path.write_text("\n".join(lines) + "\n")


def _mean(results: list[EpisodeResult]) -> float:
    return float(np.mean([r.objective for r in results]))


def main() -> None:
    """Parse the command line, learn and evaluate a policy."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--expert", choices=("optimisation", "greedy"), default="optimisation"
    )
    parser.add_argument("--mode", choices=("oneshot", "dagger"), default="dagger")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=4000)
    parser.add_argument("--per-iteration", type=int, default=10)
    parser.add_argument("--n-valid", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--policy-action", choices=("sample", "mode"), default="sample")
    parser.add_argument("--out", required=True)
    parser.add_argument("ppa", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    ppa_args = args.ppa[1:] if args.ppa[:1] == ["--"] else args.ppa
    config = config_from_args(ppa_args)

    problem = PPAProblem(config)
    expert = PPAGreedy() if args.expert == "greedy" else PPAExpert(config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runner = Runner(problem, args.workers)
    torch.set_num_threads(max(1, args.workers))
    start = time.perf_counter()
    cfg = LearnConfig(
        mode=args.mode,
        seed=args.seed,
        per_iteration=args.per_iteration,
        epochs=args.epochs,
        policy_action=args.policy_action,
    )
    base = 10**6 * (args.seed + 1)
    res = learn(
        runner,
        expert,
        cfg,
        train_seeds=[base + i for i in range(args.episodes)],
        valid_seeds=[VALID_SEED + n for n in range(args.n_valid)],
        log=lambda line: print(line, flush=True),
    )
    test_seeds = [TEST_SEED + n for n in range(args.n_test)]
    by_rule = {}
    for policy in (res.policy, res.policy.with_sampling(not res.policy.sample)):
        rule = "sample" if policy.sample else "mode"
        by_rule[rule] = runner.run(test_seeds, policy=policy)
        _write(out / f"test_{rule}.csv", by_rule[rule])
    runner.close()
    summary = {
        "expert": expert.name,
        "mode": args.mode,
        "seed": args.seed,
        "episodes": args.episodes,
        "best": res.best,
        "test_cost_sample": _mean(by_rule["sample"]),
        "test_cost_mode": _mean(by_rule["mode"]),
        "label_seconds": res.label_seconds,
        "wall_seconds": time.perf_counter() - start,
        "curve": res.curve,
    }
    (out / "test.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "curve"}))


if __name__ == "__main__":
    main()
