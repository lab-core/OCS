"""Generic command line of an application.

An application declares an :class:`Application` (its options, problem,
experts, episode seeds and hindsight optimum) and calls :func:`main`. Every
application then has the same sub-commands, options and output files:

* ``hindsight``: offline optimum of the validation and test episodes
  (``hindsight_valid.csv``, ``hindsight_test.csv``);
* ``expert``: an expert used online on the test episodes;
* ``learn``: imitation learning (one-shot or DAgger) from an expert.

See :mod:`udagger.experiment` for the files of ``expert`` and ``learn``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from udagger.experiment import add_learn_arguments, evaluate_expert, run_learning
from udagger.problem import Expert, Problem
from udagger.rollout import Runner


@dataclass(frozen=True)
class Application:
    """Description of an application for the generic command line.

    The callables must be module-level functions (they are sent to worker
    processes).

    Attributes:
        name: Name of the application (used in messages and ``$<NAME>_OUT``).
        objective: Column name of the objective in the result files.
        expert_kinds: Values of ``--expert``.
        add_arguments: Adds the instance options to the global parser.
        make_problem: Builds the problem from the parsed options.
        make_expert: Builds the expert from the parsed options.
        seeds: ``seeds(split, n, run_seed)`` returns the episode seeds of a
            split (``train``, ``valid`` or ``test``).
        hindsight: ``hindsight(args, seed)`` returns the offline optimum of an
            episode.
        hindsight_time_limit: Default ``--time-limit`` of ``hindsight``, or
            None if the optimum has no time limit.
    """

    name: str
    objective: str
    expert_kinds: Sequence[str]
    add_arguments: Callable[[argparse.ArgumentParser], None]
    make_problem: Callable[[argparse.Namespace], Problem]
    make_expert: Callable[[argparse.Namespace], Expert]
    seeds: Callable[[str, int, int], list[int]]
    hindsight: Callable[[argparse.Namespace, int], float]
    hindsight_time_limit: float | None = None


def _hindsight_task(task: tuple) -> tuple[int, float]:
    hindsight, args, seed = task
    return seed, float(hindsight(args, seed))


def cmd_hindsight(app: Application, args: argparse.Namespace) -> None:
    """Compute the offline optimum of the validation and test episodes."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for split, n in (("valid", args.n_valid), ("test", args.n_test)):
        tasks = [(app.hindsight, args, s) for s in app.seeds(split, n, 0)]
        with mp.get_context("spawn").Pool(args.workers) as pool:
            rows = pool.map(_hindsight_task, tasks)
        text = "seed,hindsight\n" + "".join(f"{s},{v}\n" for s, v in rows)
        (out / f"hindsight_{split}.csv").write_text(text)
        print(split, "mean hindsight", sum(v for _, v in rows) / max(1, len(rows)))


def cmd_expert(app: Application, args: argparse.Namespace) -> None:
    """Evaluate an expert used directly as an online policy."""
    expert = app.make_expert(args)
    runner = Runner(app.make_problem(args), args.workers)
    summary = evaluate_expert(
        runner,
        expert,
        app.seeds("test", args.n_test, 0),
        Path(args.out),
        tag=f"{expert.name}_gap{args.gap:g}",
        objective=app.objective,
        meta={"expert": expert.name, "gap": args.gap},
    )
    runner.close()
    print(json.dumps(summary))


def cmd_learn(app: Application, args: argparse.Namespace) -> None:
    """Learn a policy from an expert with one-shot cloning or DAgger."""
    expert = app.make_expert(args)
    runner = Runner(app.make_problem(args), args.workers)
    tag = f"{expert.name}_gap{args.gap:g}_{args.mode}_s{args.seed}"
    run_learning(
        runner,
        expert,
        args,
        train_seeds=app.seeds("train", args.episodes, args.seed),
        valid_seeds=app.seeds("valid", args.n_valid, 0),
        test_seeds=app.seeds("test", args.n_test, 0),
        out=Path(args.out) / tag,
        objective=app.objective,
        meta={
            "expert": expert.name,
            "kind": getattr(expert, "kind", expert.name),
            "scenarios": getattr(expert, "n_scenarios", 1),
            "gap": args.gap,
        },
    )
    runner.close()


def add_expert_arguments(parser: argparse.ArgumentParser, kinds: Sequence[str]) -> None:
    """Add the options of :class:`udagger.experts.ExpertSpec`."""
    parser.add_argument("--expert", choices=list(kinds), required=True)
    parser.add_argument("--scenarios", type=int, default=1)
    parser.add_argument("--gap", type=float, default=0.02)
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument(
        "--sample-vote",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="act by sampling the votes of the solved models (else majority)",
    )


def build_parser(app: Application, description: str) -> argparse.ArgumentParser:
    """Return the argument parser of an application."""
    parser = argparse.ArgumentParser(description=description)
    env = f"{app.name.upper()}_OUT"
    parser.add_argument("--out", default=os.environ.get(env, "results"))
    parser.add_argument(
        "--out-suffix", default="", help="appended to --out (e.g. per variant)"
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--n-valid", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=1000)
    app.add_arguments(parser)
    sub = parser.add_subparsers(dest="command", required=True)

    p_hind = sub.add_parser("hindsight")
    if app.hindsight_time_limit is not None:
        p_hind.add_argument(
            "--time-limit", type=float, default=app.hindsight_time_limit
        )
    p_hind.set_defaults(func=cmd_hindsight)

    p_exp = sub.add_parser("expert")
    add_expert_arguments(p_exp, app.expert_kinds)
    p_exp.set_defaults(func=cmd_expert)

    p_learn = sub.add_parser("learn")
    add_expert_arguments(p_learn, app.expert_kinds)
    add_learn_arguments(p_learn)
    p_learn.set_defaults(func=cmd_learn)
    return parser


def main(app: Application, description: str, argv: Sequence[str] | None = None) -> None:
    """Parse the command line and run the selected sub-command."""
    args = build_parser(app, description).parse_args(argv)
    args.out = args.out + args.out_suffix
    args.func(app, args)
