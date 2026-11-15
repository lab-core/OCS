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
            episode, or ``(value, gap)`` with the final relative MIP gap when
            the optimum is solved with a time limit.
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
    hindsight: Callable[[argparse.Namespace, int], float | tuple[float, float]]
    hindsight_time_limit: float | None = None


def _hindsight_task(task: tuple) -> tuple[int, float, float]:
    hindsight, args, seed = task
    try:
        out = hindsight(args, seed)
    except Exception as err:
        # One failed episode (e.g. no solution within the time limit) must
        # not lose the results of the whole split.
        print(f"hindsight of episode {seed} failed: {err!r}", flush=True)
        return seed, float("nan"), float("nan")
    value, gap = out if isinstance(out, tuple) else (out, 0.0)
    return seed, float(value), float("nan") if gap is None else float(gap)


def cmd_hindsight(app: Application, args: argparse.Namespace) -> None:
    """Compute the offline optimum of the validation and test episodes."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for split, n in (("valid", args.n_valid), ("test", args.n_test)):
        tasks = [(app.hindsight, args, s) for s in app.seeds(split, n, 0)]
        with mp.get_context("spawn").Pool(args.workers) as pool:
            rows = pool.map(_hindsight_task, tasks)
        text = "seed,hindsight,gap\n" + "".join(f"{s},{v},{g}\n" for s, v, g in rows)
        (out / f"hindsight_{split}.csv").write_text(text)
        mean = sum(v for _, v, _ in rows) / max(1, len(rows))
        print(split, "mean hindsight", mean)
        failed = [s for s, v, _ in rows if v != v]
        if failed:
            print(
                f"warning: {len(failed)} {split} optima failed, e.g. seed {failed[0]}"
            )
        not_optimal = [s for s, v, g in rows if v == v and not g <= 1e-6]
        if not_optimal:
            print(
                f"warning: {len(not_optimal)} {split} optima not proven "
                f"(time limit), e.g. seed {not_optimal[0]}; see the gap column"
            )


def cmd_expert(app: Application, args: argparse.Namespace) -> None:
    """Evaluate an expert used directly as an online policy."""
    expert = app.make_expert(args)
    tag = f"{expert.name}_gap{args.gap:g}{args.tag}"
    options = _options(args)
    if _done(Path(args.out) / f"expert_{tag}.json", options, args.force):
        return
    runner = Runner(app.make_problem(args), args.workers)
    try:
        summary = evaluate_expert(
            runner,
            expert,
            app.seeds("test", args.n_test, 0),
            Path(args.out),
            tag=tag,
            objective=app.objective,
            meta={"expert": expert.name, "gap": args.gap, "options": options},
        )
    finally:
        runner.close()
    print(json.dumps(summary))


def cmd_learn(app: Application, args: argparse.Namespace) -> None:
    """Learn a policy from an expert with one-shot cloning or DAgger."""
    expert = app.make_expert(args)
    tag = f"{expert.name}_gap{args.gap:g}_{args.mode}_s{args.seed}{args.tag}"
    options = _options(args)
    if _done(Path(args.out) / tag / "test.json", options, args.force):
        return
    # Options that only matter after learning do not prevent a resume.
    context = {k: v for k, v in options.items() if k not in _AFTER_LEARNING}
    runner = Runner(app.make_problem(args), args.workers)
    try:
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
                "options": options,
            },
            context=context,
        )
    finally:
        runner.close()


# Options that do not change the experiment (where results go, parallelism,
# whether labels are saved, checkpoints, reruns): ignored when comparing runs.
_NOT_EXPERIMENT = (
    "func",
    "workers",
    "out",
    "out_suffix",
    "save_data",
    "checkpoint_every",
    "force",
)
# Options only used after learning (test episodes, folder name).
_AFTER_LEARNING = ("n_test", "tag", "command")


def _options(args: argparse.Namespace) -> dict:
    return {k: v for k, v in vars(args).items() if k not in _NOT_EXPERIMENT}


def _done(summary: Path, options: dict, force: bool) -> bool:
    """Whether the run is already complete (then skip it).

    Raises:
        SystemExit: If the file holds a run made with other options.
    """
    if not summary.exists():
        return False
    previous = json.loads(summary.read_text()).get("options")
    if previous is not None and previous != options:
        changed = sorted(k for k in options if previous.get(k) != options[k])
        raise SystemExit(
            f"{summary} holds a run with other options ({changed}); "
            "use --tag to name this variant"
        )
    if force:
        return False
    print(f"{summary} is complete with the same options; skipped (--force reruns)")
    return True


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
    parser.add_argument(
        "--tag", default="", help="suffix of the result names (variants of options)"
    )
    parser.add_argument(
        "--force", action="store_true", help="rerun even if the run is complete"
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
