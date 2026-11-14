"""Command line of the physician-to-patient experiments (generic commands).

Usage: ``python -m udagger_ppa.cli [options] {hindsight,expert,learn} ...``.
The sub-commands, options and output files are those of :mod:`udagger.cli`;
this module only adds the instance options. The defaults are the protocol of
the original study: 4,000 training episodes, DAgger with 10 episodes per
iteration and a retraining after every iteration, 100 training epochs,
actions sampled from the learned policy, and expert MIPs stopped at a 2% gap
or after 30 seconds.
"""

from __future__ import annotations

import argparse
from dataclasses import fields

from udagger.cli import Application, main

from udagger_ppa.experts import EXPERT_KINDS, Expert
from udagger_ppa.instance import Instance, split_seeds
from udagger_ppa.problem import PPAProblem
from udagger_ppa.solver import offline_optimum


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """One option per field of :class:`udagger_ppa.instance.Instance`."""
    for f in fields(Instance):
        kind = int if f.name == "release_cap" else type(f.default)
        parser.add_argument(
            f"--{f.name.replace('_', '-')}", type=kind, default=f.default
        )


def instance(args: argparse.Namespace) -> Instance:
    """Instance defined by the options."""
    return Instance(**{f.name: getattr(args, f.name) for f in fields(Instance)})


def make_problem(args: argparse.Namespace) -> PPAProblem:
    """PPA problem of the options."""
    return PPAProblem(instance(args))


def make_expert(args: argparse.Namespace) -> Expert:
    """Expert of the options."""
    return Expert(
        args.expert,
        n_scenarios=args.scenarios,
        mip_gap=args.gap,
        time_limit=args.time_limit,
        sample_vote=args.sample_vote,
        instance=instance(args),
    )


def seeds(split: str, n: int, run_seed: int) -> list[int]:
    """Episode seeds of a split."""
    return split_seeds(split, n, run_seed=run_seed)


def hindsight(args: argparse.Namespace, seed: int) -> float:
    """Offline optimum (cost) of an episode."""
    return offline_optimum(instance(args), seed, args.time_limit)[0]


APP = Application(
    name="ppa",
    objective="cost",
    expert_kinds=EXPERT_KINDS,
    add_arguments=add_arguments,
    make_problem=make_problem,
    make_expert=make_expert,
    seeds=seeds,
    hindsight=hindsight,
    hindsight_time_limit=300.0,
)


if __name__ == "__main__":
    # Import the application by module name so that its functions can be sent
    # to worker processes (not as attributes of ``__main__``).
    from udagger_ppa.cli import APP as _APP

    main(_APP, __doc__.splitlines()[0])
