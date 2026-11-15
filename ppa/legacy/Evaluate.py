"""Parallel evaluation of PPA policies on the validation or test episodes.

This script reproduces the evaluation of ``Learn_Sim.py`` (episode ``n`` of a
split is generated with ``set_seed(base + n)``, with ``base = 5001`` for test
and ``6001`` for validation), but runs episodes in parallel and can evaluate:

* ``policy``: learned models ``NN_<k>.pth`` of a training folder, with the
  action sampled from the predicted distribution (``--action_type random``,
  the default of the original scripts) or the most likely one (``max``);
* ``expert``: the expert defined by the usual options (``--stoch``, ``--new``,
  ``--scenarios``, ``--mipgap``, ``--time_limit``) used directly as a policy;
* ``greedy``: preferred physician if possible, else a random eligible one;
* ``hindsight``: the offline optimum of each episode.

Extra options (``--target``, ``--split``, ``--sims``, ``--workers``,
``--policy_dir``, ``--models``, ``--out``) are read first; all remaining
options are passed to ``utility.parse_args``. One CSV row is written per
episode (and per model).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch

from Learn_Module import IO_learning
from NN import SupervisedNN
from OCS import AppointmentScheduler
from utility import Config, build_config, parse_args

SPLIT_SEED = {"test": 5001, "valid": 6001}
# Offset of the scenario seeds of experts evaluated online. Training uses
# config.seed * 10**7 + 100 (n runs + i) with --seed_in_name, and otherwise
# the original 1000 + 100 n + i, which overlaps the test (5001+) and
# validation (6001+) episode seeds; see Learn_Main.py.
EXPERT_SEED_OFFSET = 10_000_000
_W: dict = {}


def make_sim(config: Config) -> AppointmentScheduler:
    """Build a simulator exactly as in ``Learn_Main.py``.

    Args:
        config: Configuration built by ``utility.build_config``.

    Returns:
        The simulator with its solver set up.
    """
    p = config.ocs_param
    sim = AppointmentScheduler(
        n_patients1=p["N1"],
        sigma1=p["sigma1"],
        n_patients2=p["N2"],
        sigma2=p["sigma2"],
        n_phys=p["P"],
        n_slots=p["I"],
        k_max=p["k_max"],
        k_min=p["k_min"],
        c_miss_1=p["c_miss_1"],
        c_miss_2=p["c_miss_2"],
        c_overtime=p["co"],
        c_np_per=p["cnp"],
        physician_weights=p["physician_weights"],
        release_cap=p["Rmax"],
        cap_per_phys=p["L"],
        time_limit=config.time_limit,
        mipgap=config.mipgap,
        var=p["var"],
        seed=config.seed,
        out_dir=_W.get("tmp", "."),
        suffix=False,
    )
    sim.setup_solver()
    return sim


class _Chooser:
    """Minimal stand-in for ``SupervisedNN.random_action`` used by experts."""

    def __init__(self, rng: np.random.Generator, num_class: int) -> None:
        """Store the generator and the number of actions."""
        self.rng = rng
        self.num_class = num_class

    def random_action(self, prob: list[float]) -> int:
        """Sample an action from a distribution."""
        return int(self.rng.choice(range(self.num_class), p=prob))


def _init(config: Config, args: argparse.Namespace) -> None:
    torch.set_num_threads(1)
    _W["config"] = config
    _W["args"] = args
    _W["tmp"] = os.path.join(args.out + ".tmp", str(os.getpid()))
    os.makedirs(_W["tmp"], exist_ok=True)
    _W["sim"] = make_sim(config)
    _W["learner"] = IO_learning(_W["sim"], asdict(config), folder=_W["tmp"])
    _W["policies"] = {}


def _policy(model: int) -> SupervisedNN:
    if model not in _W["policies"]:
        cfg = asdict(_W["config"])
        cfg.update(train=False, model=model)
        folder = os.path.join(_W["args"].policy_dir, "")
        _W["policies"] = {model: SupervisedNN(_W["learner"].feature_names, cfg, folder)}
    return _W["policies"][model]


def _episode(task: tuple[int, int]) -> dict:
    model, n = task
    config, args = _W["config"], _W["args"]
    sim, learner = _W["sim"], _W["learner"]
    seed = SPLIT_SEED[args.split] + n
    sim.set_seed(seed)
    sim.generate_instance()
    start = time.perf_counter()
    row = {"seed": seed, "model": model}

    if args.target == "hindsight":
        sim.build_model(sim.patients)
        status, term, gap, sol_time = sim.solve(tee=False)
        _, obj = sim.extract_solution(sim.patients)
        row.update(total_cost=obj, gap=gap, sol_time=sol_time, termination=term)
        return row

    rng = np.random.default_rng([seed, 104729])
    if args.target == "policy":
        policy = _policy(model)
        policy.chooser = rng
    elif args.target == "expert":
        # set_scenarios adds the scenario index j < 1000 to this seed.
        learner.set_scenarios(EXPERT_SEED_OFFSET + 1000 * seed, asdict(config))
        chooser = _Chooser(rng, config.num_class)

    states, actions, rewards, feasibles = [], [], [], []
    solve_time = 0.0
    max_gap = 0.0
    state, done = sim.reset_simulator()
    while not done:
        if args.target == "policy":
            action, _ = policy.get_action(learner.get_feature_vector(state))
        elif args.target == "greedy":
            action = sim.select_physician(state)
        else:
            expert_actions, gap, _, sol_time = learner.get_expert_action(
                state, f"eval_{seed}"
            )
            action, _ = learner.sampled_vote(expert_actions, chooser)
            solve_time += sol_time or 0.0
            max_gap = max(max_gap, gap or 0.0)
        states.append(state)
        state, reward, feasible, done = sim.step(int(action))
        actions.append(int(action))
        rewards.append(reward)
        feasibles.append(feasible)
    states.append(state)
    stats = sim.collect_stats(states, actions, rewards, feasibles)
    keep = (
        "total_cost",
        "rejected_p1",
        "rejected_p2",
        "nonpref_assignments_total",
        "total_arrivals_p1",
        "total_arrivals_p2",
    )
    row.update({k: stats[k] for k in keep})
    row.update(sol_time=solve_time, gap=max_gap, wall=time.perf_counter() - start)
    return row


def parse() -> tuple[argparse.Namespace, Config]:
    """Split the command line into evaluation options and PPA options."""
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument(
        "--target",
        choices=("policy", "expert", "greedy", "hindsight"),
        required=True,
    )
    extra.add_argument("--split", choices=tuple(SPLIT_SEED), default="test")
    extra.add_argument("--n_episodes", type=int, default=1000)
    extra.add_argument("--workers", type=int, default=1)
    extra.add_argument("--policy_dir", default="")
    extra.add_argument("--models", type=int, nargs="*", default=[0])
    extra.add_argument("--out", required=True)
    args, rest = extra.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    config = build_config(parse_args())
    config.train = 0
    return args, config


def main() -> None:
    """Evaluate the requested target and write one row per episode."""
    args, config = parse()
    models = args.models if args.target == "policy" else [-1]
    tasks = [(m, n) for m in models for n in range(args.n_episodes)]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if args.workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers, _init, (config, args)) as pool:
            rows = pool.map(_episode, tasks, chunksize=max(1, args.n_episodes // 50))
    else:
        _init(config, args)
        rows = [_episode(t) for t in tasks]
    df = pd.DataFrame(rows)
    df["action_type"] = config.action_type
    df.to_csv(args.out, index=False)
    print(df.groupby("model")["total_cost"].mean().to_string())


if __name__ == "__main__":
    main()
