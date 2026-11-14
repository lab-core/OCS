"""Adapters of the PPA simulator and experts to the ``udagger`` interfaces."""

from __future__ import annotations

import itertools
import os
import sys
import tempfile
from dataclasses import asdict
from typing import Any

import numpy as np
from udagger import Label, aggregate_votes, one_hot
from Learn_Module import IO_learning
from OCS import AppointmentScheduler
from utility import Config, build_config, parse_args

# Expert scenario seeds of an episode: offset + 1000 * episode seed + j, with
# j < 1000 the scenario index (disjoint from the episode seeds themselves).
EXPERT_SEED_OFFSET = 10_000_000
_EPISODE_IDS = itertools.count()


def config_from_args(argv: list[str]) -> Config:
    """Build a PPA configuration from options of the original scripts.

    Args:
        argv: Options of ``utility.parse_args`` (e.g. ``["--stoch", "1"]``).

    Returns:
        The configuration of ``utility.build_config``.
    """
    saved = sys.argv
    try:
        sys.argv = [saved[0], *argv]
        return build_config(parse_args())
    finally:
        sys.argv = saved


def expert_name(config: Config) -> str:
    """Name of the expert defined by a configuration."""
    if config.stoch == 1:
        return f"saa{config.scenarios}"
    if config.new == 0:
        return "full"
    return "det" if config.scenarios == 1 else f"agg{config.scenarios}"


class _Context:
    """Simulator and learner of one process (built lazily, not pickled)."""

    def __init__(self, config: Config) -> None:
        """Build the simulator exactly as in ``Learn_Main.py``."""
        p = config.ocs_param
        self.tmp = tempfile.mkdtemp(prefix="ppa_")
        self.sim = AppointmentScheduler(
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
            out_dir=self.tmp,
            suffix=False,
        )
        self.sim.setup_solver()
        self.learner = IO_learning(self.sim, asdict(config), folder=self.tmp)


class PPAEpisode:
    """One PPA episode (``udagger.Episode`` interface).

    Actions ``0..P-1`` assign the patient to a physician and ``P`` rejects
    the request. The objective (``total_cost``) is minimised.
    """

    def __init__(self, ctx: _Context, seed: int) -> None:
        """Generate the instance of ``seed`` (as ``Learn_Sim.py``)."""
        self.ctx = ctx
        self.seed = seed
        self.uid = next(_EPISODE_IDS)
        ctx.sim.set_seed(seed)
        ctx.sim.generate_instance()
        self.state, self._done = ctx.sim.reset_simulator()
        self.states: list[dict] = []
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.feasibles: list[bool] = []

    @property
    def done(self) -> bool:
        """Whether the booking horizon is over."""
        return self._done

    def observe(self) -> dict:
        """Return the simulator state."""
        return self.state

    def features(self, state: dict) -> np.ndarray:
        """Features of the original study (``IO_learning.get_feature_vector``)."""
        return np.asarray(self.ctx.learner.get_feature_vector(state), np.float32)

    def mask(self, state: dict) -> np.ndarray:
        """Feasible actions; the last feature block flags blocked actions."""
        n = self.ctx.sim.P + 1
        blocked = self.features(state)[-n:]
        return blocked == 0

    def step(self, action: int) -> None:
        """Assign or reject the current patient."""
        self.states.append(self.state)
        self.state, reward, feasible, self._done = self.ctx.sim.step(int(action))
        self.actions.append(int(action))
        self.rewards.append(reward)
        self.feasibles.append(feasible)

    def info(self, state: dict) -> dict[str, float]:
        """Arrival score and priority of the patient, stored with the labels."""
        cur = state["current_patient"]
        return {"epoch_time": float(cur["score"]), "priority": int(cur["priority"])}

    def stats(self) -> dict[str, float]:
        """Total cost (the objective) and rejection statistics."""
        st = self.ctx.sim.collect_stats(
            [*self.states, self.state], self.actions, self.rewards, self.feasibles
        )
        keys = ("rejected_p1", "rejected_p2", "nonpref_assignments_total")
        return {"objective": float(st["total_cost"]), **{k: st[k] for k in keys}}


class PPAProblem:
    """Factory of PPA episodes (``udagger.Problem`` interface).

    The simulator is built lazily in every process, so that the problem can
    be sent to worker processes; a process runs one episode at a time.

    Args:
        config: Configuration of the original scripts.
    """

    maximize = False

    def __init__(self, config: Config) -> None:
        """Store the configuration."""
        self.config = config
        self.n_actions = config.num_class
        self._ctx: _Context | None = None

    def __getstate__(self) -> dict[str, Any]:
        """Pickle the configuration only."""
        return {"config": self.config, "n_actions": self.n_actions, "_ctx": None}

    @property
    def ctx(self) -> _Context:
        """Simulator and learner of the current process."""
        if self._ctx is None:
            self._ctx = _Context(self.config)
        return self._ctx

    def episode(self, seed: int) -> PPAEpisode:
        """Create the episode of a seed."""
        return PPAEpisode(self.ctx, seed)


class PPAExpert:
    """Optimisation expert of the original study (``udagger.Expert``).

    The kind of expert is set by the configuration: ``stoch = 1`` is the
    two-stage expert, ``new = 0`` the full-information expert, and otherwise
    one (deterministic) or several (aggregated) sampled scenarios. As in the
    original scripts, the executed action is drawn from the votes.

    Args:
        config: Configuration of the expert (``--stoch``, ``--new``,
            ``--scenarios``, ``--mipgap``, ``--time_limit``).
    """

    def __init__(self, config: Config) -> None:
        """Store the configuration."""
        self.config = config
        self.name = expert_name(config)
        self._episode: tuple[int, int] | None = None

    def __call__(
        self, episode: PPAEpisode, state: dict, rng: np.random.Generator
    ) -> Label:
        """Solve the expert models at a state of an episode."""
        learner = episode.ctx.learner
        key = (os.getpid(), episode.uid)
        if self._episode != key:
            # New episode: scenario models seeded from the episode seed.
            learner.new = self.config.new
            learner.stoch = self.config.stoch
            learner.num_scenarios = self.config.scenarios
            learner.set_scenarios(
                EXPERT_SEED_OFFSET + 1000 * episode.seed, asdict(self.config)
            )
            self._episode = key
        votes, gap, _, sol_time = learner.get_expert_action(state, self.name)
        action, dist = aggregate_votes(
            [int(v) for v in votes], self.config.num_class, rng, sample=True
        )
        return Label(action, dist, float(sol_time or 0.0), float(gap or 0.0) / 100)


class PPAGreedy:
    """Myopic rule: preferred physician if possible, else a random eligible one."""

    name = "greedy"

    def __call__(
        self, episode: PPAEpisode, state: dict, rng: np.random.Generator
    ) -> Label:
        """Return the greedy action (drawn by the simulator, as originally)."""
        action = int(episode.ctx.sim.select_physician(state))
        return Label(action, one_hot(action, episode.ctx.sim.P + 1))
