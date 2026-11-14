"""The physician-to-patient assignment problem as a ``udagger`` problem."""

from __future__ import annotations

import itertools

import numpy as np

from udagger_ppa.features import blocked, features
from udagger_ppa.instance import Instance
from udagger_ppa.simulator import AppointmentScheduler, make_scheduler

_EPISODE_IDS = itertools.count()


class PPAEpisode:
    """One booking horizon (``udagger.Episode`` interface).

    Actions ``0..P-1`` assign the calling patient to a physician and ``P``
    rejects the request. The objective (total cost) is minimised.

    Args:
        sim: Simulator of the current process.
        seed: Episode seed (as ``Learn_Sim.py``: ``set_seed(seed)``).
    """

    def __init__(self, sim: AppointmentScheduler, seed: int) -> None:
        """Generate the requests of the episode and reset the simulator."""
        self.sim = sim
        self.seed = seed
        self.uid = next(_EPISODE_IDS)
        # The simulator is shared by the episodes of a process: only the most
        # recent episode may use it.
        sim.udagger_episode = self.uid
        sim.set_seed(seed)
        sim.generate_instance()
        self.state, self._done = sim.reset_simulator()
        self._x: tuple[int, np.ndarray] | None = None
        self.states: list[dict] = []
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.feasibles: list[bool] = []

    @property
    def done(self) -> bool:
        """Whether the booking horizon is over."""
        return self._done

    def _check(self) -> None:
        if getattr(self.sim, "udagger_episode", None) != self.uid:
            raise RuntimeError("the simulator now runs a newer episode")

    def observe(self) -> dict:
        """Return the simulator state."""
        self._check()
        return self.state

    def features(self, state: dict) -> np.ndarray:
        """Features of a state (computed once per state)."""
        # Cached for the current decision (patient index ``t``).
        if self._x is None or self._x[0] != state["t"]:
            self._x = (state["t"], features(state, self.sim))
        return self._x[1]

    def mask(self, state: dict) -> np.ndarray:
        """Feasible actions (rejection is always feasible)."""
        return np.array([b == 0 for b in blocked(state, self.sim.P)] + [True])

    def step(self, action: int) -> None:
        """Assign or reject the calling patient."""
        self._check()
        self.states.append(self.state)
        out = self.sim.step(int(action))
        # The simulator returns (state, reward, done) when called on a
        # finished episode and (state, reward, feasible, done) otherwise.
        if len(out) == 3:
            self.state, reward, self._done = out
            feasible = True
        else:
            self.state, reward, feasible, self._done = out
        self.actions.append(int(action))
        self.rewards.append(reward)
        self.feasibles.append(feasible)

    def info(self, state: dict) -> dict[str, float]:
        """Arrival score and priority of the patient, stored with the labels."""
        cur = state["current_patient"]
        return {"epoch_time": float(cur["score"]), "priority": int(cur["priority"])}

    def stats(self) -> dict[str, float]:
        """Total cost (the objective), rejections and arrivals."""
        self._check()
        st = self.sim.collect_stats(
            [*self.states, self.state], self.actions, self.rewards, self.feasibles
        )
        keys = (
            "rejected_p1",
            "rejected_p2",
            "nonpref_assignments_total",
            "total_arrivals_p1",
            "total_arrivals_p2",
        )
        return {"objective": float(st["total_cost"]), **{k: st[k] for k in keys}}


class PPAProblem:
    """Factory of PPA episodes (``udagger.Problem`` interface).

    The simulator is built lazily in each process, so that the problem can
    be sent to worker processes; a process runs one episode at a time.

    Args:
        instance: Instance parameters.
    """

    maximize = False

    def __init__(self, instance: Instance) -> None:
        """Store the instance parameters."""
        self.instance = instance
        self.n_actions = instance.n_actions
        self._sim: AppointmentScheduler | None = None

    def __getstate__(self) -> dict:
        """Pickle the parameters only."""
        return {**self.__dict__, "_sim": None}

    def episode(self, seed: int) -> PPAEpisode:
        """Create the episode of a seed."""
        if self._sim is None:
            self._sim = make_scheduler(self.instance, seed)
        return PPAEpisode(self._sim, seed)
