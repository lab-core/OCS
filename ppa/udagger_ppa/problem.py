"""The physician-to-patient assignment problem as a ``udagger`` problem."""

from __future__ import annotations

import itertools

import numpy as np

from udagger_ppa.features import blocked, features
from udagger_ppa.instance import Instance, make_scheduler
from udagger_ppa.scheduler import AppointmentScheduler

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

    def observe(self) -> dict:
        """Return the simulator state."""
        return self.state

    def features(self, state: dict) -> np.ndarray:
        """Features of a state (computed once per state)."""
        if self._x is None or self._x[0] != id(state):
            self._x = (id(state), features(state, self.sim))
        return self._x[1]

    def mask(self, state: dict) -> np.ndarray:
        """Feasible actions (rejection is always feasible)."""
        return np.array([b == 0 for b in blocked(state, self.sim.P)] + [True])

    def step(self, action: int) -> None:
        """Assign or reject the calling patient."""
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


def offline_optimum(
    instance: Instance, seed: int, time_limit: float = 300.0
) -> tuple[float, float | None]:
    """Hindsight optimum of an episode (all requests known in advance).

    Returns:
        The optimal (or best found) cost and the final MIP gap in %.
    """
    sim = make_scheduler(instance, seed, time_limit=time_limit, mipgap=0.0)
    sim.set_seed(seed)
    sim.generate_instance()
    sim.build_model(sim.patients)
    _, _, gap, _ = sim.solve(tee=False)
    _, cost = sim.extract_solution(sim.patients)
    return float(cost), gap
