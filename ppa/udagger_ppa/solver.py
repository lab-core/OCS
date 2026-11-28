"""Optimisation models of the physician-to-patient assignment.

The two-stage stochastic model ``SAppointmentScheduler`` comes from the
original study (``_stochastic.py``); the deterministic model is part of the
simulator class (``simulator.AppointmentScheduler``).
"""

from __future__ import annotations

from udagger_ppa._stochastic import SAppointmentScheduler
from udagger_ppa.instance import Instance, model_kwargs
from udagger_ppa.simulator import make_scheduler

__all__ = ["SAppointmentScheduler", "make_stochastic", "offline_optimum"]


def make_stochastic(
    inst: Instance,
    seed: int,
    n_scenarios: int,
    time_limit: float = 30.0,
    mipgap: float = 0.02,
) -> SAppointmentScheduler:
    """Two-stage stochastic model with ``n_scenarios`` scenarios."""
    sched = SAppointmentScheduler(
        **model_kwargs(inst, seed, time_limit, mipgap), num_scenarios=n_scenarios
    )
    sched.setup_solver()
    return sched


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
