"""Simulator of the physician-to-patient assignment.

``AppointmentScheduler`` (from the original study, ``_scheduler.py``) holds
both the episode simulator (``reset_simulator``, ``step``,
``collect_stats``) and the deterministic assignment MIP (``build_model``,
``solve``) used by the full-information, deterministic and aggregated
experts.
"""

from __future__ import annotations

from udagger_ppa._scheduler import AppointmentScheduler, Patient
from udagger_ppa.instance import Instance, model_kwargs

__all__ = ["AppointmentScheduler", "Patient", "make_scheduler"]


def make_scheduler(
    inst: Instance, seed: int, time_limit: float = 30.0, mipgap: float = 0.02
) -> AppointmentScheduler:
    """Simulator and deterministic model, built as in the original scripts."""
    sched = AppointmentScheduler(**model_kwargs(inst, seed, time_limit, mipgap))
    sched.setup_solver()
    return sched
