"""State features of the PPA policy (from the original ``IO_learning``)."""

from __future__ import annotations

import numpy as np

from udagger_ppa.scheduler import AppointmentScheduler

_EPS = 1e-9


def feature_names(n_phys: int) -> list[str]:
    """Names of the features, in the order of :func:`features`."""
    names = ["arrival_score", "remaining_release", "priority_is_p1", "duration"]
    for p in range(n_phys):
        names += [
            f"phys{p}_is_preferred",
            f"phys{p}_cap_remaining_norm",
            f"phys{p}_acc_p1_norm",
            f"phys{p}_load_pressure",
        ]
    names += [f"mask_phys{p}" for p in range(n_phys)] + ["mask_reject"]
    return names


def blocked(state: dict, n_phys: int) -> list[int]:
    """1 for each physician the patient cannot be assigned to, else 0."""
    cur = state["current_patient"]
    available = list(cur.get("available_phys") or [1] * n_phys)
    eligible = list(cur.get("eligible_phys") or [1] * n_phys)
    if len(available) != len(eligible):
        raise ValueError("availability and eligibility lengths differ")
    return [0 if a * e else 1 for a, e in zip(available, eligible, strict=True)]


def features(state: dict, sim: AppointmentScheduler) -> np.ndarray:
    """Encode a decision state.

    Args:
        state: Simulator state (``AppointmentScheduler.step`` output).
        sim: Simulator of the episode (capacities and session length).

    Returns:
        Float32 vector: arrival score, remaining release capacity, priority,
        duration (hours); per physician: preferred, remaining capacity,
        accepted priority-1 patients, workload; then the blocked-action flags
        (rejection is never blocked).

    Raises:
        ValueError: If the state is terminal.
    """
    if state.get("done", False) or state.get("current_patient") is None:
        raise ValueError("no current patient to encode")
    cur = state["current_patient"]
    n_phys = state["P"]
    release = state.get("remaining_release")
    rmax = state.get("Rmax", sim.Rmax)
    if release is not None and rmax is not None and float(rmax) > 0:
        release_norm = float(release) / max(_EPS, float(rmax))
    else:
        release_norm = 0.0
    vec = [
        cur["score"],
        release_norm,
        1 if int(cur.get("priority", 2)) == 1 else 0,
        cur.get("duration") / 60,
    ]

    cap = state.get("cap_remaining")
    if cap is None:
        assigned = state.get("assigned_per_phys", [0] * n_phys)
        cap = [max(0, int(sim.cap_per_phys) - int(a)) for a in assigned]
    if sim.cap_per_phys is not None and sim.cap_per_phys > 0:
        cap_norm = [c / max(_EPS, float(sim.cap_per_phys)) for c in cap]
    else:
        cap_norm = [1.0] * n_phys
    work = [float(w) for w in state.get("work_per_phys", [0.0] * n_phys)]
    accepted = state["accepted_priority_phy"]
    preferred = int(cur.get("preferred_phys", 0))
    for p in range(n_phys):
        vec += [
            1 if p == preferred else 0,
            cap_norm[p],
            accepted[p][1],
            work[p] / max(_EPS, sim.session_time),
        ]
    return np.asarray(vec + blocked(state, n_phys) + [0], dtype=np.float32)
