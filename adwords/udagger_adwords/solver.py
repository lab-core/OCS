"""Type-count mixed-integer programs used by the optimisation-based experts.

In AdWords the offline problem does not depend on the arrival order: only the
number of queries of each type matters. Given the remaining budgets, the
current query and a set of scenarios of future type counts, the two-stage
model is::

    max  sum_i b_iq z_i + 1/S sum_s sum_(i,j) b_ij y_sij
    s.t. sum_i z_i <= 1
         b_iq z_i + sum_j b_ij y_sij <= R_i          for all s, i
         sum_i y_sij <= n_sj                         for all s, j
         z binary (fixed to 0 when infeasible), y integer >= 0

where ``z`` is the here-and-now decision for the current query and ``y_s``
are recourse assignments of future queries in scenario ``s``. A single
scenario gives the deterministic and full-information models; ``query=None``
gives the offline (hindsight) problem of a whole episode.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import gurobipy as gp
import numpy as np
import scipy.sparse as sp
from gurobipy import GRB

from udagger_adwords.instance import Market

_ENV: gp.Env | None = None
# Bonus on the revenue of the current query: among equally good solutions,
# serve the query now. It only decides ties when the MIP is solved to
# optimality (or to a gap below 0.1%); at the study's 2% gap Gurobi may return
# any incumbent within the gap, so ties among near-optimal solutions remain.
TIE_BREAK = 1e-3
LICENSE_RETRIES = 10


def get_env() -> gp.Env:
    """Return a process-wide silent single-threaded Gurobi environment.

    Starting the environment is retried with a growing delay, because a
    shared cluster licence may temporarily refuse new tokens.
    """
    global _ENV
    if _ENV is None:
        for attempt in range(LICENSE_RETRIES):
            env = gp.Env(empty=True)
            env.setParam("OutputFlag", 0)
            env.setParam("Threads", 1)
            try:
                env.start()
                break
            except gp.GurobiError:
                # A shared cluster licence can be saturated temporarily.
                env.dispose()
                if attempt == LICENSE_RETRIES - 1:
                    raise
                time.sleep(30.0 * (attempt + 1))
        _ENV = env
    return _ENV


@dataclass
class SolveResult:
    """Result of a count MIP.

    Attributes:
        action: Selected action (0 = skip, ``i + 1`` = advertiser ``i``), or
            -1 when no current query was given.
        objective: Objective value of the incumbent.
        bound: Best bound reported by the solver.
        runtime: Wall-clock seconds spent building and solving the model.
    """

    action: int
    objective: float
    bound: float
    runtime: float


class _Structure:
    """Sparse matrices describing the (advertiser, type) bid pairs."""

    def __init__(self, market: Market) -> None:
        """Pre-compute the bid pairs and the constraint matrices."""
        adv, typ = np.nonzero(market.bids)
        self.adv = adv
        self.typ = typ
        self.bid = market.bids[adv, typ].astype(float)
        n_pairs = adv.size
        cols = np.arange(n_pairs)
        self.budget_mat = sp.csr_matrix(
            (self.bid, (adv, cols)), shape=(market.n_adv, n_pairs)
        )
        self.count_mat = sp.csr_matrix(
            (np.ones(n_pairs), (typ, cols)), shape=(market.n_types, n_pairs)
        )


_STRUCTURES: dict[bytes, _Structure] = {}


def _structure(market: Market) -> _Structure:
    # Markets reach workers by pickling, so object identity cannot be the key:
    # the model structure only depends on the bids and the query classes.
    key = market.bids.tobytes() + market.type_class.tobytes()
    if key not in _STRUCTURES:
        _STRUCTURES[key] = _Structure(market)
    return _STRUCTURES[key]


def solve_count_mip(
    market: Market,
    budgets: np.ndarray,
    query: int | None,
    scenarios: np.ndarray,
    mip_gap: float = 0.0,
    time_limit: float = 10.0,
) -> SolveResult:
    """Solve the (two-stage) type-count MIP.

    Args:
        market: The market.
        budgets: Remaining budget of each advertiser.
        query: Type of the current query, or None for the offline problem.
        scenarios: Future type counts, shape (S, n_types).
        mip_gap: Relative MIP gap stopping criterion.
        time_limit: Time limit in seconds.

    Returns:
        The solve result; the action is the first-stage decision.
    """
    start = time.perf_counter()
    st = _structure(market)
    scenarios = np.atleast_2d(scenarios)
    n_scen = scenarios.shape[0]
    model = gp.Model(env=get_env())
    model.Params.MIPGap = mip_gap
    model.Params.TimeLimit = time_limit

    y = model.addMVar((n_scen, st.bid.size), vtype=GRB.INTEGER, lb=0.0)
    objective = (y @ st.bid).sum() / n_scen
    z = None
    q_bid = np.zeros(market.n_adv)
    if query is not None:
        q_bid = market.bids[:, query].astype(float)
        ub = ((q_bid > 0) & (q_bid <= budgets)).astype(float)
        z = model.addMVar(market.n_adv, vtype=GRB.BINARY, ub=ub)
        model.addConstr(z.sum() <= 1)
        # A small bonus on immediate revenue breaks the many ties between
        # serving the current query now and serving a future one instead.
        objective = objective + (1.0 + TIE_BREAK) * q_bid @ z
    model.setObjective(objective, GRB.MAXIMIZE)
    rhs = budgets.astype(float)
    for s in range(n_scen):
        if z is not None:
            model.addConstr(st.budget_mat @ y[s] + np.diag(q_bid) @ z <= rhs)
        else:
            model.addConstr(st.budget_mat @ y[s] <= rhs)
        model.addConstr(st.count_mat @ y[s] <= scenarios[s].astype(float))
    model.optimize()

    action = -1
    if model.SolCount == 0:
        # No incumbent within the time limit: skip the query (always feasible).
        result = SolveResult(
            action=0 if z is not None else -1,
            objective=0.0,
            bound=float(model.ObjBound),
            runtime=time.perf_counter() - start,
        )
        model.dispose()
        return result
    if z is not None:
        zval = z.X
        action = int(np.argmax(zval)) + 1 if zval.max() > 0.5 else 0
    result = SolveResult(
        action=action,
        objective=float(model.ObjVal),
        bound=float(model.ObjBound),
        runtime=time.perf_counter() - start,
    )
    model.dispose()
    return result


def offline_optimum(market: Market, type_counts: np.ndarray) -> float:
    """Return the hindsight optimal revenue of an episode.

    Args:
        market: The market.
        type_counts: Number of queries of each type in the episode.

    Returns:
        The optimal offline revenue.
    """
    res = solve_count_mip(
        market, market.budgets, None, type_counts[None, :], mip_gap=0.0, time_limit=60.0
    )
    return res.objective
