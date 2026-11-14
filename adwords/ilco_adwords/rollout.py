"""Episode rollouts, expert labelling and parallel evaluation.

A rollout executes a (possibly mixed) policy on one episode. When an expert is
given, every visited state is labelled by the expert, which is how both
one-shot behavioural cloning (``beta = 1``) and DAgger (``beta < 1``) collect
data. Rollouts are distributed over a process pool; every worker owns its
Gurobi environment.
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass, field

import numpy as np

from ilco_adwords.env import Simulator, feasible_mask
from ilco_adwords.experts import Expert
from ilco_adwords.market import PREMIUM, REGULAR, Market, episode_seed, sample_episode
from ilco_adwords.policy import NumpyPolicy, features


@dataclass
class EpisodeResult:
    """Outcome of one rollout.

    Attributes:
        seed: Episode seed.
        revenue: Total revenue collected.
        n_queries: Number of queries per class (regular, premium).
        served: Number of served queries per class.
        budget_used: Fraction of the total budget spent.
        solve_time: Total expert solve time.
        n_labels: Number of expert labels collected.
        max_gap: Largest MIP gap of the expert models.
        x: Features of labelled states.
        target: Expert target distributions.
        mask: Feasibility masks.
        epoch_time: Normalised time of each labelled state.
        expert_action: Expert action of each labelled state.
        query_class: Class of the query of each labelled state.
    """

    seed: int
    revenue: float
    n_queries: np.ndarray
    served: np.ndarray
    budget_used: float
    solve_time: float = 0.0
    n_labels: int = 0
    max_gap: float = 0.0
    x: np.ndarray | None = None
    target: np.ndarray | None = None
    mask: np.ndarray | None = None
    epoch_time: np.ndarray | None = None
    expert_action: np.ndarray | None = None
    query_class: np.ndarray | None = field(default=None)


def run_episode(
    market: Market,
    seed: int,
    expert: Expert | None = None,
    policy: NumpyPolicy | None = None,
    beta: float = 1.0,
    collect: bool = False,
) -> EpisodeResult:
    """Run one episode with a mixture of an expert and a learned policy.

    At each state the expert action is executed with probability ``beta``
    and the policy action otherwise. If ``collect`` is True, the expert is
    queried at every state and the labelled samples are returned.

    Args:
        market: The market.
        seed: Episode seed (defines the query sequence).
        expert: Expert used for labelling and/or acting.
        policy: Learned policy.
        beta: Probability of executing the expert action.
        collect: Whether to return labelled samples.

    Returns:
        The episode result.

    Raises:
        ValueError: If neither an expert nor a policy can act.
    """
    episode = sample_episode(market, np.random.default_rng(seed))
    rng = np.random.default_rng([seed, 7919])
    sim = Simulator(market, episode)
    xs, targets, masks, times, acts, classes = [], [], [], [], [], []
    solve_time = 0.0
    max_gap = 0.0
    while not sim.done:
        state = sim.state()
        use_expert = expert is not None and (policy is None or rng.random() < beta)
        label = None
        if expert is not None and (collect or use_expert):
            future = sim.true_future_counts() if expert.kind == "full" else None
            label = expert(market, state, rng, future)
            solve_time += label.solve_time
            max_gap = max(max_gap, label.gap)
        if collect and label is not None:
            xs.append(features(market, state))
            targets.append(label.dist)
            masks.append(feasible_mask(market, state))
            times.append(state.time)
            acts.append(label.action)
            classes.append(market.type_class[state.query])
        if use_expert and label is not None:
            action = label.action
        elif policy is not None:
            action = policy.act(market, state)
        else:
            raise ValueError("no policy to act")
        sim.step(action)

    n_queries = np.array(
        [
            np.sum(market.type_class[episode.types] == REGULAR),
            np.sum(market.type_class[episode.types] == PREMIUM),
        ]
    )
    result = EpisodeResult(
        seed=seed,
        revenue=float(sim.revenue),
        n_queries=n_queries,
        served=sim.served.copy(),
        budget_used=float(1.0 - sim.budgets.sum() / market.budgets.sum()),
        solve_time=solve_time,
        n_labels=len(xs),
        max_gap=max_gap,
    )
    if collect and xs:
        result.x = np.stack(xs)
        result.target = np.stack(targets)
        result.mask = np.stack(masks)
        result.epoch_time = np.asarray(times)
        result.expert_action = np.asarray(acts)
        result.query_class = np.asarray(classes)
    return result


_WORKER: dict = {}


def _init_worker(market: Market) -> None:
    _WORKER["market"] = market


def _task(args: tuple) -> EpisodeResult:
    seed, expert, policy, beta, collect = args
    return run_episode(_WORKER["market"], seed, expert, policy, beta, collect)


class Runner:
    """Process pool that runs episodes in parallel.

    Args:
        market: The market shared by all workers.
        workers: Number of processes (1 runs in the current process).
    """

    def __init__(self, market: Market, workers: int = 1) -> None:
        """Start the pool."""
        self.market = market
        self.workers = workers
        self.pool = None
        if workers > 1:
            ctx = mp.get_context("spawn")
            self.pool = ctx.Pool(workers, initializer=_init_worker, initargs=(market,))
        else:
            _init_worker(market)

    def run(
        self,
        seeds: list[int],
        expert: Expert | None = None,
        policy: NumpyPolicy | None = None,
        beta: float = 1.0,
        collect: bool = False,
    ) -> list[EpisodeResult]:
        """Run one episode per seed and return results in seed order."""
        tasks = [(s, expert, policy, beta, collect) for s in seeds]
        if self.pool is None:
            return [_task(t) for t in tasks]
        return self.pool.map(_task, tasks, chunksize=1)

    def close(self) -> None:
        """Terminate the pool."""
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None


def split_seeds(split: str, n: int, run_seed: int = 0, start: int = 0) -> list[int]:
    """Return ``n`` episode seeds of a split starting at index ``start``."""
    return [episode_seed(split, start + i, run_seed) for i in range(n)]
