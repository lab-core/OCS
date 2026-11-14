"""Episode rollouts with expert labelling, run in parallel.

A rollout executes a (possibly mixed) policy on one episode. When an expert
is given, every visited state can be labelled by the expert, which is how
both one-shot behavioural cloning (``beta = 1``) and DAgger (``beta < 1``)
collect data. Rollouts are distributed over a process pool; the problem is
sent once to every worker.
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass, field

import numpy as np

from udagger.policy import NumpyPolicy
from udagger.problem import Expert, Problem

# Salts of the per-episode random streams: one for the beta-mixing and the
# expert, one for the learned policy, so that test episodes stay paired across
# policies whatever the expert does.
EXPERT_STREAM = 7919
POLICY_STREAM = 104729


@dataclass
class EpisodeResult:
    """Outcome of one rollout.

    Attributes:
        seed: Episode seed.
        stats: End-of-episode statistics of the problem (with ``objective``).
        solve_time: Total expert labelling time.
        max_gap: Largest optimality gap reported by the expert.
        n_labels: Number of labelled states.
        data: Labelled samples (``x``, ``target``, ``mask``, ``expert_action``
            and the keys of ``Episode.info``), or None.
    """

    seed: int
    stats: dict[str, float]
    solve_time: float = 0.0
    max_gap: float = 0.0
    n_labels: int = 0
    data: dict[str, np.ndarray] | None = field(default=None, repr=False)

    @property
    def objective(self) -> float:
        """Objective value of the episode."""
        return float(self.stats["objective"])


def run_episode(
    problem: Problem,
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
        problem: The problem.
        seed: Episode seed.
        expert: Expert used for labelling and/or acting.
        policy: Learned policy.
        beta: Probability of executing the expert action.
        collect: Whether to return labelled samples.

    Returns:
        The episode result.

    Raises:
        ValueError: If neither an expert nor a policy can act.
    """
    episode = problem.episode(seed)
    rng = np.random.default_rng([seed, EXPERT_STREAM])
    policy_rng = np.random.default_rng([seed, POLICY_STREAM])
    rows: dict[str, list] = {}
    solve_time = 0.0
    max_gap = 0.0
    while not episode.done:
        state = episode.observe()
        use_expert = expert is not None and (policy is None or rng.random() < beta)
        label = None
        if expert is not None and (collect or use_expert):
            label = expert(episode, state, rng)
            solve_time += label.solve_time
            max_gap = max(max_gap, label.gap)
        x = mask = None
        if collect and label is not None:
            x, mask = episode.features(state), episode.mask(state)
            sample = {
                "x": x,
                "target": label.dist,
                "mask": mask,
                "expert_action": label.action,
                **episode.info(state),
            }
            for key, value in sample.items():
                rows.setdefault(key, []).append(value)
        if use_expert and label is not None:
            action = label.action
        elif policy is not None:
            if x is None:
                x, mask = episode.features(state), episode.mask(state)
            action = policy.act(x, mask, policy_rng)
        else:
            raise ValueError("no policy to act")
        episode.step(action)

    result = EpisodeResult(
        seed=seed,
        stats=episode.stats(),
        solve_time=solve_time,
        max_gap=max_gap,
        n_labels=len(rows.get("x", [])),
    )
    if rows:
        result.data = {
            k: np.stack(v) if k in ("x", "target", "mask") else np.asarray(v)
            for k, v in rows.items()
        }
    return result


_WORKER: dict = {}


def _init_worker(problem: Problem) -> None:
    _WORKER["problem"] = problem


def _task(args: tuple) -> EpisodeResult:
    seed, expert, policy, beta, collect = args
    return run_episode(_WORKER["problem"], seed, expert, policy, beta, collect)


class Runner:
    """Process pool that runs episodes in parallel.

    Args:
        problem: The problem shared by all workers.
        workers: Number of processes (1 runs in the current process).
    """

    def __init__(self, problem: Problem, workers: int = 1) -> None:
        """Start the pool."""
        self.problem = problem
        self.workers = workers
        self.pool = None
        if workers > 1:
            ctx = mp.get_context("spawn")
            self.pool = ctx.Pool(workers, initializer=_init_worker, initargs=(problem,))
        else:
            _init_worker(problem)

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


def stack(results: list[EpisodeResult]) -> dict[str, np.ndarray]:
    """Concatenate the labelled samples of several episodes.

    An ``episode`` array with the seed of each sample is added.
    """
    keep = [r for r in results if r.data is not None]
    if not keep:
        return {}
    data = {k: np.concatenate([r.data[k] for r in keep]) for k in keep[0].data}
    data["episode"] = np.concatenate(
        [np.full(r.n_labels, r.seed) for r in keep]
    ).astype(np.int64)
    return data


def merge(
    a: dict[str, np.ndarray] | None, b: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Append the samples of ``b`` to those of ``a``."""
    if not a:
        return b
    if not b:
        return a
    return {k: np.concatenate([a[k], b[k]]) for k in a}
