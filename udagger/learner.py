"""Imitation learning loops: one-shot behavioural cloning and DAgger.

Both modes label states with an expert and train the same policy:

* one-shot: the expert controls every training episode (``beta = 1``);
  optionally, policies are also trained on the first ``sizes`` episodes;
* DAgger: episodes are collected in iterations of ``per_iteration``
  episodes; at iteration ``i`` the expert acts with probability
  ``beta_decay ** i`` (1 at the first iteration) and the learned policy
  otherwise, every visited state is labelled, and the policy is retrained on
  the aggregated data every ``retrain_every`` iterations.

Every trained policy is scored on the validation episodes and the best one
is returned.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from udagger import checkpoint as ckpt
from udagger.policy import NumpyPolicy, train_policy
from udagger.problem import Expert
from udagger.rollout import EpisodeResult, Runner, merge, stack


@dataclass
class LearnConfig:
    """Settings of a learning run.

    Attributes:
        mode: ``"oneshot"`` or ``"dagger"``.
        seed: Seed of the network training (replicate index).
        sizes: One-shot only: also train on these first numbers of episodes.
        per_iteration: DAgger: episodes per iteration.
        retrain_every: DAgger: iterations between two retrainings.
        beta_decay: DAgger: decay of the probability that the expert acts.
        warm_start: DAgger: start each training from the previous weights.
        policy_action: ``"sample"`` to draw actions from the learned
            distribution, ``"mode"`` to take the most likely action.
        epochs: Training epochs.
        batch_size: Mini-batch size.
        patience: Early-stopping patience (None: all epochs).
        val_fraction: Fraction of the samples held out to select the epoch.
        hidden: Width of the hidden layers.
        depth: Number of hidden layers.
        lr: Adam learning rate.
    """

    mode: Literal["oneshot", "dagger"] = "dagger"
    seed: int = 0
    sizes: list[int] = field(default_factory=list)
    per_iteration: int = 10
    retrain_every: int = 1
    beta_decay: float = 0.8
    warm_start: bool = True
    policy_action: Literal["sample", "mode"] = "sample"
    epochs: int = 100
    batch_size: int = 256
    patience: int | None = None
    val_fraction: float = 0.2
    hidden: int = 128
    depth: int = 2
    lr: float = 1e-3


@dataclass
class LearnResult:
    """Outcome of a learning run.

    Attributes:
        policy: Best policy on validation.
        best: Learning-curve row of the best policy.
        curve: One row per trained policy.
        data: All labelled samples.
        label_seconds: Total expert labelling time.
        wall_seconds: Wall time of the learning, including the time spent
            before a resume.
    """

    policy: NumpyPolicy
    best: dict
    curve: list[dict]
    data: dict[str, np.ndarray]
    label_seconds: float
    wall_seconds: float = 0.0


def mean_objective(results: list[EpisodeResult]) -> float:
    """Average objective of a list of episodes."""
    return float(np.mean([r.objective for r in results]))


def _train(
    cfg: LearnConfig,
    data: dict[str, np.ndarray],
    seed: int,
    init: NumpyPolicy | None = None,
) -> tuple[NumpyPolicy, dict]:
    if not data or data["x"].shape[0] == 0:
        raise ValueError("no state was labelled by the expert")
    policy, stats = train_policy(
        data["x"],
        data["target"],
        data["mask"],
        seed=seed,
        hidden=cfg.hidden,
        depth=cfg.depth,
        lr=cfg.lr,
        batch_size=cfg.batch_size,
        max_epochs=cfg.epochs,
        patience=cfg.patience,
        val_fraction=cfg.val_fraction,
        init=init,
    )
    return policy.with_sampling(cfg.policy_action == "sample"), asdict(stats)


def learn(
    runner: Runner,
    expert: Expert,
    cfg: LearnConfig,
    train_seeds: list[int],
    valid_seeds: list[int],
    log: Callable[[str], None] = print,
    checkpoint_dir: Path | None = None,
    checkpoint_every: int = 10,
    context: dict | None = None,
) -> LearnResult:
    """Learn a policy from an expert.

    Args:
        runner: Runner of the problem.
        expert: Expert providing the labels.
        cfg: Settings of the run.
        train_seeds: Seeds of the training episodes (their number is the
            labelling budget).
        valid_seeds: Seeds of the validation episodes.
        log: Called with one JSON line per trained policy.
        checkpoint_dir: Folder of the checkpoint (None: no checkpoint). An
            existing checkpoint of the same run is resumed.
        checkpoint_every: DAgger iterations (one-shot: blocks of
            ``per_iteration`` labelled episodes) between two checkpoints.
        context: Other settings of the run (e.g. the instance options); a
            checkpoint is only resumed by a run with the same context.

    Returns:
        The best policy on validation and the learning history.

    Raises:
        ValueError: If the DAgger budget is not a multiple of
            ``per_iteration`` or a checkpoint belongs to another run.
    """
    checkpoint_every = max(1, checkpoint_every)
    sign = 1.0 if runner.problem.maximize else -1.0
    start = time.perf_counter()
    elapsed = 0.0  # wall time of the run before a resume
    label_seconds = 0.0
    curve: list[dict] = []
    data: dict[str, np.ndarray] = {}
    best: dict = {"score": -np.inf}
    fingerprint = {
        "config": asdict(cfg),
        "expert": getattr(expert, "name", type(expert).__name__),
        "train_seeds": [len(train_seeds), *train_seeds[:1], *train_seeds[-1:]],
        "valid_seeds": [len(valid_seeds), *valid_seeds[:1], *valid_seeds[-1:]],
        "context": context or {},
    }
    saved = ckpt.load(checkpoint_dir, fingerprint) if checkpoint_dir else None
    if saved is not None:
        st = saved["state"]
        elapsed, label_seconds, curve = st["elapsed"], st["label_seconds"], st["curve"]
        data = saved["data"]
        if saved["best"] is not None:
            best = {"score": st["best_score"], "policy": saved["best"]}
            best["row"] = st["best_row"]
        log(json.dumps({"resumed": checkpoint_dir.as_posix(), "next": st["next"]}))

    def wall() -> float:
        return elapsed + time.perf_counter() - start

    def save(next_step: int, policy: NumpyPolicy | None) -> None:
        if checkpoint_dir is None:
            return
        state = {
            "next": next_step,
            "elapsed": wall(),
            "label_seconds": label_seconds,
            "curve": curve,
            "best_score": best["score"] if "policy" in best else None,
            "best_row": best.get("row"),
        }
        ckpt.save(checkpoint_dir, fingerprint, state, data, policy, best.get("policy"))

    def checkpoint(policy: NumpyPolicy, it: int, episodes: int, stats: dict) -> None:
        nonlocal best
        val = mean_objective(runner.run(valid_seeds, policy=policy))
        row = {
            "iteration": it,
            "episodes": episodes,
            "samples": int(data["x"].shape[0]),
            "label_seconds": label_seconds,
            "wall_seconds": wall(),
            "val_objective": val,
            **stats,
        }
        curve.append(row)
        log(json.dumps(row))
        if sign * val > best["score"]:
            best = {"score": sign * val, "policy": policy, "row": row}

    n_episodes = len(train_seeds)
    if n_episodes == 0 or not valid_seeds:
        raise ValueError("learn needs training and validation episodes")
    if cfg.mode == "oneshot":
        # Label in blocks (one checkpoint per block); the episodes are
        # independent, so the data equal those of a single call.
        block = max(1, cfg.per_iteration * checkpoint_every)
        done = saved["state"]["next"] if saved is not None else 0
        while done < n_episodes:
            seeds = train_seeds[done : done + block]
            results = runner.run(seeds, expert=expert, collect=True)
            label_seconds += sum(r.solve_time for r in results)
            data = merge(data, stack(results))
            done += len(seeds)
            save(done, None)
        data_all = data
        if not data_all:
            raise ValueError("no state was labelled by the expert")
        sizes = {int(s) for s in cfg.sizes if 0 < s <= n_episodes}
        sizes = sorted(sizes | {n_episodes})
        for size in sizes:
            keep = np.isin(data_all["episode"], train_seeds[:size])
            data = {k: v[keep] for k, v in data_all.items()}
            policy, stats = _train(cfg, data, cfg.seed)
            checkpoint(policy, 0, size, stats)
        data = data_all
    else:
        per_it = cfg.per_iteration
        if n_episodes % per_it:
            raise ValueError(
                "the number of episodes must be a multiple of per_iteration"
            )
        n_it = n_episodes // per_it
        policy = saved["policy"] if saved is not None else None
        first = saved["state"]["next"] if saved is not None else 0
        for it in range(first, n_it):
            beta = cfg.beta_decay**it
            seeds = train_seeds[it * per_it : (it + 1) * per_it]
            results = runner.run(
                seeds, expert=expert, policy=policy, beta=beta, collect=True
            )
            label_seconds += sum(r.solve_time for r in results)
            data = merge(data, stack(results))
            last = it == n_it - 1
            if (it + 1) % cfg.retrain_every == 0 or it == 0 or last:
                init = policy if cfg.warm_start else None
                train_seed = cfg.seed * 100_000 + it
                policy, stats = _train(cfg, data, train_seed, init)
                checkpoint(policy, it + 1, (it + 1) * per_it, stats)
            if (it + 1) % checkpoint_every == 0 and not last:
                save(it + 1, policy)

    if "policy" not in best:
        raise ValueError(
            "no policy was selected: no state was labelled or every validation "
            "objective is NaN"
        )
    return LearnResult(
        policy=best["policy"],
        best=best["row"],
        curve=curve,
        data=data,
        label_seconds=label_seconds,
        wall_seconds=wall(),
    )
