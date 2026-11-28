"""Job lines of the experiment grids (one command-line call per line).

Each line holds the options of ``python -m udagger_<app>.cli`` after the
global ``--workers`` and ``--out`` set by ``slurm/array.sh``. The functions
write the protocol shared by the examples: every expert learned with
one-shot cloning and DAgger, a MIP-gap study of one expert with DAgger, and
the experts used online on the test episodes.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

ExpertList = Sequence[tuple[str, int]]


def learn_lines(
    experts: ExpertList,
    seeds: Iterable[int],
    episodes: int,
    modes: Sequence[str] = ("oneshot", "dagger"),
    save_seed: int | None = 0,
    prefix: str = "",
    suffix: str = "",
) -> list[str]:
    """One learning run per expert, replicate and mode.

    Args:
        experts: ``(kind, scenarios)`` pairs.
        seeds: Replicates.
        episodes: Labelled training episodes.
        modes: Learning modes.
        save_seed: Replicate whose labels are saved (``--save-data``).
        prefix: Global options put before the sub-command.
        suffix: Options appended to every line (e.g. ``" --tag _ep1000"``).
    """
    lines = []
    for kind, k in experts:
        for seed in seeds:
            save = " --save-data" if seed == save_seed else ""
            base = f"{prefix}learn --expert {kind} --scenarios {k} --seed {seed}"
            lines += [
                f"{base} --mode {m} --episodes {episodes}{save}{suffix}" for m in modes
            ]
    return lines


def gap_lines(
    expert: tuple[str, int],
    gaps: Iterable[float],
    seeds: Iterable[int],
    episodes: int,
    prefix: str = "",
) -> list[str]:
    """DAgger runs of one expert at several MIP gaps."""
    kind, k = expert
    seeds = list(seeds)
    return [
        f"{prefix}learn --expert {kind} --scenarios {k} --seed {seed} --gap {gap} "
        f"--mode dagger --episodes {episodes}"
        for gap in gaps
        for seed in seeds
    ]


def expert_lines(
    experts: ExpertList,
    gap_expert: tuple[str, int] | None = None,
    gaps: Iterable[float] = (),
    prefix: str = "",
) -> list[str]:
    """Experts used online, then one expert at several MIP gaps."""
    lines = [f"{prefix}expert --expert {kind} --scenarios {k}" for kind, k in experts]
    if gap_expert is not None:
        kind, k = gap_expert
        lines += [
            f"{prefix}expert --expert {kind} --scenarios {k} --gap {g}" for g in gaps
        ]
    return lines


def vote_lines(scenarios: Sequence[int], prefix: str = "") -> list[str]:
    """Aggregated experts that execute the majority of their votes.

    One line per number of scenarios (``--tag _majority``), then the usual
    aggregated expert (sampled vote) with the largest number, for comparison.
    """
    experts = [("agg", k) for k in scenarios]
    lines = expert_lines(experts, prefix=prefix)
    lines = [f"{x} --no-sample-vote --tag _majority" for x in lines]
    return lines + expert_lines([("agg", max(scenarios))], prefix=prefix)


def write_grids(folder: Path, **grids: list[str]) -> None:
    """Write ``<name>_grid.txt`` for each grid and print their sizes."""
    for name, lines in grids.items():
        (folder / f"{name}_grid.txt").write_text("\n".join(lines) + "\n")
        print(f"{name}_grid.txt: {len(lines)} lines")
