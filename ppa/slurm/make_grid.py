"""Write the job lines of the PPA rerun (protocol of the IJOO study).

Experts are configured as in ``utility.build_config``: ``--stoch 0`` solves
independent deterministic models (``--new 0``: realised future, i.e. the
full-information expert; ``--new 1``: sampled futures, deterministic for one
scenario and aggregated for several), and ``--stoch 1`` the two-stage model.
"""

from __future__ import annotations

from pathlib import Path

EXPERTS = {
    "full": "--stoch 0 --new 0 --scenarios 1",
    "det": "--stoch 0 --new 1 --scenarios 1",
    "agg5": "--stoch 0 --new 1 --scenarios 5",
    "agg15": "--stoch 0 --new 1 --scenarios 15",
    "agg30": "--stoch 0 --new 1 --scenarios 30",
    "saa5": "--stoch 1 --new 1 --scenarios 5",
    "saa15": "--stoch 1 --new 1 --scenarios 15",
    "saa30": "--stoch 1 --new 1 --scenarios 30",
}
REPS = range(3)
EXTRA_GAPS = (0.01, 0.05, 0.1)
BASELINE_GAP = 0.002


def line(**kw: object) -> str:
    """Format one grid line as shell assignments."""
    return " ".join(
        f'{k}="{v}"' if k == "expert" else f"{k}={v}" for k, v in kw.items()
    )


def main() -> None:
    """Write ``oneshot_grid.txt``, ``dagger_grid.txt`` and ``expert_grid.txt``."""
    here = Path(__file__).parent
    for mode in ("oneshot", "dagger"):
        rows = [
            line(kind="learn", mode=mode, expert=opts, rep=r)
            for opts in EXPERTS.values()
            for r in REPS
        ]
        if mode == "dagger":
            rows += [
                line(kind="learn", mode=mode, expert=EXPERTS["saa30"], mg=g, rep=r)
                for g in EXTRA_GAPS
                for r in REPS
            ]
        (here / f"{mode}_grid.txt").write_text("\n".join(rows) + "\n")
    rows = [line(kind="greedy"), line(kind="hindsight", mg=0, tl=300)]
    rows += [line(kind="expert", expert=o) for k, o in EXPERTS.items() if k != "full"]
    rows += [
        line(kind="expert", expert=EXPERTS["saa30"], mg=g)
        for g in (BASELINE_GAP, *EXTRA_GAPS)
    ]
    (here / "expert_grid.txt").write_text("\n".join(rows) + "\n")
    for f in ("oneshot", "dagger", "expert"):
        print(f, len((here / f"{f}_grid.txt").read_text().splitlines()))


if __name__ == "__main__":
    main()
