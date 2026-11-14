"""Write the job lines of the PPA experiments for ``slurm/array.sh``.

Same protocol and layout as ``adwords/slurm/make_grid.py``: every expert
learned with one-shot cloning and DAgger (4,000 episodes, 3 replicates),
the MIP-gap study of the two-stage expert with 30 scenarios, and the experts
used online on the test episodes (with the two-stage baseline at a 0.2% gap).
Run from the repository root::

    python ppa/slurm/make_grid.py
    sbatch --array=1-14 slurm/array.sh ppa ppa/slurm/expert_grid.txt
    sbatch --array=1-57 --time=7-00:00:00 slurm/array.sh ppa ppa/slurm/learn_grid.txt
"""

from __future__ import annotations

from pathlib import Path

EXPERTS = [
    ("full", 1),
    ("det", 1),
    ("agg", 5),
    ("agg", 15),
    ("agg", 30),
    ("saa", 5),
    ("saa", 15),
    ("saa", 30),
]
SEEDS = range(3)
EXTRA_GAPS = (0.01, 0.05, 0.1)
BASELINE_GAP = 0.002
EPISODES = 4000


def main() -> None:
    """Write ``learn_grid.txt`` and ``expert_grid.txt`` next to this script."""
    learn = []
    for kind, k in EXPERTS:
        for seed in SEEDS:
            save = " --save-data" if seed == 0 else ""
            base = f"learn --expert {kind} --scenarios {k} --seed {seed}"
            learn.append(f"{base} --mode oneshot --episodes {EPISODES}{save}")
            learn.append(f"{base} --mode dagger --episodes {EPISODES}{save}")
    for gap in EXTRA_GAPS:
        for seed in SEEDS:
            learn.append(
                f"learn --expert saa --scenarios 30 --seed {seed} --gap {gap} "
                f"--mode dagger --episodes {EPISODES}"
            )
    experts = ["hindsight", "expert --expert greedy"]
    experts += [f"expert --expert {kind} --scenarios {k}" for kind, k in EXPERTS]
    experts += [
        f"expert --expert saa --scenarios 30 --gap {g}"
        for g in (BASELINE_GAP, *EXTRA_GAPS)
    ]
    here = Path(__file__).parent
    (here / "learn_grid.txt").write_text("\n".join(learn) + "\n")
    (here / "expert_grid.txt").write_text("\n".join(experts) + "\n")
    print(len(learn), "learning runs,", len(experts), "expert evaluations")


if __name__ == "__main__":
    main()
