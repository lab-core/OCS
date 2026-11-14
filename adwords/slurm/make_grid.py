"""Write the list of experiment commands (one per line) for the job array."""

from __future__ import annotations

from pathlib import Path

EXPERTS = [
    ("greedy", 1),
    ("msvv", 1),
    ("full", 1),
    ("det", 1),
    ("agg", 5),
    ("agg", 15),
    ("agg", 30),
    ("saa", 5),
    ("saa", 15),
    ("saa", 30),
]
SEEDS = range(5)
GAPS = (0.01, 0.02, 0.05, 0.1)
EPISODES = 2000


def main() -> None:
    """Write ``slurm/learn_grid.txt`` and ``slurm/expert_grid.txt``."""
    learn = []
    for kind, k in EXPERTS:
        for seed in SEEDS:
            save = " --save-data" if seed == 0 else ""
            base = f"learn --expert {kind} --scenarios {k} --seed {seed}"
            learn.append(
                f"{base} --mode oneshot --episodes {EPISODES} "
                f"--sizes 100 250 500 1000{save}"
            )
            learn.append(
                f"{base} --mode dagger --episodes {EPISODES} "
                f"--per-iteration 10 --retrain-every 10{save}"
            )
    for gap in GAPS:
        for seed in SEEDS:
            learn.append(
                f"learn --expert saa --scenarios 30 --seed {seed} --gap {gap} "
                f"--mode dagger --episodes {EPISODES} --per-iteration 10 "
                "--retrain-every 10"
            )
    experts = [f"expert --expert {kind} --scenarios {k}" for kind, k in EXPERTS]
    experts += [f"expert --expert saa --scenarios 30 --gap {g}" for g in GAPS]
    here = Path(__file__).parent
    (here / "learn_grid.txt").write_text("\n".join(learn) + "\n")
    (here / "expert_grid.txt").write_text("\n".join(experts) + "\n")
    print(len(learn), "learning runs,", len(experts), "expert evaluations")


if __name__ == "__main__":
    main()
