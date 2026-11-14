"""Write the list of experiment commands (one per line) for the job arrays.

The settings follow the protocol of the physician-to-patient study: 4,000
training episodes, DAgger with 10 episodes per iteration and a retraining after
every iteration, expert MIPs stopped at a 2% gap (or 30 s), and a two-stage
baseline with 30 scenarios solved to a 0.2% gap. These are the defaults of
``udagger_adwords.cli``; only the varying arguments are written here.
"""

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
EXTRA_GAPS = (0.01, 0.05, 0.1)
BASELINE_GAP = 0.002
EPISODES = 4000
# Uncertainty study: only the spread of the number of premium queries changes
# (the main study uses 10). Levels fixed before seeing any result.
PREMIUM_STDS = (5, 15)
STD_EXPERTS = [("greedy", 1), ("full", 1), ("det", 1), ("agg", 5), ("saa", 5)]


def main() -> None:
    """Write the grids of the main, expert and uncertainty studies."""
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
    experts = [f"expert --expert {kind} --scenarios {k}" for kind, k in EXPERTS]
    experts += [
        f"expert --expert saa --scenarios 30 --gap {g}"
        for g in (BASELINE_GAP, *EXTRA_GAPS)
    ]
    # Each line starts with global options and writes to its own directory.
    std_lines = []
    for std in PREMIUM_STDS:
        opts = f"--premium-std {std} --out-suffix _std{std}"
        std_lines.append(f"{opts} hindsight")
        std_lines += [
            f"{opts} expert --expert {kind} --scenarios {k}" for kind, k in STD_EXPERTS
        ]
        for kind, k in STD_EXPERTS:
            for seed in SEEDS:
                std_lines.append(
                    f"{opts} learn --expert {kind} --scenarios {k} --seed {seed} "
                    f"--mode oneshot --episodes {EPISODES}"
                )
    here = Path(__file__).parent
    (here / "learn_grid.txt").write_text("\n".join(learn) + "\n")
    (here / "expert_grid.txt").write_text("\n".join(experts) + "\n")
    (here / "std_grid.txt").write_text("\n".join(std_lines) + "\n")
    print(len(learn), "learning runs,", len(experts), "expert evaluations,")
    print(len(std_lines), "uncertainty-study commands")


if __name__ == "__main__":
    main()
