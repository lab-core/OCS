"""Write the AdWords experiment grids for ``slurm/array.sh adwords GRID``.

Protocol of the physician study (defaults of the command line): 4,000
training episodes, DAgger with 10 episodes per iteration, MIPs stopped at a
2% gap or 30 s, and a two-stage baseline with 30 scenarios at a 0.2% gap.
The uncertainty study only changes the spread of the number of premium
queries (10 in the main study); its levels were fixed before any result.
"""

from __future__ import annotations

from pathlib import Path

from udagger.grid import expert_lines, gap_lines, learn_lines, write_grids

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
EPISODES = 4000
GAP_EXPERT = ("saa", 30)
EXTRA_GAPS = (0.01, 0.05, 0.1)
BASELINE_GAP = 0.002
PREMIUM_STDS = (5, 15)
STD_EXPERTS = [("greedy", 1), ("full", 1), ("det", 1), ("agg", 5), ("saa", 5)]


def main() -> None:
    """Write the main, expert and uncertainty grids next to this script."""
    learn = learn_lines(EXPERTS, SEEDS, EPISODES)
    learn += gap_lines(GAP_EXPERT, EXTRA_GAPS, SEEDS, EPISODES)
    experts = ["hindsight"]
    experts += expert_lines(EXPERTS, GAP_EXPERT, (BASELINE_GAP, *EXTRA_GAPS))
    std = []
    for level in PREMIUM_STDS:
        # Global options first: each level writes to its own directory.
        prefix = f"--premium-std {level} --out-suffix _std{level} "
        std.append(f"{prefix}hindsight")
        std += expert_lines(STD_EXPERTS, prefix=prefix)
        std += learn_lines(
            STD_EXPERTS, SEEDS, EPISODES, ("oneshot",), save_seed=None, prefix=prefix
        )
    write_grids(Path(__file__).parent, learn=learn, expert=experts, std=std)


if __name__ == "__main__":
    main()
