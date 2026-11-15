"""Write the PPA experiment grids for ``slurm/array.sh ppa GRID``.

Same protocol as ``adwords/slurm/make_grid.py`` (defaults of the command
line), with 3 replicates as in the rerun of the physician study.
"""

from __future__ import annotations

from pathlib import Path

from udagger.grid import expert_lines, gap_lines, learn_lines, write_grids

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
EPISODES = 4000
GAP_EXPERT = ("saa", 30)
EXTRA_GAPS = (0.01, 0.05, 0.1)
BASELINE_GAP = 0.002


def main() -> None:
    """Write the learning and expert grids next to this script."""
    learn = learn_lines(EXPERTS, SEEDS, EPISODES)
    learn += gap_lines(GAP_EXPERT, EXTRA_GAPS, SEEDS, EPISODES)
    experts = ["hindsight"] + expert_lines(
        [("greedy", 1), *EXPERTS], GAP_EXPERT, (BASELINE_GAP, *EXTRA_GAPS)
    )
    write_grids(Path(__file__).parent, learn=learn, expert=experts)


if __name__ == "__main__":
    main()
