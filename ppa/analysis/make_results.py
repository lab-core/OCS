"""Build the PPA tables and figures from the results of ``udagger_ppa.cli``.

Usage::

    python analysis/make_results.py RESULTS_DIR PAPER_DIR

Writes ``tables/ppa_{main,gap,tests}.tex`` and ``figs/ppa_curves.pdf`` into
``PAPER_DIR`` (same tables as the AdWords example; costs are compared through
their regret, the relative excess cost over the hindsight optimum).
"""

from __future__ import annotations

import sys
from pathlib import Path

from udagger.analysis import Study, standard_outputs

EXPERT_ORDER = [
    "greedy",
    "full",
    "det",
    "agg5",
    "agg15",
    "agg30",
    "saa5",
    "saa15",
    "saa30",
]
LABELS = {
    "greedy": r"Myopic greedy $\pi^m$",
    "full": r"Full-information $\pi^f$",
    "det": r"Deterministic $\pi^d$",
    "agg5": r"Aggregated $\pi^a_5$",
    "agg15": r"Aggregated $\pi^a_{15}$",
    "agg30": r"Aggregated $\pi^a_{30}$",
    "saa5": r"Two-stage $\pi^t_5$",
    "saa15": r"Two-stage $\pi^t_{15}$",
    "saa30": r"Two-stage $\pi^t_{30}$",
}
SHORT = {
    "greedy": "Greedy",
    "full": "Full-info",
    "det": "Determ.",
    "agg5": "Agg-5",
    "agg15": "Agg-15",
    "agg30": "Agg-30",
    "saa5": "2S-5",
    "saa15": "2S-15",
    "saa30": "2S-30",
}
COLORS = {
    "greedy": "#9e9e9e",
    "full": "#1f77b4",
    "det": "#d62728",
    "agg5": "#f2b6b6",
    "agg15": "#e88080",
    "agg30": "#c44e52",
    "saa5": "#b9e3b0",
    "saa15": "#5fb85a",
    "saa30": "#1b7a2e",
}

STUDY = Study(
    prefix="ppa",
    objective="cost",
    maximize=False,
    order=EXPERT_ORDER,
    labels=LABELS,
    short=SHORT,
    colors=COLORS,
    group_ends=("greedy", "det", "agg30"),
    comparisons=(
        (("full", "oneshot"), ("det", "oneshot")),
        (("det", "oneshot"), ("agg30", "oneshot")),
        (("agg30", "oneshot"), ("saa5", "oneshot")),
        (("saa5", "oneshot"), ("saa30", "oneshot")),
        (("full", "oneshot"), ("full", "dagger")),
        (("det", "oneshot"), ("det", "dagger")),
        (("agg30", "oneshot"), ("agg30", "dagger")),
        (("saa30", "oneshot"), ("saa30", "dagger")),
    ),
    curve_experts=("full", "det", "agg30", "saa5", "saa15", "saa30"),
)


def main() -> None:
    """Entry point."""
    standard_outputs(STUDY, Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == "__main__":
    main()
