"""Build the AdWords tables and figures of the CPAIOR paper from raw results.

Usage::

    python analysis/make_results.py RESULTS_DIR PAPER_DIR

``RESULTS_DIR`` is the output directory of ``udagger_adwords.cli``; LaTeX tables
are written to ``PAPER_DIR/tables`` and PDF figures to ``PAPER_DIR/figs``.
Every number reported in the paper for AdWords is produced by this script.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from udagger.analysis import Study, fmt, load, standard_outputs

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

EXPERT_ORDER = [
    "greedy",
    "msvv",
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
    "msvv": r"Myopic MSVV $\pi^{m'}$",
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
    "msvv": "MSVV",
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
    "msvv": "#616161",
    "full": "#1f77b4",
    "det": "#d62728",
    "agg5": "#f2b6b6",
    "agg15": "#e88080",
    "agg30": "#c44e52",
    "saa5": "#b9e3b0",
    "saa15": "#5fb85a",
    "saa30": "#1b7a2e",
}
PPA_ONESHOT = {
    "full": 2137.4,
    "det": 1961.9,
    "agg5": 1948.4,
    "agg15": 1946.9,
    "agg30": 1942.2,
    "saa5": 1758.3,
    "saa15": 1695.1,
    "saa30": 1678.5,
}
PPA_DAGGER = {
    "full": 1957.0,
    "det": 1943.3,
    "agg5": 1934.8,
    "agg15": 1937.2,
    "agg30": 1939.1,
    "saa5": 1748.2,
    "saa15": 1693.5,
    "saa30": 1674.2,
}
PPA_ONLINE = {"greedy": 3380.1, "full": 1299.2, "saa30": 1647.2}
# Protocol of the physician study: experts stopped at a 2% MIP gap, and the
# two-stage baseline with 30 scenarios solved to a 0.2% gap.
BASE_GAP = 0.02
BASELINE_GAP = 0.002


STUDY = Study(
    prefix="adwords",
    objective="revenue",
    maximize=True,
    order=EXPERT_ORDER,
    labels=LABELS,
    short=SHORT,
    colors=COLORS,
    group_ends=("msvv", "det", "agg30"),
    base_gap=BASE_GAP,
    gap_expert="saa30",
    baseline_gap=BASELINE_GAP,
    comparisons=(
        (("full", "oneshot"), ("det", "oneshot")),
        (("det", "oneshot"), ("agg30", "oneshot")),
        (("agg30", "oneshot"), ("saa5", "oneshot")),
        (("saa5", "oneshot"), ("saa30", "oneshot")),
        (("full", "oneshot"), ("full", "dagger")),
        (("det", "oneshot"), ("det", "dagger")),
        (("agg30", "oneshot"), ("agg30", "dagger")),
        (("saa30", "oneshot"), ("saa30", "dagger")),
        (("saa30", "dagger"), ("saa5", "oneshot")),
    ),
    curve_experts=("full", "det", "agg30", "saa5", "saa15", "saa30"),
)


STD_LEVELS = (5, 10, 15)
STD_EXPERTS = ("greedy", "full", "det", "agg5", "saa5")


def std_table(results: Path, out: Path) -> pd.DataFrame | None:
    """Table of the uncertainty study (spread of the premium count).

    The main results directory holds the middle level; the other levels are
    in sibling directories suffixed with ``_std<level>``.
    """
    rows = []
    for level in STD_LEVELS:
        folder = results if level == 10 else Path(f"{results}_std{level}")
        if not (folder / "hindsight_test.csv").exists():
            continue
        experts, runs, _ = load(folder, STUDY)
        for name in STD_EXPERTS:
            on = experts[(experts.expert == name) & np.isclose(experts.gap, BASE_GAP)]
            one = runs[
                (runs.expert == name)
                & (runs["mode"] == "oneshot")
                & np.isclose(runs.gap, BASE_GAP)
            ]
            rows.append(
                {
                    "std": level,
                    "expert": name,
                    "online": on["regret"].mean() if not on.empty else np.nan,
                    "oneshot": one["regret"].mean() if not one.empty else np.nan,
                    "sd": one["regret"].std() if len(one) > 1 else np.nan,
                }
            )
    if not rows:
        return None
    df = pd.DataFrame(rows)
    levels = sorted(df["std"].unique())
    cols = "l" + "rr" * len(levels)
    lines = [
        rf"\begin{{tabular}}{{@{{}}{cols}@{{}}}}",
        r"\toprule",
        " & ".join(
            ["$\\sigma$ premium"]
            + [rf"\multicolumn{{2}}{{c}}{{{lv}}}" for lv in levels]
        )
        + r" \\",
        "".join(
            rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(levels))
        ),
        " & ".join(["Expert"] + ["online", "learned"] * len(levels)) + r" \\",
        r"\midrule",
    ]
    for name in STD_EXPERTS:
        cells = [LABELS[name]]
        for lv in levels:
            r = df[(df["std"] == lv) & (df.expert == name)]
            if r.empty:
                cells += ["--", "--"]
                continue
            on, le, sd = r.iloc[0][["online", "oneshot", "sd"]]
            cells.append("--" if np.isnan(on) else f"{on:.2f}")
            cells.append("--" if np.isnan(le) else fmt(le, sd))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tables" / "adwords_std.tex").write_text("\n".join(lines) + "\n")
    df.to_csv(out / "tables" / "adwords_std.csv", index=False)
    return df


def fig_teacher(table: pd.DataFrame, out: Path) -> None:
    """Plot learned-policy quality per expert for both problems."""
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.0))
    ax = axes[0]
    ppa_oneshot, ppa_dagger, ppa_online = PPA_ONESHOT, PPA_DAGGER, PPA_ONLINE
    regret_units = False
    ppa_csv = out / "tables" / "ppa_main.csv"
    if ppa_csv.exists():
        # Rerun of the physician study: costs (legacy/analysis) or regrets in %
        # (ppa/analysis, column metric = "regret").
        ppa = pd.read_csv(ppa_csv).set_index("expert")
        regret_units = "metric" in ppa and (ppa["metric"] == "regret").all()
        ppa_oneshot = ppa["oneshot"].dropna().to_dict()
        ppa_dagger = ppa["dagger"].dropna().to_dict()
        ppa_online = ppa["online"].dropna().to_dict()
        if regret_units:
            ppa_online["full"] = 0.0  # the hindsight optimum has no regret
    names = [n for n in EXPERT_ORDER if n in ppa_oneshot and n in ppa_dagger]
    x = np.arange(len(names))
    ax.bar(
        x - 0.2, [ppa_oneshot[n] for n in names], 0.4, color="#bdbdbd", label="one-shot"
    )
    ax.bar(
        x + 0.2,
        [ppa_dagger[n] for n in names],
        0.4,
        color=[COLORS[n] for n in names],
        label="DAgger",
    )
    offset = 0.3 if regret_units else 15.0  # label above its line
    for key, style, text in (
        ("full", ":", "hindsight"),
        ("saa30", "--", "2S-30 online"),
    ):
        if key in ppa_online:
            ax.axhline(ppa_online[key], color=COLORS[key], ls=style, lw=1)
            ax.text(
                len(names) - 0.5,
                ppa_online[key] + offset,
                text,
                ha="right",
                fontsize=7,
            )
    ax.set_xticks(x, [SHORT[n] for n in names], fontsize=8)
    if regret_units:
        ax.set_ylabel("regret w.r.t. hindsight (%)", fontsize=8)
    else:
        low = min([*ppa_online.values(), *ppa_dagger.values()])
        ax.set_ylim(0.9 * low, 1.05 * max(ppa_oneshot.values()))
        ax.set_ylabel("average cost (lower is better)", fontsize=8)
    ax.set_title("(a) PPA: learned policies", fontsize=9)
    ax.legend(fontsize=7, loc="upper right", frameon=False)

    ax = axes[1]
    tab = table.set_index("expert")
    names = [n for n in EXPERT_ORDER if n in tab.index]
    x = np.arange(len(names))
    ax.bar(x - 0.2, tab.loc[names, "oneshot"], 0.4, color="#bdbdbd", label="one-shot")
    ax.bar(
        x + 0.2,
        tab.loc[names, "dagger"],
        0.4,
        color=[COLORS[n] for n in names],
        label="DAgger",
    )
    ax.scatter(
        x,
        tab.loc[names, "online"],
        marker="_",
        s=180,
        color="black",
        zorder=3,
        label="expert online",
    )
    ax.set_xticks(x, [SHORT[n] for n in names], fontsize=7, rotation=35)
    ax.set_ylabel("regret w.r.t. hindsight (%)", fontsize=8)
    ax.set_title("(b) AdWords: learned policies", fontsize=9)
    ax.legend(fontsize=7, loc="upper right", frameon=False)
    for a in axes:
        a.tick_params(labelsize=7)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "figs" / "teacher.pdf")
    plt.close(fig)


def fig_behaviour(results: Path, out: Path, n_premium_adv: int = 3) -> None:
    """Plot how experts use premium-capable advertisers on regular queries."""
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 2.6), sharex=True)
    bins = np.linspace(0, 1, 21)
    mid = 0.5 * (bins[1:] + bins[:-1])
    for name in ("full", "det", "agg30", "saa5", "saa30", "greedy"):
        path = results / f"{name}_gap{BASE_GAP:g}_oneshot_s0" / "labels.npz"
        if not path.exists():
            continue
        d = np.load(path)
        reg = d["query_class"] == 0
        idx = np.digitize(d["epoch_time"][reg], bins) - 1
        act = d["expert_action"][reg]
        to_prem = (act >= 1) & (act <= n_premium_adv)
        skip = act == 0
        f1 = [
            to_prem[idx == b].mean() if np.any(idx == b) else np.nan for b in range(20)
        ]
        f2 = [skip[idx == b].mean() if np.any(idx == b) else np.nan for b in range(20)]
        axes[0].plot(mid, f1, color=COLORS[name], label=SHORT[name], lw=1.3)
        axes[1].plot(mid, f2, color=COLORS[name], lw=1.3)
    axes[0].set_ylabel("share to premium-capable", fontsize=8)
    axes[1].set_ylabel("share skipped", fontsize=8)
    for a in axes:
        a.set_xlabel("arrival time of regular query", fontsize=8)
        a.tick_params(labelsize=7)
        a.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "figs" / "adwords_behaviour.pdf")
    plt.close(fig)


def main() -> None:
    """Entry point."""
    results, paper = Path(sys.argv[1]), Path(sys.argv[2])
    table = standard_outputs(STUDY, results, paper)
    std = std_table(results, paper)
    if std is not None:
        print(std.to_string())
    fig_teacher(table, paper)
    fig_behaviour(results, paper)


if __name__ == "__main__":
    main()
