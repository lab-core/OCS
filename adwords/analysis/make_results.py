"""Build the AdWords tables and figures of the CPAIOR paper from raw results.

Usage::

    python analysis/make_results.py RESULTS_DIR PAPER_DIR

``RESULTS_DIR`` is the output directory of ``ilco_adwords.cli``; LaTeX tables
are written to ``PAPER_DIR/tables`` and PDF figures to ``PAPER_DIR/figs``.
Every number reported in the paper for AdWords is produced by this script.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

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


def regret(revenue: pd.Series, opt: pd.Series) -> pd.Series:
    """Percentage of the hindsight optimum lost by a policy."""
    return 100.0 * (1.0 - revenue / opt)


def load(results: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load expert evaluations, learned-policy tests and learning curves."""
    opt = pd.read_csv(results / "hindsight_test.csv").set_index("seed")["hindsight"]
    opt_val = pd.read_csv(results / "hindsight_valid.csv")["hindsight"].mean()

    experts = []
    for path in sorted(results.glob("expert_*.csv")):
        meta = json.loads(path.with_suffix(".json").read_text())
        df = pd.read_csv(path).set_index("seed")
        reg = regret(df["revenue"], opt.loc[df.index])
        experts.append(
            {
                "expert": meta["expert"],
                "gap": meta["gap"],
                "regret": reg.mean(),
                "regret_se": reg.std() / np.sqrt(len(reg)),
                "sec_per_decision": meta["solve_seconds"] / meta["decisions"],
            }
        )
    experts_df = pd.DataFrame(experts)

    runs = []
    curves = []
    for path in sorted(results.glob("*_s*/test.json")):
        meta = json.loads(path.read_text())
        df = pd.read_csv(path.with_suffix(".csv")).set_index("seed")
        reg = regret(df["revenue"], opt.loc[df.index])
        mode_path = path.parent / "test_mode.csv"
        reg_mode = np.nan
        if mode_path.exists():
            dm = pd.read_csv(mode_path).set_index("seed")
            reg_mode = regret(dm["revenue"], opt.loc[dm.index]).mean()
        runs.append(
            {
                "expert": meta["expert"],
                "gap": meta["gap"],
                "mode": meta["mode"],
                "seed": meta["seed"],
                "regret": reg.mean(),
                "regret_mode": reg_mode,
                "best_episodes": meta["best"]["episodes"],
                "label_hours": meta["label_seconds"] / 3600.0,
                "wall_hours": meta["wall_seconds"] / 3600.0,
            }
        )
        for row in meta["curve"]:
            curves.append(
                {
                    "expert": meta["expert"],
                    "gap": meta["gap"],
                    "mode": meta["mode"],
                    "seed": meta["seed"],
                    "episodes": row["episodes"],
                    "val_regret": 100.0 * (1.0 - row["val_revenue"] / opt_val),
                    "label_hours": row["label_seconds"] / 3600.0,
                }
            )
    return experts_df, pd.DataFrame(runs), pd.DataFrame(curves)


def fmt(mean: float, sd: float | None = None) -> str:
    """Format a mean (and standard deviation) for LaTeX."""
    if sd is None or np.isnan(sd):
        return f"{mean:.2f}"
    return f"{mean:.2f} {{\\scriptsize$\\pm${sd:.2f}}}"


def main_table(experts: pd.DataFrame, runs: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Write the main AdWords table and return its numbers."""
    exact_runs = runs[np.isclose(runs["gap"], BASE_GAP)]
    exact_exp = experts[np.isclose(experts["gap"], BASE_GAP)].set_index("expert")
    agg = (
        exact_runs.groupby(["expert", "mode"])
        .agg(
            regret=("regret", "mean"),
            sd=("regret", "std"),
            hours=("label_hours", "mean"),
            n=("seed", "count"),
        )
        .reset_index()
    )
    rows = []
    lines = [
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"& Expert & \multicolumn{2}{c}{Learned policy regret (\%)} & Gain & Label \\",
        r"\cmidrule(lr){3-4}",
        r"Expert & online (\%) & one-shot & DAgger & (pp) & time (h) \\",
        r"\midrule",
    ]
    for name in EXPERT_ORDER:
        one = agg[(agg.expert == name) & (agg["mode"] == "oneshot")]
        dag = agg[(agg.expert == name) & (agg["mode"] == "dagger")]
        if one.empty and dag.empty:
            continue
        online = exact_exp["regret"].get(name, np.nan)
        o_m = one["regret"].iloc[0] if not one.empty else np.nan
        d_m = dag["regret"].iloc[0] if not dag.empty else np.nan
        hours = dag["hours"].iloc[0] if not dag.empty else np.nan
        rows.append(
            {
                "expert": name,
                "online": online,
                "oneshot": o_m,
                "dagger": d_m,
                "hours": hours,
            }
        )
        o_s = fmt(o_m, one["sd"].iloc[0]) if not one.empty else "--"
        d_s = fmt(d_m, dag["sd"].iloc[0]) if not dag.empty else "--"
        online_s = f"{online:.2f}" if not np.isnan(online) else "--"
        gain_s = "--" if np.isnan(o_m - d_m) else f"{o_m - d_m:+.2f}"
        hours_s = "--" if np.isnan(hours) else f"{hours:.1f}"
        lines.append(
            f"{LABELS[name]} & {online_s} & {o_s} & {d_s} & "
            f"{gain_s} & {hours_s} \\\\"
        )
        if name in ("msvv", "det", "agg30"):
            lines.append(r"\addlinespace")
    base = experts[
        (experts.expert == "saa30") & np.isclose(experts["gap"], BASELINE_GAP)
    ]
    if not base.empty:
        lines += [
            r"\midrule",
            r"\multicolumn{6}{@{}l}{\emph{Two-stage baseline:} $\pi^t_{30}$ online "
            f"at a 0.2\\% gap, {base['regret'].iloc[0]:.2f}\\% regret}} \\\\",
        ]
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tables" / "adwords_main.tex").write_text("\n".join(lines) + "\n")
    return pd.DataFrame(rows)


def gap_table(experts: pd.DataFrame, runs: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Write the MIP-gap table for the two-stage expert with 30 scenarios."""
    sub = runs[(runs.expert == "saa30") & (runs["mode"] == "dagger")]
    agg = (
        sub.groupby("gap")
        .agg(
            regret=("regret", "mean"),
            sd=("regret", "std"),
            hours=("label_hours", "mean"),
        )
        .reset_index()
    )
    exp = experts[experts.expert == "saa30"].set_index("gap")
    lines = [
        r"\begin{tabular}{@{}rrrrr@{}}",
        r"\toprule",
        r"MIP gap & Expert online & Learned (DAgger) & Label time & Time per \\",
        r"(\%) & regret (\%) & regret (\%) & (h) & decision (s) \\",
        r"\midrule",
    ]
    for _, row in agg.iterrows():
        g = row["gap"]
        on = exp["regret"].get(g, np.nan)
        spd = exp["sec_per_decision"].get(g, np.nan)
        lines.append(
            f"{100 * g:g} & {on:.2f} & {fmt(row['regret'], row['sd'])} & "
            f"{row['hours']:.1f} & {spd:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tables" / "adwords_gap.tex").write_text("\n".join(lines) + "\n")
    return agg


def episode_regret(results: Path, expert: str, mode: str, gap: float) -> pd.Series:
    """Per-test-episode regret of a configuration, averaged over seeds."""
    opt = pd.read_csv(results / "hindsight_test.csv").set_index("seed")["hindsight"]
    per_seed = []
    for path in sorted(results.glob(f"{expert}_gap{gap:g}_{mode}_s*/test.csv")):
        df = pd.read_csv(path).set_index("seed")
        per_seed.append(regret(df["revenue"], opt.loc[df.index]))
    if not per_seed:
        return pd.Series(dtype=float)
    return pd.concat(per_seed, axis=1).mean(axis=1)


COMPARISONS = [
    (("full", "oneshot"), ("det", "oneshot")),
    (("det", "oneshot"), ("agg30", "oneshot")),
    (("agg30", "oneshot"), ("saa5", "oneshot")),
    (("saa5", "oneshot"), ("saa30", "oneshot")),
    (("full", "oneshot"), ("full", "dagger")),
    (("det", "oneshot"), ("det", "dagger")),
    (("agg30", "oneshot"), ("agg30", "dagger")),
    (("saa30", "oneshot"), ("saa30", "dagger")),
    (("saa30", "dagger"), ("saa5", "oneshot")),
]
MODE_LABEL = {"oneshot": "one-shot", "dagger": "DAgger"}


def paired_tests(results: Path, out: Path) -> pd.DataFrame:
    """Compare configurations on the shared test episodes.

    For every pair, the per-episode regrets (averaged over seeds) are compared
    with a Wilcoxon signed-rank test; the table gives the mean difference
    (first minus second, in percentage points), its 95% confidence interval
    and the p-value.
    """
    rows = []
    for (e1, m1), (e2, m2) in COMPARISONS:
        a = episode_regret(results, e1, m1, BASE_GAP)
        b = episode_regret(results, e2, m2, BASE_GAP)
        if a.empty or b.empty:
            continue
        d = (a - b).dropna()
        half = 1.96 * d.std() / np.sqrt(len(d))
        rows.append(
            {
                "first": f"{SHORT[e1]} {MODE_LABEL[m1]}",
                "second": f"{SHORT[e2]} {MODE_LABEL[m2]}",
                "diff": d.mean(),
                "low": d.mean() - half,
                "high": d.mean() + half,
                "p": stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0,
                "n": len(d),
            }
        )
    df = pd.DataFrame(rows)
    lines = [
        r"\begin{tabular}{@{}llrr@{}}",
        r"\toprule",
        r"First & Second & Difference (pp) [95\% CI] & $p$ \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        p = "$<10^{-4}$" if r["p"] < 1e-4 else f"{r['p']:.3f}"
        lines.append(
            f"{r['first']} & {r['second']} & {r['diff']:+.2f} "
            f"[{r['low']:+.2f}, {r['high']:+.2f}] & {p} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tables" / "adwords_tests.tex").write_text("\n".join(lines) + "\n")
    df.to_csv(out / "tables" / "adwords_tests.csv", index=False)
    return df


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
        experts, runs, _ = load(folder)
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
    ppa_csv = out / "tables" / "ppa_main.csv"
    if ppa_csv.exists():
        # Rerun of the physician study (ppa/analysis/make_results.py).
        ppa = pd.read_csv(ppa_csv).set_index("expert")
        ppa_oneshot = ppa["oneshot"].dropna().to_dict()
        ppa_dagger = ppa["dagger"].dropna().to_dict()
        ppa_online = ppa["online"].dropna().to_dict()
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
    ax.axhline(ppa_online["full"], color=COLORS["full"], ls=":", lw=1)
    ax.text(
        len(names) - 0.5, ppa_online["full"] + 15, "hindsight", ha="right", fontsize=7
    )
    ax.axhline(ppa_online["saa30"], color=COLORS["saa30"], ls="--", lw=1)
    ax.text(
        len(names) - 0.5,
        ppa_online["saa30"] + 15,
        "2S-30 online",
        ha="right",
        fontsize=7,
    )
    ax.set_xticks(x, [SHORT[n] for n in names], fontsize=8)
    ax.set_ylim(0.9 * ppa_online["full"], 1.05 * max(ppa_oneshot.values()))
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


def fig_curves(curves: pd.DataFrame, out: Path) -> None:
    """Plot DAgger validation regret against labelled episodes and time."""
    sub = curves[(curves["mode"] == "dagger") & np.isclose(curves.gap, BASE_GAP)]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 2.8), sharey=True)
    for name in ("full", "det", "agg30", "saa5", "saa15", "saa30"):
        d = sub[sub.expert == name]
        if d.empty:
            continue
        g = d.groupby("episodes").agg(
            r=("val_regret", "mean"), h=("label_hours", "mean")
        )
        axes[0].plot(g.index, g["r"], color=COLORS[name], label=SHORT[name], lw=1.3)
        axes[1].plot(g["h"], g["r"], color=COLORS[name], lw=1.3)
        one = curves[
            (curves["mode"] == "oneshot")
            & (curves.expert == name)
            & np.isclose(curves.gap, BASE_GAP)
        ]
        if not one.empty:
            best = one.groupby("seed")["val_regret"].min().mean()
            axes[0].axhline(best, color=COLORS[name], ls=":", lw=0.9)
    axes[0].set_xlabel("expert-labelled episodes", fontsize=8)
    axes[1].set_xlabel("expert labelling time (CPU hours)", fontsize=8)
    axes[1].set_xscale("log")
    axes[0].set_ylabel("validation regret (%)", fontsize=8)
    axes[0].legend(fontsize=7, ncol=2, frameon=False)
    for a in axes:
        a.tick_params(labelsize=7)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "figs" / "adwords_curves.pdf")
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
    (paper / "tables").mkdir(parents=True, exist_ok=True)
    (paper / "figs").mkdir(parents=True, exist_ok=True)
    experts, runs, curves = load(results)
    pd.set_option("display.width", 200)
    print(experts.sort_values(["expert", "gap"]).to_string())
    print(
        runs.groupby(["expert", "gap", "mode"])
        .agg(
            regret=("regret", "mean"),
            sd=("regret", "std"),
            regret_mode=("regret_mode", "mean"),
            n=("seed", "count"),
            ep=("best_episodes", "mean"),
            label_h=("label_hours", "mean"),
            wall_h=("wall_hours", "mean"),
        )
        .to_string()
    )
    table = main_table(experts, runs, paper)
    gap_table(experts, runs, paper)
    print(paired_tests(results, paper).to_string())
    std = std_table(results, paper)
    if std is not None:
        print(std.to_string())
    fig_teacher(table, paper)
    fig_curves(curves, paper)
    fig_behaviour(results, paper)
    table.to_csv(paper / "tables" / "adwords_main.csv", index=False)


if __name__ == "__main__":
    main()
