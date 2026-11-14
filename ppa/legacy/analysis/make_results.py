"""Build the PPA tables of the CPAIOR paper from the rerun on Rorqual.

Usage::

    python ppa/legacy/analysis/make_results.py DATA_DIR PAPER_DIR

``DATA_DIR`` is ``ppa/data`` (training folders and ``evaluation/``). The
script writes ``tables/ppa_main.tex``, ``tables/ppa_gap.tex``,
``tables/ppa_tests.tex`` and ``tables/ppa_main.csv`` (read by the shared
figure of ``adwords/analysis/make_results.py``) into ``PAPER_DIR``.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE_GAP = 0.02
BASELINE_GAP = 0.002
EXPERT_ORDER = [
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
    "full": "Full-info",
    "det": "Determ.",
    "agg30": "Agg-30",
    "saa5": "2S-5",
    "saa30": "2S-30",
}
RUN_RE = re.compile(
    r"NN_iter(?P<iter>\d+)_runs(?P<runs>\d+)_um(?P<um>\d+)"
    r"_new(?P<new>\d)_stoch(?P<stoch>\d)_sc(?P<sc>\d+)_van\d_gat\d_tl\d+"
    r"_mg(?P<mg>[\d.e-]+)_nc\d+_seed(?P<seed>\d+)"
)


def expert_name(new: int, stoch: int, scenarios: int) -> str:
    """Name of the expert defined by the PPA options."""
    if stoch == 1:
        return f"saa{scenarios}"
    if new == 0:
        return "full"
    return "det" if scenarios == 1 else f"agg{scenarios}"


def _hours(folder: Path) -> float:
    path = folder / "Train_stats" / "Train_time.json"
    if not path.exists():
        return np.nan
    t = json.loads(path.read_text())
    delta = datetime.fromisoformat(t["end"]) - datetime.fromisoformat(t["start"])
    return delta.total_seconds() / 3600.0


def load_runs(data: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the test results of every training run.

    Returns:
        One row per run (mean test cost of both action rules, training
        time) and one row per run and test episode (sampled actions).
    """
    runs, episodes = [], []
    for test in sorted(
        data.glob("Data_Dagger_*/Instances_*/NN_*/eval/test_random.csv")
    ):
        folder = test.parent.parent
        m = RUN_RE.fullmatch(folder.name)
        if m is None:
            continue
        name = expert_name(int(m["new"]), int(m["stoch"]), int(m["sc"]))
        # One-shot runs train once at the end (um = number of iterations).
        mode = "dagger" if int(m["um"]) == 1 else "oneshot"
        df = pd.read_csv(test)
        mode_path = test.with_name("test_max.csv")
        cost_max = (
            pd.read_csv(mode_path)["total_cost"].mean()
            if mode_path.exists()
            else np.nan
        )
        runs.append(
            {
                "expert": name,
                "mode": mode,
                "gap": float(m["mg"]),
                "seed": int(m["seed"]),
                "model": int(df["model"].iloc[0]),
                "cost": df["total_cost"].mean(),
                "cost_max": cost_max,
                "rej_p1": df["rejected_p1"].mean(),
                "hours": _hours(folder),
            }
        )
        episodes.append(
            df[["seed", "total_cost"]].assign(
                expert=name, mode=mode, gap=float(m["mg"]), run=int(m["seed"])
            )
        )
    return pd.DataFrame(runs), pd.concat(episodes) if episodes else pd.DataFrame()


def load_online(data: Path) -> pd.DataFrame:
    """Load the experts used online on the test episodes."""
    rows = []
    for path in sorted((data / "evaluation").glob("*.csv")):
        kind, rest = path.stem.split("_", 1)
        opts, gap = rest.rsplit("_mg", 1)
        if kind == "expert":
            m = re.fullmatch(r"stoch(\d)new(\d)scenarios(\d+)", opts)
            name = expert_name(int(m[2]), int(m[1]), int(m[3]))
        else:
            name = kind
        df = pd.read_csv(path)
        rows.append(
            {
                "expert": name,
                "gap": float(gap),
                "cost": df["total_cost"].mean(),
                "sec_per_episode": df["sol_time"].mean(),
            }
        )
    return pd.DataFrame(rows)


def fmt(mean: float, sd: float) -> str:
    """Format a mean and a standard deviation for LaTeX."""
    if np.isnan(mean):
        return "--"
    if np.isnan(sd):
        return f"{mean:.1f}"
    return f"{mean:.1f} {{\\scriptsize$\\pm${sd:.1f}}}"


def _num(x: float, spec: str = ".1f") -> str:
    return "--" if np.isnan(x) else format(x, spec)


def main_table(runs: pd.DataFrame, online: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Write the main PPA table and return its numbers."""
    base = runs[np.isclose(runs.gap, BASE_GAP)]
    agg = (
        base.groupby(["expert", "mode"])
        .agg(
            cost=("cost", "mean"),
            sd=("cost", "std"),
            cost_max=("cost_max", "mean"),
            hours=("hours", "mean"),
        )
        .reset_index()
        .set_index(["expert", "mode"])
    )
    on = online[np.isclose(online.gap, BASE_GAP)].set_index("expert")["cost"]
    hind = online[online.expert == "hindsight"]["cost"]
    lines = [
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        r"& Expert & \multicolumn{2}{c}{Learned policy cost} & Gain "
        r"& \multicolumn{2}{c}{Train time (h)} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){6-7}",
        r"Expert & online & one-shot & DAgger & (\%) & one-shot & DAgger \\",
        r"\midrule",
    ]
    rows = []
    for name in EXPERT_ORDER:
        one = agg.loc[(name, "oneshot")] if (name, "oneshot") in agg.index else None
        dag = agg.loc[(name, "dagger")] if (name, "dagger") in agg.index else None
        if one is None and dag is None:
            continue
        o_m = one["cost"] if one is not None else np.nan
        d_m = dag["cost"] if dag is not None else np.nan
        online_cost = (
            hind.iloc[0] if name == "full" and not hind.empty else on.get(name, np.nan)
        )
        rows.append(
            {
                "expert": name,
                "online": online_cost,
                "oneshot": o_m,
                "dagger": d_m,
                "oneshot_max": one["cost_max"] if one is not None else np.nan,
                "dagger_max": dag["cost_max"] if dag is not None else np.nan,
            }
        )
        gain = 100.0 * (o_m - d_m) / o_m
        lines.append(
            f"{LABELS[name]} & {_num(online_cost)} & "
            f"{fmt(o_m, one['sd'] if one is not None else np.nan)} & "
            f"{fmt(d_m, dag['sd'] if dag is not None else np.nan)} & "
            f"{_num(gain)} & "
            f"{_num(one['hours'] if one is not None else np.nan)} & "
            f"{_num(dag['hours'] if dag is not None else np.nan)} \\\\"
        )
        if name in ("det", "agg30"):
            lines.append(r"\addlinespace")
    extra = []
    for name, label in (("greedy", r"greedy $\pi^m$"), ("hindsight", "hindsight")):
        if name in on.index or (name == "hindsight" and not hind.empty):
            cost = hind.iloc[0] if name == "hindsight" else on[name]
            extra.append(f"{label} {cost:.1f}")
    base_on = online[(online.expert == "saa30") & np.isclose(online.gap, BASELINE_GAP)]
    if not base_on.empty:
        extra.append(
            rf"two-stage $\pi^t_{{30}}$ (0.2\% gap) {base_on['cost'].iloc[0]:.1f}"
        )
    if extra:
        lines += [
            r"\midrule",
            r"\multicolumn{7}{@{}l}{\emph{Baselines:} " + "; ".join(extra) + r"} \\",
        ]
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tables" / "ppa_main.tex").write_text("\n".join(lines) + "\n")
    table = pd.DataFrame(rows)
    table.to_csv(out / "tables" / "ppa_main.csv", index=False)
    return table


def gap_table(runs: pd.DataFrame, online: pd.DataFrame, out: Path) -> None:
    """Write the MIP-gap table of the two-stage expert with 30 scenarios."""
    sub = runs[(runs.expert == "saa30") & (runs["mode"] == "dagger")]
    agg = sub.groupby("gap").agg(
        cost=("cost", "mean"), sd=("cost", "std"), hours=("hours", "mean")
    )
    on = online[online.expert == "saa30"].set_index("gap")["cost"]
    lines = [
        r"\begin{tabular}{@{}rrrr@{}}",
        r"\toprule",
        r"MIP gap (\%) & Expert online & Learned (DAgger) & Train time (h) \\",
        r"\midrule",
    ]
    for gap, row in agg.iterrows():
        lines.append(
            f"{100 * gap:g} & {_num(on.get(gap, np.nan))} & "
            f"{fmt(row['cost'], row['sd'])} & {_num(row['hours'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tables" / "ppa_gap.tex").write_text("\n".join(lines) + "\n")


COMPARISONS = [
    (("full", "oneshot"), ("det", "oneshot")),
    (("det", "oneshot"), ("agg30", "oneshot")),
    (("agg30", "oneshot"), ("saa5", "oneshot")),
    (("saa5", "oneshot"), ("saa30", "oneshot")),
    (("full", "oneshot"), ("full", "dagger")),
    (("det", "oneshot"), ("det", "dagger")),
    (("agg30", "oneshot"), ("agg30", "dagger")),
    (("saa30", "oneshot"), ("saa30", "dagger")),
]
MODE_LABEL = {"oneshot": "one-shot", "dagger": "DAgger"}


def paired_tests(episodes: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Wilcoxon tests on the per-episode costs (averaged over seeds)."""
    base = episodes[np.isclose(episodes.gap, BASE_GAP)]
    mean = base.groupby(["expert", "mode", "seed"])["total_cost"].mean()
    rows = []
    for (e1, m1), (e2, m2) in COMPARISONS:
        if (e1, m1) not in mean.index or (e2, m2) not in mean.index:
            continue
        d = (mean.loc[(e1, m1)] - mean.loc[(e2, m2)]).dropna()
        half = 1.96 * d.std() / np.sqrt(len(d))
        rows.append(
            {
                "first": f"{SHORT.get(e1, e1)} {MODE_LABEL[m1]}",
                "second": f"{SHORT.get(e2, e2)} {MODE_LABEL[m2]}",
                "diff": d.mean(),
                "low": d.mean() - half,
                "high": d.mean() + half,
                "p": stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0,
            }
        )
    df = pd.DataFrame(rows)
    lines = [
        r"\begin{tabular}{@{}llrr@{}}",
        r"\toprule",
        r"First & Second & Difference [95\% CI] & $p$ \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        p = "$<10^{-4}$" if r["p"] < 1e-4 else f"{r['p']:.3f}"
        lines.append(
            f"{r['first']} & {r['second']} & {r['diff']:+.1f} "
            f"[{r['low']:+.1f}, {r['high']:+.1f}] & {p} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tables" / "ppa_tests.tex").write_text("\n".join(lines) + "\n")
    return df


def main() -> None:
    """Entry point."""
    data, paper = Path(sys.argv[1]), Path(sys.argv[2])
    (paper / "tables").mkdir(parents=True, exist_ok=True)
    runs, episodes = load_runs(data)
    online = load_online(data)
    pd.set_option("display.width", 200)
    print(online.to_string())
    print(
        runs.groupby(["expert", "gap", "mode"])
        .agg(
            cost=("cost", "mean"),
            sd=("cost", "std"),
            cost_max=("cost_max", "mean"),
            n=("seed", "count"),
            hours=("hours", "mean"),
        )
        .to_string()
    )
    print(main_table(runs, online, paper).to_string())
    gap_table(runs, online, paper)
    if not episodes.empty:
        print(paired_tests(episodes, paper).to_string())


if __name__ == "__main__":
    main()
