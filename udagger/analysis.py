"""Tables and figures from the result files of :mod:`udagger.cli`.

Every application writes the same files, so the analysis is shared. Policies
are compared by their *regret*, the relative distance to the hindsight
optimum of each test episode, in percent: ``100 (1 - value / OPT)`` for a
maximised objective and ``100 (value / OPT - 1)`` for a minimised one. An
application describes its study with a :class:`Study` (objective, labels,
colours, MIP gaps, comparisons) and calls the functions below.

Requires pandas, scipy and matplotlib (``pip install udagger[analysis]``).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (the backend must be set first)

MODE_LABEL = {"oneshot": "one-shot", "dagger": "DAgger"}
EXPERT_COLUMNS = ["expert", "gap", "tag", "regret", "regret_se", "sec_per_decision"]
CURVE_COLUMNS = ["expert", "gap", "mode", "seed", "tag", "episodes", "val_regret"]
CURVE_COLUMNS += ["label_hours"]
Config = tuple[str, str]  # (expert name, mode)


@dataclass(frozen=True)
class Study:
    """Description of the results of an application.

    Attributes:
        prefix: Prefix of the table and figure files (e.g. ``adwords``).
        objective: Column of the objective in the result files.
        maximize: Whether the objective is maximised.
        order: Experts in table order.
        labels: LaTeX label of each expert.
        short: Short label of each expert (figures, tests).
        colors: Colour of each expert (figures).
        group_ends: Experts followed by a blank line in the main table.
        base_gap: MIP gap of the main study.
        gap_expert: Expert of the MIP-gap study.
        baseline_gap: Gap of the online baseline of ``gap_expert`` (None: no
            baseline row).
        comparisons: Pairs of configurations compared by :func:`paired_tests`.
        curve_experts: Experts drawn by :func:`fig_curves`.
        replicates: Planned replicates per configuration. When set, the main
            table marks cells with fewer replicates (superscript count) and
            missing cells (``\\pending``, to be defined in the document).
        dagger_tags: ``--tag`` of the DAgger runs reported for an expert,
            when they differ from the main runs (e.g. ``"_ep1000"`` for a
            smaller budget); other experts use the untagged runs.
    """

    prefix: str
    objective: str
    maximize: bool
    order: Sequence[str]
    labels: dict[str, str]
    short: dict[str, str]
    colors: dict[str, str]
    group_ends: Sequence[str] = ()
    base_gap: float = 0.02
    gap_expert: str = "saa30"
    baseline_gap: float | None = 0.002
    comparisons: Sequence[tuple[Config, Config]] = field(default_factory=tuple)
    curve_experts: Sequence[str] = ()
    replicates: int | None = None
    dagger_tags: dict[str, str] = field(default_factory=dict)

    def main_tag(self, expert: str, mode: str) -> str:
        """Tag of the runs of a configuration reported in the tables."""
        return self.dagger_tags.get(expert, "") if mode == "dagger" else ""


def regret(value: pd.Series, opt: pd.Series, maximize: bool) -> pd.Series:
    """Relative distance to the hindsight optimum, in percent."""
    if maximize:
        return 100.0 * (1.0 - value / opt)
    return 100.0 * (value / opt - 1.0)


def _hindsight(results: Path, split: str) -> pd.Series:
    path = results / f"hindsight_{split}.csv"
    return pd.read_csv(path).set_index("seed")["hindsight"]


def load(
    results: Path, study: Study
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the experts used online, the learning runs and their curves.

    Returns:
        ``experts`` (one row per expert and gap), ``runs`` (one row per
        learning run) and ``curves`` (one row per trained policy). Test
        regrets are means of per-episode regrets; the validation regret of
        the curves compares the mean validation objective with the mean
        hindsight value (only means are stored during learning).
    """
    opt = _hindsight(results, "test")
    opt_val = _hindsight(results, "valid").mean()
    col, mx = study.objective, study.maximize

    experts = []
    for path in sorted(results.glob("expert_*.csv")):
        meta = json.loads(path.with_suffix(".json").read_text())
        df = pd.read_csv(path).set_index("seed")
        reg = regret(df[col], opt.loc[df.index], mx)
        experts.append(
            {
                "expert": meta["expert"],
                "gap": meta["gap"],
                "tag": meta.get("options", {}).get("tag", ""),
                "regret": reg.mean(),
                "regret_se": reg.std() / np.sqrt(len(reg)),
                "sec_per_decision": meta["solve_seconds"] / meta["decisions"],
            }
        )

    runs, curves = [], []
    for path in sorted(results.glob("*_s*/test.json")):
        meta = json.loads(path.read_text())
        df = pd.read_csv(path.with_suffix(".csv")).set_index("seed")
        reg = regret(df[col], opt.loc[df.index], mx)
        mode_path = path.parent / "test_mode.csv"
        reg_mode = np.nan
        if mode_path.exists():
            dm = pd.read_csv(mode_path).set_index("seed")
            reg_mode = regret(dm[col], opt.loc[dm.index], mx).mean()
        key = {k: meta[k] for k in ("expert", "gap", "mode", "seed")}
        key["tag"] = meta.get("options", {}).get("tag", "")
        runs.append(
            {
                **key,
                "regret": reg.mean(),
                "regret_mode": reg_mode,
                "best_episodes": meta["best"]["episodes"],
                "label_hours": meta["label_seconds"] / 3600.0,
                "wall_hours": meta["wall_seconds"] / 3600.0,
            }
        )
        for row in meta["curve"]:
            # Results written before udagger stored "val_revenue".
            val = row.get("val_objective", row.get(f"val_{col}"))
            curves.append(
                {
                    **key,
                    "episodes": row["episodes"],
                    "val_regret": float(regret(pd.Series([val]), opt_val, mx)[0]),
                    "label_hours": row["label_seconds"] / 3600.0,
                }
            )
    run_cols = ["expert", "gap", "mode", "seed", "tag", "regret", "regret_mode"]
    run_cols += ["best_episodes", "label_hours", "wall_hours"]
    return (
        pd.DataFrame(experts, columns=EXPERT_COLUMNS),
        pd.DataFrame(runs, columns=run_cols),
        pd.DataFrame(curves, columns=CURVE_COLUMNS),
    )


def summary(experts: pd.DataFrame, runs: pd.DataFrame) -> str:
    """Text summary of the experts and of the runs (mean over replicates)."""
    pd.set_option("display.width", 200)
    by_run = runs.groupby(["expert", "gap", "mode"]).agg(
        regret=("regret", "mean"),
        sd=("regret", "std"),
        regret_mode=("regret_mode", "mean"),
        n=("seed", "count"),
        ep=("best_episodes", "mean"),
        label_h=("label_hours", "mean"),
        wall_h=("wall_hours", "mean"),
    )
    return (
        experts.sort_values(["expert", "gap"]).to_string() + "\n" + by_run.to_string()
    )


def fmt(mean: float, sd: float | None = None) -> str:
    """Format a mean (and standard deviation) for LaTeX."""
    if sd is None or np.isnan(sd):
        return f"{mean:.2f}"
    return f"{mean:.2f} {{\\scriptsize$\\pm${sd:.2f}}}"


def _cell(study: Study, row: pd.DataFrame) -> str:
    """Mean and sd of a configuration, marked when replicates are missing."""
    if row.empty:
        return r"\pending" if study.replicates else "--"
    text = fmt(row["regret"].iloc[0], row["sd"].iloc[0])
    n = int(row["n"].iloc[0])
    if study.replicates and n < study.replicates:
        text += rf"\textsuperscript{{({n})}}"
    return text


def _stats(values: pd.Series) -> pd.DataFrame:
    """One-row frame with the mean, sd and count of replicate regrets."""
    values = values.dropna()
    if values.empty:
        return pd.DataFrame(columns=["regret", "sd", "n"])
    sd = values.std() if len(values) > 1 else np.nan
    return pd.DataFrame({"regret": [values.mean()], "sd": [sd], "n": [len(values)]})


def write_main_table(
    study: Study,
    cells: dict[str, dict[str, object]],
    out: Path,
    footer: str | Sequence[str] | None = None,
) -> pd.DataFrame:
    """Write ``<prefix>_main.tex`` (and ``.csv``) from per-expert cells.

    Every application uses this layout, so that all results read the same
    way: regret of the expert used online, regret of the policies learned
    with one-shot cloning and DAgger (sampled actions) and of the one-shot
    policy taking its most likely action, DAgger gain and time.

    Args:
        study: Description of the study (order, labels, replicates).
        cells: For each expert, ``online`` (float), ``oneshot``, ``dagger``
            and ``mode`` (frames from :func:`_stats`) and ``hours`` (float).
        out: Folder that receives ``tables/``.
        footer: Optional LaTeX text of last rows spanning the table (one
            row per string).

    Returns:
        The numbers of the table, one row per expert.
    """
    rows = []
    lines = [
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        r"& Expert & \multicolumn{3}{c}{Learned policy regret (\%)} & Gain & Time \\",
        r"\cmidrule(lr){3-5}",
        r"Expert & online (\%) & one-shot & DAgger & most likely & (pp) & (h) \\",
        r"\midrule",
    ]
    empty = _stats(pd.Series(dtype=float))
    for name in study.order:
        if name not in cells:
            continue
        c = cells[name]
        one, dag, mode = (c.get(k, empty) for k in ("oneshot", "dagger", "mode"))
        online, hours = float(c.get("online", np.nan)), float(c.get("hours", np.nan))
        o_m = one["regret"].iloc[0] if not one.empty else np.nan
        d_m = dag["regret"].iloc[0] if not dag.empty else np.nan
        m_m = mode["regret"].iloc[0] if not mode.empty else np.nan
        rows.append(
            {"expert": name, "online": online, "oneshot": o_m, "dagger": d_m}
            | {"mode": m_m, "hours": hours}
        )
        online_s = "--" if np.isnan(online) else f"{online:.2f}"
        mode_s = "--" if mode.empty else fmt(m_m, mode["sd"].iloc[0])
        gain_s = "--" if np.isnan(o_m - d_m) else f"{o_m - d_m:+.2f}"
        hours_s = "--" if np.isnan(hours) else f"{hours:.1f}"
        lines.append(
            f"{study.labels[name]} & {online_s} & {_cell(study, one)} & "
            f"{_cell(study, dag)} & {mode_s} & {gain_s} & {hours_s} \\\\"
        )
        if name in study.group_ends:
            lines.append(r"\addlinespace")
    if footer:
        lines.append(r"\midrule")
        for text in [footer] if isinstance(footer, str) else footer:
            lines.append(rf"\multicolumn{{7}}{{@{{}}l}}{{{text}}} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    (tables / f"{study.prefix}_main.tex").write_text("\n".join(lines) + "\n")
    table = pd.DataFrame(rows)
    table["metric"] = "regret"  # percent
    table.to_csv(tables / f"{study.prefix}_main.csv", index=False)
    return table


def main_table(
    study: Study, experts: pd.DataFrame, runs: pd.DataFrame, out: Path
) -> pd.DataFrame:
    """Write ``<prefix>_main.tex`` (and ``.csv``) and return its numbers.

    See :func:`write_main_table` for the layout; the most likely action is
    that of the one-shot policies and the time is the DAgger labelling time.
    """
    base_runs = runs[np.isclose(runs["gap"], study.base_gap)]
    base_exp = experts[np.isclose(experts["gap"], study.base_gap)].set_index("expert")
    cells: dict[str, dict[str, object]] = {}
    for name in study.order:
        one = base_runs[(base_runs.expert == name) & (base_runs["mode"] == "oneshot")]
        dag = base_runs[(base_runs.expert == name) & (base_runs["mode"] == "dagger")]
        if one.empty and dag.empty:
            continue
        cells[name] = {
            "online": base_exp["regret"].get(name, np.nan),
            "oneshot": _stats(one["regret"]),
            "dagger": _stats(dag["regret"]),
            "mode": _stats(one["regret_mode"]),
            "hours": dag["label_hours"].mean() if not dag.empty else np.nan,
        }
    footer = None
    if study.baseline_gap is not None:
        base = experts[
            (experts.expert == study.gap_expert)
            & np.isclose(experts["gap"], study.baseline_gap)
        ]
        if not base.empty:
            label = study.labels[study.gap_expert].split(" ", 1)[-1]
            footer = (
                rf"\emph{{Two-stage baseline:}} {label} online at a "
                f"{100 * study.baseline_gap:g}\\% gap, "
                f"{base['regret'].iloc[0]:.2f}\\% regret"
            )
    return write_main_table(study, cells, out, footer)


def gap_table(
    study: Study, experts: pd.DataFrame, runs: pd.DataFrame, out: Path
) -> pd.DataFrame:
    """Write ``<prefix>_gap.tex``: the MIP-gap study of ``gap_expert``."""
    sub = runs[(runs.expert == study.gap_expert) & (runs["mode"] == "dagger")]
    agg = (
        sub.groupby("gap")
        .agg(
            regret=("regret", "mean"),
            sd=("regret", "std"),
            hours=("label_hours", "mean"),
        )
        .reset_index()
    )
    exp = experts[experts.expert == study.gap_expert].set_index("gap")
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
    (out / "tables" / f"{study.prefix}_gap.tex").write_text("\n".join(lines) + "\n")
    return agg


def episode_regret(
    study: Study, results: Path, expert: str, mode: str, gap: float
) -> pd.Series:
    """Per-test-episode regret of a configuration, averaged over replicates."""
    opt = _hindsight(results, "test")
    per_seed = []
    tag = study.main_tag(expert, mode)
    for path in sorted(results.glob(f"{expert}_gap{gap:g}_{mode}_s*/test.csv")):
        suffix = path.parent.name.rsplit("_s", 1)[-1]
        if not (suffix.endswith(tag) and suffix[: len(suffix) - len(tag)].isdigit()):
            continue  # another --tag variant
        df = pd.read_csv(path).set_index("seed")
        per_seed.append(regret(df[study.objective], opt.loc[df.index], study.maximize))
    if not per_seed:
        return pd.Series(dtype=float)
    return pd.concat(per_seed, axis=1).mean(axis=1)


def paired_tests(study: Study, results: Path, out: Path) -> pd.DataFrame:
    """Compare configurations on the shared test episodes.

    For every pair of ``study.comparisons``, the per-episode regrets
    (averaged over replicates) are compared with a Wilcoxon signed-rank test.
    ``<prefix>_tests.tex/.csv`` give the mean difference (first minus second,
    in percentage points), its 95% confidence interval and the p-value.
    """
    rows = []
    for (e1, m1), (e2, m2) in study.comparisons:
        a = episode_regret(study, results, e1, m1, study.base_gap)
        b = episode_regret(study, results, e2, m2, study.base_gap)
        if a.empty or b.empty:
            continue
        d = (a - b).dropna()
        half = 1.96 * d.std() / np.sqrt(len(d))
        rows.append(
            {
                "first": f"{study.short[e1]} {MODE_LABEL[m1]}",
                "second": f"{study.short[e2]} {MODE_LABEL[m2]}",
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
    tables = out / "tables"
    (tables / f"{study.prefix}_tests.tex").write_text("\n".join(lines) + "\n")
    df.to_csv(tables / f"{study.prefix}_tests.csv", index=False)
    return df


def fig_curves(study: Study, curves: pd.DataFrame, out: Path) -> None:
    """Plot DAgger validation regret against labelled episodes and time.

    Dotted lines: best one-shot policy of the same expert.
    """
    sub = curves[(curves["mode"] == "dagger") & np.isclose(curves.gap, study.base_gap)]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 2.8), sharey=True)
    for name in study.curve_experts:
        d = sub[sub.expert == name]
        if d.empty:
            continue
        g = d.groupby("episodes").agg(
            r=("val_regret", "mean"), h=("label_hours", "mean")
        )
        color = study.colors[name]
        axes[0].plot(g.index, g["r"], color=color, label=study.short[name], lw=1.3)
        axes[1].plot(g["h"], g["r"], color=color, lw=1.3)
        one = curves[
            (curves["mode"] == "oneshot")
            & (curves.expert == name)
            & np.isclose(curves.gap, study.base_gap)
        ]
        if not one.empty:
            best = one.groupby("seed")["val_regret"].min().mean()
            axes[0].axhline(best, color=color, ls=":", lw=0.9)
    axes[0].set_xlabel("expert-labelled episodes", fontsize=8)
    axes[1].set_xlabel("expert labelling time (CPU hours)", fontsize=8)
    axes[1].set_xscale("log")
    axes[0].set_ylabel("validation regret (%)", fontsize=8)
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(fontsize=7, ncol=2, frameon=False)
    for a in axes:
        a.tick_params(labelsize=7)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "figs" / f"{study.prefix}_curves.pdf")
    plt.close(fig)


def standard_outputs(study: Study, results: Path, out: Path) -> pd.DataFrame:
    """Write the tables and figures every application has; return the main table.

    Writes ``<prefix>_main``, ``<prefix>_gap`` and ``<prefix>_tests`` tables
    and the ``<prefix>_curves`` figure, and prints a summary.
    """
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figs").mkdir(parents=True, exist_ok=True)
    experts, runs, curves = load(results, study)
    # Runs saved under a --tag are variants, not replicates of the main study,
    # except the DAgger runs that the study reports for some experts.
    experts = experts[experts["tag"] == ""]
    main = [study.main_tag(e, m) for e, m in zip(runs["expert"], runs["mode"])]
    runs = runs[runs["tag"] == pd.Series(main, index=runs.index)]
    curves = curves[curves["tag"] == ""]
    print(summary(experts, runs))
    table = main_table(study, experts, runs, out)
    gap_table(study, experts, runs, out)
    print(paired_tests(study, results, out).to_string())
    fig_curves(study, curves, out)
    return table
