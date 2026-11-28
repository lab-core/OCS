"""Build the PPA main table of the paper from the original runs, in regret.

Usage::

    python original_results.py PPA_RESULTS PAPER_DIR

``PPA_RESULTS`` holds the per-episode results of the physician study:

* ``test_episodes.csv.gz``: original runs of Gawas et al. (one-shot cloning
  and every DAgger iteration, sampled actions), extracted from their JMP files;
* ``ppa_original_summary.csv``: the DAgger iteration chosen on validation;
* ``legacy_reruns/online/``: experts used online and the hindsight optimum;
* ``legacy_reruns/eval/``: one-shot reruns (3 seeds) whose policies take
  their most likely action (``test_max.csv``).

Writes ``tables/ppa_main.{tex,csv}`` into ``PAPER_DIR`` with the layout of the
AdWords table (:func:`udagger.analysis.write_main_table`): costs become
regrets ``100 (cost / OPT - 1)`` averaged over the 1,000 test episodes. It
also prints paired comparisons of per-episode regrets used in the text.
"""

from __future__ import annotations

import dataclasses
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from make_results import STUDY
from scipy import stats
from udagger.analysis import _stats, regret, write_main_table

FIRST_TEST_SEED = 5001  # episode k of the original runs has seed 5001 + k
# Original runs: label column -> expert (label 1 is the full-information
# expert, label 2 the deterministic one; the others by scenarios and type).
LABELS = {1: "full", 2: "det"}
# Experts used online: file name -> expert.
ONLINE = {
    "expert_stoch0new1scenarios1_mg0.02.csv": "det",
    "expert_stoch0new1scenarios5_mg0.02.csv": "agg5",
    "expert_stoch0new1scenarios15_mg0.02.csv": "agg15",
    "expert_stoch0new1scenarios30_mg0.02.csv": "agg30",
    "expert_stoch1new1scenarios5_mg0.02.csv": "saa5",
    "expert_stoch1new1scenarios15_mg0.02.csv": "saa15",
    "expert_stoch1new1scenarios30_mg0.02.csv": "saa30",
    "greedy__mg0.02.csv": "greedy",
    "expert_stoch1new1scenarios30_mg0.002.csv": "saa30_baseline",
}


def expert_name(scenarios: int, stoch: int, new: int) -> str:
    """Name of an expert from the parameters of the original code."""
    if scenarios == 1:
        return "det" if new else "full"
    return f"{'saa' if stoch else 'agg'}{scenarios}"


def episode_regrets(df: pd.DataFrame, opt: pd.Series) -> pd.Series:
    """Per-episode regret (%) of a frame with ``seed`` and ``total_cost``."""
    cost = df.set_index("seed")["total_cost"]
    return regret(cost, opt.loc[cost.index], maximize=False)


def load(folder: Path) -> tuple[dict, dict, dict, dict, dict]:
    """Per-episode regrets of every configuration of the physician study.

    Returns:
        ``online`` (expert -> regrets), ``oneshot`` and ``dagger`` (expert
        -> regrets, sampled actions, original runs), ``mode`` (expert -> list
        of per-seed regrets, most likely action) and ``hours`` (expert ->
        DAgger training time).
    """
    legacy = folder / "legacy_reruns"
    hind = pd.read_csv(legacy / "online" / "hindsight__mg0.csv")
    opt = hind.set_index("seed")["total_cost"]
    online = {"full": regret(opt, opt, maximize=False)}
    for file, name in ONLINE.items():
        online[name] = episode_regrets(pd.read_csv(legacy / "online" / file), opt)

    runs = pd.read_csv(folder / "test_episodes.csv.gz")
    runs["seed"] = runs["k"].astype(int) + FIRST_TEST_SEED
    runs = runs[np.isclose(runs["mg"], 0.02)]
    chosen = pd.read_csv(folder / "ppa_original_summary.csv")
    chosen = chosen[(chosen["mode"] == "dagger400") & np.isclose(chosen["mg"], 0.02)]
    chosen = chosen.set_index("expert")["chosen"]
    oneshot, dagger, hours = {}, {}, {}
    for key, df in runs.groupby(["scenarios", "stoch", "new"]):
        name = expert_name(*(int(v) for v in key))
        one = df[df["learn_iter"] == 1]
        oneshot[name] = episode_regrets(one, opt)
        dag = df[(df["learn_iter"] == 400) & (df["instance_id"] == chosen[name])]
        dagger[name] = episode_regrets(dag, opt)
        hours[name] = float(dag["train_time"].iloc[0])

    mode: dict[str, list[float]] = {}
    pattern = re.compile(r"um400_new(\d)_stoch(\d)_sc(\d+)_.*_mg0\.02_")
    for path in sorted(legacy.glob("eval/**/test_max.csv")):
        match = pattern.search(str(path))
        if match is None:
            continue
        new, stoch, scenarios = (int(v) for v in match.groups())
        name = expert_name(scenarios, stoch, new)
        reg = episode_regrets(pd.read_csv(path), opt)
        mode.setdefault(name, []).append(reg.mean())
    return online, oneshot, dagger, mode, hours


def paired(first: pd.Series, second: pd.Series) -> str:
    """Mean paired difference with 95% CI and Wilcoxon p-value."""
    diff = first - second.loc[first.index]
    half = 1.96 * diff.std() / np.sqrt(len(diff))
    p = stats.wilcoxon(diff).pvalue
    return (
        f"{diff.mean():+.2f} [{diff.mean() - half:+.2f}, {diff.mean() + half:+.2f}]"
        f" p={p:.2g}"
    )


def main() -> None:
    """Entry point."""
    folder, paper = Path(sys.argv[1]), Path(sys.argv[2])
    online, oneshot, dagger, mode, hours = load(folder)
    order = [n for n in STUDY.order if n in oneshot]
    study = dataclasses.replace(STUDY, order=order, group_ends=("det", "agg30"))
    cells = {
        name: {
            "online": online[name].mean(),
            "oneshot": _stats(pd.Series([oneshot[name].mean()])),
            "dagger": _stats(pd.Series([dagger[name].mean()])),
            "mode": _stats(pd.Series(mode.get(name, []), dtype=float)),
            "hours": hours[name],
        }
        for name in order
    }
    footer = [
        r"\emph{Other experts used online:} greedy $\pi^m$ "
        f"{online['greedy'].mean():.2f}\\%;",
        r"\quad two-stage $\pi^t_{30}$ at a 0.2\% gap "
        f"{online['saa30_baseline'].mean():.2f}\\%",
    ]
    table = write_main_table(study, cells, paper, footer)
    pd.set_option("display.width", 200)
    print(table.round(2).to_string(index=False))
    print("\nPaired differences of per-episode regret (pp), first - second:")
    for a, b in STUDY.comparisons:
        runs = {"oneshot": oneshot, "dagger": dagger}
        print(f"  {a} vs {b}: {paired(runs[a[1]][a[0]], runs[b[1]][b[0]])}")
    for name in ("agg30", "saa30"):
        print(f"  {name} online vs dagger: {paired(online[name], dagger[name])}")
    print(
        f"  saa30 0.2% online vs dagger: "
        f"{paired(online['saa30_baseline'], dagger['saa30'])}"
    )


if __name__ == "__main__":
    main()
