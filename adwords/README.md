# Stochastic AdWords

Online allocation of search queries to budget-constrained advertisers, used as
the second case study of the CPAIOR 2027 paper on expert design for imitation
learning.

## Problem

- 8 advertisers and 12 query types: 8 *regular* types and 4 *premium* types.
- Premium queries are bid on only by the 3 premium-capable advertisers.
- Serving query type `j` with advertiser `i` needs a remaining budget of at
  least the bid `b_ij`, earns `b_ij` and uses `b_ij` of the budget. A query
  can also be skipped.
- The number of regular and premium queries follows N(120, 20) and N(30, 10).
- Arrival times follow Beta(1, 1) for regular and Beta(4, 1) for premium
  queries, so premium queries come late.
- Budgets are 1.2 (regular advertisers) and 0.7 (premium-capable advertisers)
  times their expected spend on their own query class.

The difficulty is to keep enough budget of the premium-capable advertisers for
the late premium queries. The offline problem only depends on the number of
queries of each type, so every MIP expert solves a compact type-count model
(`udagger_adwords/solver.py`).

## Layout

| Path | Content |
|---|---|
| `udagger_adwords/instance.py` | market generator, episode sampling and seeds |
| `udagger_adwords/simulator.py` | simulator and Bayesian posterior on the remaining queries |
| `udagger_adwords/solver.py` | type-count MIPs (Gurobi) and hindsight optimum |
| `udagger_adwords/features.py` | state features of the learned policy |
| `udagger_adwords/experts.py` | greedy, MSVV, full-information, deterministic, aggregated, two-stage |
| `udagger_adwords/problem.py` | the AdWords problem and its episodes for `udagger` |
| `udagger_adwords/cli.py` | the application for the generic command line (`udagger.cli`) |
| `slurm/make_grid.py` | experiment grids (run with the shared `../slurm/array.sh`) |
| `analysis/make_results.py` | tables and figures of the paper (with `udagger.analysis`) |
| `scripts/calibrate.py` | calibration of the budget tightness |
| `tests/` | unit tests and a regression test against the pre-refactor results |

## Installation

Python 3.11+ and a Gurobi licence are required. The learning code is the
`udagger` package at the repository root:

```bash
pip install -e ..            # udagger
pip install -r requirements.txt
PYTHONPATH=$PWD pytest -q tests
```

Without installing `udagger`, use `PYTHONPATH=$PWD/..:$PWD` instead. The
command line, options and output files are the same as in the PPA example
(`../ppa`).

## Reproducing the experiments

All commands are run from `adwords/`. Results go to `--out`, or to
`$ADWORDS_OUT` if `--out` is not given.

```bash
# Hindsight optimum on the validation and test episodes.
python -m udagger_adwords.cli --workers 32 hindsight
# One expert used online on the test episodes.
python -m udagger_adwords.cli --workers 32 expert --expert saa --scenarios 30
# One learning run.
python -m udagger_adwords.cli --workers 32 learn --expert saa --scenarios 5 \
    --seed 0 --mode dagger
```

To run the full study on a SLURM cluster, from the repository root with the
shared job script (results in `$UDAGGER_OUT/adwords`, by default
`~/scratch/udagger/adwords`; give your allocation with `--account`):

```bash
python adwords/slurm/make_grid.py
sbatch --array=1-15 --time=3:00:00 slurm/array.sh adwords adwords/slurm/expert_grid.txt
# One-shot runs are the odd lines and DAgger runs the even lines (then 101-115).
sbatch --array=1-99:2 --time=3:00:00 slurm/array.sh adwords adwords/slurm/learn_grid.txt
sbatch --array=2-100:2,101-115 --time=3-00:00:00 slurm/array.sh adwords adwords/slurm/learn_grid.txt
sbatch --array=1-62 --time=3:00:00 slurm/array.sh adwords adwords/slurm/std_grid.txt
```

Then run `python analysis/make_results.py RESULTS_DIR PAPER_DIR`. It writes
`tables/adwords_*.tex` and `figs/*.pdf` into the paper folder.

## Experimental protocol

The defaults of `udagger_adwords.cli` follow the physician study (`../ppa`).

- **Episodes:** 4,000 expert-labelled training episodes, about 149 queries
  each (≈600,000 labelled states).
- **One-shot cloning:** the expert controls all 4,000 episodes, and one
  policy is trained on them.
- **DAgger:** 400 iterations of 10 episodes, with β₀ = 1 and βᵢ = 0.8ⁱ. The
  policy is retrained after every iteration, warm-started from the previous
  one. All 400 policies are scored on validation, and the best is kept.
- **Training:** MLP with 2 hidden layers of 128 units, 100 epochs, batches of
  256 and Adam (lr 10⁻³). Soft cross-entropy on the expert distribution, with
  the best epoch chosen on a 20% hold-out.
- **Learned policy:** the action is sampled from the predicted distribution
  (`--policy-action sample`). Test results are also stored for the most likely
  action (`test_mode.csv`).
- **Aggregated expert:** executes an action sampled from its votes.
- **Expert MIPs:** stopped at a 2% gap or after 30 s. The gap study adds 1, 5
  and 10% for the 30-scenario two-stage expert. The two-stage baseline used
  online is solved to a 0.2% gap.
- **Evaluation:** 1,000 validation and 1,000 test episodes, shared by all
  policies.
- **Seeds:** 5 per configuration; the physician study used one.

## Calibration of the budgets

`scripts/calibrate.py 100 32` scores each setting on 100 calibration episodes,
which are disjoint from the training, validation and test episodes. The score
is the margin of the two-stage expert (10 scenarios) over the best myopic rule,
with ties broken by its margin over the deterministic expert. The numbers are
revenue as a fraction of the hindsight optimum:

| regular | premium | greedy | MSVV | det. | agg-10 | 2S-10 | score |
|---|---|---|---|---|---|---|---|
| 0.8 | 0.7 | 0.955 | 0.971 | 0.979 | 0.981 | 0.995 | 0.024 |
| 0.8 | 1.0 | 0.949 | 0.966 | 0.974 | 0.977 | 0.990 | 0.024 |
| 1.0 | 0.7 | 0.915 | 0.939 | 0.962 | 0.965 | 0.991 | 0.051 |
| 1.0 | 1.0 | 0.914 | 0.931 | 0.956 | 0.961 | 0.983 | 0.053 |
| **1.2** | **0.7** | 0.885 | 0.904 | 0.950 | 0.960 | 0.989 | **0.085** |
| 1.2 | 1.0 | 0.886 | 0.903 | 0.944 | 0.951 | 0.978 | 0.076 |

The selected setting (1.2, 0.7) is the default of `udagger_adwords.cli`. The
experts of this table use optimal MIPs, a 10 s limit and majority votes
(`exact`). With the settings of the study (`protocol`: 2% gap, 30 s, sampled
votes), the same setting is selected (score 0.077, next best 0.070).
