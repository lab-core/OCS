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
(`ilco_adwords/solver.py`).

## Layout

| Path | Content |
|---|---|
| `ilco_adwords/market.py` | instance generator and episode sampling |
| `ilco_adwords/env.py` | simulator and Bayesian posterior on the remaining queries |
| `ilco_adwords/solver.py` | type-count MIPs (Gurobi) and hindsight optimum |
| `ilco_adwords/experts.py` | greedy, MSVV, full-information, deterministic, aggregated, two-stage |
| `ilco_adwords/policy.py` | features, MLP training (PyTorch), NumPy inference |
| `ilco_adwords/rollout.py` | parallel episode rollouts with β-mixing |
| `ilco_adwords/cli.py` | `hindsight`, `expert` and `learn` commands |
| `slurm/` | job-array scripts and experiment grids |
| `analysis/make_results.py` | tables and figures of the paper |
| `scripts/calibrate.py` | calibration of the budget tightness |
| `tests/` | unit tests |

## Installation

Python 3.11+ and a Gurobi licence are required:

```bash
pip install -r requirements.txt
PYTHONPATH=$PWD pytest -q tests
```

## Reproducing the experiments

All commands are run from `adwords/`. Results go to `--out`, or to
`$ADWORDS_OUT` if `--out` is not given.

```bash
# Hindsight optimum on the validation and test episodes.
python -m ilco_adwords.cli --workers 32 hindsight
# One expert used online on the test episodes.
python -m ilco_adwords.cli --workers 32 expert --expert saa --scenarios 30
# One learning run.
python -m ilco_adwords.cli --workers 32 learn --expert saa --scenarios 5 \
    --seed 0 --mode dagger --episodes 2000 --per-iteration 10 --retrain-every 10
```

To run the full study on Rorqual (account `def-legraina_cpu`):

```bash
python slurm/make_grid.py
sbatch --array=1-14 --cpus-per-task=32 --time=6:00:00 \
    slurm/array.sh slurm/expert_grid.txt
sbatch --array=1-120 --cpus-per-task=32 --time=8:00:00 \
    slurm/array.sh slurm/learn_grid.txt
```

Then run `python analysis/make_results.py RESULTS_DIR PAPER_DIR`. It writes
`tables/adwords_*.tex` and `figs/*.pdf` into the paper folder.

## Experimental protocol

Learning from each expert:

- **Episodes:** 2,000 expert-labelled training episodes, about 149 queries
  each (≈298,000 labelled states).
- **Validation and test:** 300 validation and 1,000 test episodes, shared by
  all policies.
- **Seeds:** 5 per configuration. The seed changes the training episodes, the
  scenario samples and the network initialisation.
- **One-shot cloning:** the expert controls all trajectories. Policies are
  trained on the first 100, 250, 500, 1,000 and 2,000 episodes, and the best
  size on validation is kept.
- **DAgger:** 200 iterations of 10 episodes, with β₀ = 1 and βᵢ = 0.8ⁱ. The
  policy is retrained every 10 iterations (100 episodes), and the best
  retrained policy on validation is kept.
- **Scenarios per label:** 1 for the deterministic expert, K ∈ {5, 15, 30} for
  the aggregated and two-stage experts, and the realised future for the
  full-information expert. Learned policies use no scenarios at test time.
- **MIPs:** solved to optimality, except in the gap study of the 30-scenario
  two-stage expert (1, 2, 5 and 10%).
