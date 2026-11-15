# Dynamic physician-to-patient assignment (PPA)

Appointment requests for a clinic session arrive one at a time until a
cut-off; each request must be assigned at once to an eligible physician or
rejected. This is the first case study of the paper on expert design for
imitation learning, and the problem of Gawas, Legrain and Rousseau (2024).

## Problem

- **Physicians:** P = 4 physicians, each with a regular session of 240
  minutes plus at most 60 minutes of overtime (300 minutes in total; overtime
  has no cost in the study) and at most 20 appointments.
- **Requests:** each request has a priority (1: urgent, 2: regular), a
  duration known at call time, a preferred physician and a set of eligible
  physicians.
- **Arrivals:** the numbers of priority-1 and priority-2 requests follow
  N(25, 5) and N(75, 12). The arrival order follows Beta-distributed scores,
  Beta(3, 1) for priority 1 and Beta(1, 1) for priority 2, so urgent patients
  tend to call late.
- **Costs:** rejecting a priority-1 (priority-2) request costs 200 (50).
  Assigning a request to an eligible physician other than the preferred one
  costs 10% of its rejection cost.

The difficulty is to keep capacity for the late urgent requests. The objective
is the total cost of an episode, which is minimised.

## Layout

| Path | Content |
|---|---|
| `udagger_ppa/instance.py` | instance parameters, episode seeds and model arguments |
| `udagger_ppa/simulator.py` | simulator (with the deterministic assignment MIP) |
| `udagger_ppa/solver.py` | two-stage stochastic MIP and hindsight optimum |
| `udagger_ppa/features.py` | state features of the learned policy |
| `udagger_ppa/experts.py` | greedy, full-information, deterministic, aggregated, two-stage |
| `udagger_ppa/problem.py` | the PPA problem and its episodes for `udagger` |
| `udagger_ppa/cli.py` | the application for the generic command line (`udagger.cli`) |
| `udagger_ppa/_scheduler.py`, `_stochastic.py` | simulator and MIP classes of the original study (Pyomo + Gurobi) |
| `slurm/make_grid.py` | experiment grids (run with the shared `../slurm/array.sh`) |
| `analysis/make_results.py` | tables and figures (with `udagger.analysis`) |
| `tests/` | unit tests |

## Installation

Python 3.11+ and a Gurobi licence are required. The learning code is the
`udagger` package at the repository root:

```bash
pip install -e ..            # udagger
pip install -r requirements.txt
PYTHONPATH=$PWD pytest -q tests
```

Without installing `udagger`, use `PYTHONPATH=$PWD/..:$PWD` instead. The
command line, options and output files are the same as in the AdWords example
(`../adwords`).

## Reproducing the experiments

All commands are run from `ppa/`. Results go to `--out`, or to `$PPA_OUT` if
`--out` is not given. Episode `n` of the test and validation sets has seed
`5001 + n` and `6001 + n`.

```bash
# Hindsight optimum on the validation and test episodes.
python -m udagger_ppa.cli --workers 32 hindsight
# One expert used online on the test episodes.
python -m udagger_ppa.cli --workers 16 expert --expert saa --scenarios 30 --gap 0.002
# One learning run.
python -m udagger_ppa.cli --workers 10 learn --expert saa --scenarios 5 \
    --seed 0 --mode dagger
```

To run the full study on a SLURM cluster, from the repository root with the
shared job script (results in `$UDAGGER_OUT/ppa`, by default
`~/scratch/udagger/ppa`; give your allocation with `--account`):

```bash
python ppa/slurm/make_grid.py
# MIPs use 2 threads: one worker per 2 cores.
export UDAGGER_WORKERS=16
sbatch --array=1-14 --time=3-00:00:00 --mem-per-cpu=4G slurm/array.sh ppa ppa/slurm/expert_grid.txt
sbatch --array=1-57 --time=7-00:00:00 --mem-per-cpu=4G slurm/array.sh ppa ppa/slurm/learn_grid.txt
```

Then run `python analysis/make_results.py RESULTS_DIR PAPER_DIR`.

## Original code

The scripts of the original study (`src/Learn_Main.py`, `src/NN.py`,
`src/Learn_Sim.py`, ...) are available at the git tag `ppa`. The simulator and
MIP classes are reused here unchanged. The features and the expert decisions
of `udagger_ppa` were checked to be identical to those of the original code
on the same states and seeds. The hindsight optima of the validation episodes
are identical as well (mean 1299.2).

## Citation

> P. Gawas, A. Legrain, L.-M. Rousseau. Imitation Learning for Combinatorial
> Optimisation under Uncertainty (2024).
