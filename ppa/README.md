# Problem Description: Dynamic Physician-to-Patient Assignment (PPA)

## Overview
We consider a **dynamic physician-to-patient assignment (PPA)** problem arising in clinic appointment booking. Patients (call-ins) arrive sequentially over a booking horizon until a cutoff time for a particular session. The clinic is staffed by multiple physicians, and each incoming request must be handled immediately by a receptionist.

At each call-in, the receptionist must decide to **accept** or **reject** the request. If accepted, the receptionist must also decide **which physician** to assign to the patient among the patient’s eligible physicians. Decisions must be made online under uncertainty about future arrivals.
##scheduler_env
## Physicians
Let $\mathcal{P} = \{1, \dots, P\}$ denote the set of available physicians. Each physician $p \in \mathcal{P}$ has:
- a regular session time $T$ (in minutes),
- a maximum appointment capacity $L_p$.

## Patients (Call-ins)
Each patient request $n$ is characterized by:
- an eligible physician set $\mathcal{P}_n \subseteq \mathcal{P}$,
- a preferred physician $p_n \in \mathcal{P}_n$,
- an estimated service duration $d_n \in \mathbb{R}_{+}$ (known at call-in time),
- a priority class $r_n \in \{1,2\}$, where:
  - class 1: high priority,
  - class 2: regular priority.

## Costs / Penalties
If the receptionist rejects patient $n$, a rejection penalty is incurred:
$c^{\text{pref}}_{r_n}$.

If the receptionist accepts patient $n$ but assigns them to an eligible
non-preferred physician, a preference penalty is incurred:
$c^{\text{pref}}_{r_n}$, where $r_n \in \{1,2\}$.

## Key Challenge (Uncertainty and Trade-offs)
The number and composition of future requests are uncertain. This creates a trade-off between:
- accepting lower-priority patients now versus saving capacity for potentially higher-priority patients later,
- assigning a patient to a non-preferred physician (incurring a penalty) versus rejecting them (incurring a potentially larger penalty),
- preserving feasibility for future patients who may have limited eligibility, since poor early assignments can force future rejections.

## Decision Process View
This problem can be modeled as a **stochastic dynamic program (SDP)**:
- each decision epoch corresponds to an arriving patient call-in,
- the system state includes remaining physician capacities (and implicitly the remaining booking horizon),
- actions are to assign the patient to a physician $p \in \mathcal{P}_n$ or to reject the patient.

The objective is to choose decisions over the booking horizon to **minimize the total expected penalties** from patient rejections and non-preferred assignments, subject to physician capacity constraints.


## Layout

| Path | Content |
|---|---|
| `udagger_ppa/instance.py` | instance parameters, model construction and episode seeds |
| `udagger_ppa/scheduler.py` | simulator and deterministic MIP (Pyomo + Gurobi) |
| `udagger_ppa/stochastic.py` | two-stage stochastic MIP |
| `udagger_ppa/features.py` | state features of the learned policy |
| `udagger_ppa/experts.py` | greedy, full-information, deterministic, aggregated, two-stage |
| `udagger_ppa/problem.py` | the PPA problem and its episodes for `udagger`, hindsight optimum |
| `udagger_ppa/cli.py` | `hindsight`, `expert` and `learn` commands (learning with `udagger`) |
| `slurm/make_grid.py` | experiment grids (run with the shared `../slurm/array.sh`) |
| `tests/` | unit tests, including equivalence with the original code |
| `legacy/` | original scripts of the study (see below) |

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
`5001 + n` and `6001 + n`, as in the original evaluation.

```bash
# Hindsight optimum on the validation and test episodes.
python -m udagger_ppa.cli --workers 32 hindsight
# One expert used online on the test episodes.
python -m udagger_ppa.cli --workers 16 expert --expert saa --scenarios 30 --gap 0.002
# One learning run.
python -m udagger_ppa.cli --workers 10 learn --expert saa --scenarios 5 \
    --seed 0 --mode dagger
```

To run the full study on Rorqual, from the repository root with the shared
job script (results in `$UDAGGER_OUT/ppa`, by default `~/scratch/udagger/ppa`):

```bash
python ppa/slurm/make_grid.py
sbatch --array=1-14 --time=3-00:00:00 slurm/array.sh ppa ppa/slurm/expert_grid.txt
sbatch --array=1-57 --time=7-00:00:00 slurm/array.sh ppa ppa/slurm/learn_grid.txt
```

## Original scripts (`legacy/`)

`legacy/` keeps the scripts of the original study, which produced the
published results: `Learn_Main.py` (DAgger and data collection), `NN.py`
(network and training), `Learn_Sim.py` and `Evaluate.py` (evaluation),
`bash/` (original job scripts), and `slurm/` + `analysis/` (the CPAIOR rerun:
`slurm/run.sh`, `slurm/make_grid.py`, `analysis/make_results.py`). They
import the simulator and the MIPs from `udagger_ppa` through the
compatibility modules `legacy/OCS.py` and `legacy/SOCS.py`, and are run from
`legacy/`, e.g.:

```bash
cd ppa/legacy
python Learn_Main.py --runs 10 --new 1 --scenarios 5 --stoch 1 --um 1 \
    --learn_iter 400 --time_limit 30 --mipgap 0.02
python Learn_Sim.py --sims 1000 --model 399 --new 1 --scenarios 5 --stoch 1 \
    --um 1 --learn_iter 400 --runs 10 --time_limit 30 --mipgap 0.02
```

`tests/test_udagger_ppa.py` checks that the features and the expert decisions
of `udagger_ppa` are identical to those of the original `IO_learning` on the
same states and seeds.

## Citation

If you use our repository in your research, please cite the following paper:

> [Imitation Learning for Combinatorial Optimisation under Uncertainty](https://arxiv.org/abs/) - Prakash Gawas, Antoine Legrain, Louis-Martin Rousseau (2024)
