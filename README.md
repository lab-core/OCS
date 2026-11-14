# Imitation learning for online combinatorial optimisation under uncertainty

This repository holds `udagger`, a problem-independent package for learning
online policies by imitating optimisation experts (one-shot behavioural
cloning and DAgger). It also holds two applications that use it, which study
how the choice of the *expert* affects the learned policy.

| Folder | Content |
|---|---|
| [`udagger/`](udagger) | The package: problem interfaces, policy, rollouts with expert labelling, learning loops, shared command-line pieces |
| [`adwords/`](adwords/README.md) | Example 1: stochastic AdWords with late premium queries (CPAIOR 2027) |
| [`ppa/`](ppa/README.md) | Example 2: dynamic physician-to-patient assignment (IJOO, CPAIOR 2027) |
| [`slurm/`](slurm) | Job scripts shared by both examples (`array.sh APP GRID`) |
| [`tests/`](tests) | Tests of the package on a toy problem |

Both examples have the same structure: a package `udagger_<app>/` with the
instance, simulator, MIP solver, `features.py`, `experts.py`, `problem.py`
(the `udagger` interfaces) and `cli.py` (sub-commands `hindsight`, `expert`
and `learn` with the same options and output files), plus `slurm/make_grid.py`
and `tests/`. `ppa/legacy/` also keeps the original scripts of the
physician study.

## The `udagger` package

An application implements three small interfaces (`udagger/problem.py`):

- `Problem`: number of actions, objective sense, and `episode(seed)`;
- `Episode`: `observe`, `features`, `mask` (feasible actions), `step`, `info`
  (extra values stored with the labels) and `stats` (with `"objective"`);
- `Expert`: `expert(episode, state, rng) -> Label(action, dist, solve_time, gap)`.
  The expert receives the episode, so a full-information expert can read the
  realised future. Experts that solve several models (aggregated experts) turn
  their votes into a soft label with `aggregate_votes`.

The package provides the rest:

| Module | Provides |
|---|---|
| `policy.py` | masked MLP, NumPy policy (sampled or most likely action), training with soft cross-entropy and warm start |
| `rollout.py` | episodes mixing expert and policy with probability β, expert labelling, parallel `Runner` |
| `learner.py` | `learn(runner, expert, LearnConfig(...), train_seeds, valid_seeds)` for one-shot cloning and DAgger, with selection of the best policy on validation |
| `experiment.py` | the `learn` and `expert` commands shared by the examples (options, test of both action rules, output files) |

```python
from udagger import LearnConfig, Runner, learn

runner = Runner(MyProblem(), workers=8)
result = learn(
    runner,
    MyExpert(),
    LearnConfig(mode="dagger"),
    train_seeds=list(range(4000)),
    valid_seeds=list(range(10**6, 10**6 + 1000)),
)
test = runner.run(test_seeds, policy=result.policy)
```

The experts compared in both applications follow the same taxonomy:

- myopic (greedy, MSVV);
- deterministic (one sampled future);
- full-information (the realised future);
- aggregated deterministic (K deterministic solutions turned into soft labels);
- two-stage stochastic (sample average approximation over K futures).

They are compared at several MIP gaps, with and without interaction (DAgger).

## Installation and tests

```bash
pip install -e ".[test]"                  # the package
pip install -r adwords/requirements.txt   # example 1
pip install -r ppa/requirements.txt       # example 2 (Pyomo, Gurobi)
PYTHONPATH=$PWD/adwords:$PWD/ppa pytest tests adwords/tests ppa/tests
```

The examples import `udagger` from the repository root. Without installing
the package, add the root to the path as well: `PYTHONPATH=$PWD:...`.
