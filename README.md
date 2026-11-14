# Imitation learning for online combinatorial optimisation under uncertainty

This repository holds the code for two online assignment problems used to
study how the choice of the *expert* affects policies learned by imitation
learning (one-shot behavioural cloning and DAgger):

| Folder | Problem | Paper |
|---|---|---|
| [`ppa/`](ppa/README.md) | Dynamic physician-to-patient assignment | IJOO manuscript, CPAIOR 2027 submission |
| [`adwords/`](adwords/README.md) | Stochastic AdWords with late premium queries | CPAIOR 2027 submission |

Both studies compare the same expert families:

- myopic (greedy, MSVV);
- deterministic (one sampled future);
- full-information (the realised future);
- aggregated deterministic (K deterministic solutions turned into soft labels);
- two-stage stochastic (sample average approximation over K futures).

They are compared at several MIP gaps, with and without interaction (DAgger).

Each folder is self-contained, with its own requirements and cluster scripts.
