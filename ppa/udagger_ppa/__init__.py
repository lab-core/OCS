"""The dynamic physician-to-patient assignment problem, a ``udagger`` example.

Modules:

* ``instance``: instance parameters and episode seeds;
* ``scheduler``: simulator and deterministic MIP (from the original study);
* ``stochastic``: two-stage stochastic MIP (from the original study);
* ``features``: state features of the learned policy;
* ``experts``: greedy, full-information, deterministic, aggregated and
  two-stage experts;
* ``problem``: the problem and its episodes for ``udagger``;
* ``cli``: the ``hindsight``, ``expert`` and ``learn`` commands.
"""
