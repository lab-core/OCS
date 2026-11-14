"""Imitation learning for online combinatorial optimisation under uncertainty.

The package is problem-independent. An application implements the
:class:`~udagger.problem.Problem`, :class:`~udagger.problem.Episode` and
:class:`~udagger.problem.Expert` interfaces; the package provides the
policy, the parallel rollouts with expert labelling, and the one-shot and
DAgger learning loops.
"""

from udagger.learner import LearnConfig, LearnResult, learn, mean_objective
from udagger.policy import MLP, NumpyPolicy, TrainStats, train_policy
from udagger.problem import Episode, Expert, Label, Problem, aggregate_votes, one_hot
from udagger.rollout import EpisodeResult, Runner, merge, run_episode, stack

__all__ = [
    "MLP",
    "Episode",
    "EpisodeResult",
    "Expert",
    "Label",
    "LearnConfig",
    "LearnResult",
    "NumpyPolicy",
    "Problem",
    "Runner",
    "TrainStats",
    "aggregate_votes",
    "learn",
    "mean_objective",
    "merge",
    "one_hot",
    "run_episode",
    "stack",
    "train_policy",
]
