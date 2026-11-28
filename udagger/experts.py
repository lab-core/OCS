"""Settings shared by the experts of the applications.

Both examples describe an expert by its treatment of uncertainty (``kind``),
the number of scenarios, the stopping criteria of its MIPs and how it acts
when several models vote (aggregated experts). An application subclasses
:class:`ExpertSpec`, sets :attr:`ExpertSpec.KINDS` and implements
``__call__`` (the :class:`udagger.problem.Expert` interface).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

# Kinds of experts that solve several scenarios.
MULTI_SCENARIO = ("agg", "saa")


@dataclass
class ExpertSpec:
    """Configuration of an expert.

    Attributes:
        kind: Treatment of uncertainty, one of :attr:`KINDS` (for instance
            ``greedy``, ``full``, ``det``, ``agg`` or ``saa``).
        n_scenarios: Number of scenarios of ``agg`` and ``saa`` (1 otherwise).
        mip_gap: Relative MIP gap stopping criterion.
        time_limit: Time limit per MIP in seconds.
        sample_vote: Draw the executed action from the votes of the solved
            models (as in the physician study) instead of the majority.
    """

    KINDS: ClassVar[tuple[str, ...]] = ()

    kind: str
    n_scenarios: int = 1
    mip_gap: float = 0.02
    time_limit: float = 30.0
    sample_vote: bool = True

    def __post_init__(self) -> None:
        """Validate the kind; single-scenario kinds use one scenario."""
        if self.kind not in self.KINDS:
            raise ValueError(f"unknown expert {self.kind!r}, expected {self.KINDS}")
        if self.kind not in MULTI_SCENARIO:
            self.n_scenarios = 1

    @property
    def name(self) -> str:
        """Short identifier used in file names and tables (e.g. ``saa30``)."""
        if self.kind in MULTI_SCENARIO:
            return f"{self.kind}{self.n_scenarios}"
        return self.kind
