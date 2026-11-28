"""State features of the AdWords policy."""

from __future__ import annotations

import numpy as np

from udagger_adwords.simulator import State, feasible_mask
from udagger_adwords.instance import Market


def features(
    market: Market, state: State, mask: np.ndarray | None = None
) -> np.ndarray:
    """Encode a state as a feature vector.

    Args:
        market: The market.
        state: Current state.
        mask: Feasibility mask of the state, if already computed.

    Returns:
        Float32 feature vector: time, query type (one-hot), queries observed
        per class (relative to the mean), remaining budgets (relative),
        bids on the query (relative) and feasibility of each advertiser.
    """
    query = np.zeros(market.n_types, dtype=np.float32)
    query[state.query] = 1.0
    bid = market.bids[:, state.query]
    if mask is None:
        mask = feasible_mask(market, state)
    mask = mask[1:]
    return np.concatenate(
        [
            [state.time],
            query,
            state.observed / market.count_mean,
            state.budgets / market.budgets,
            bid / market.bids.max(),
            mask,
        ]
    ).astype(np.float32)
