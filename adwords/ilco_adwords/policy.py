"""State features and the learned (imitation) policy.

The policy is a multilayer perceptron mapping state features to logits over
the ``n_adv + 1`` actions; infeasible actions are masked. It is trained with a
(soft) cross-entropy loss against expert labels and evaluated with a fast
NumPy forward pass.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ilco_adwords.env import State, feasible_mask
from ilco_adwords.market import Market

NEG_INF = -1e9


def features(market: Market, state: State) -> np.ndarray:
    """Encode a state as a feature vector.

    Args:
        market: The market.
        state: Current state.

    Returns:
        Float32 feature vector.
    """
    one_hot = np.zeros(market.n_types, dtype=np.float32)
    one_hot[state.query] = 1.0
    bid = market.bids[:, state.query]
    mask = feasible_mask(market, state)[1:]
    return np.concatenate(
        [
            [state.time],
            one_hot,
            state.observed / market.count_mean,
            state.budgets / market.budgets,
            bid / market.bids.max(),
            mask,
        ]
    ).astype(np.float32)


class MLP(nn.Module):
    """Multilayer perceptron producing action logits.

    Args:
        n_in: Number of input features.
        n_out: Number of actions.
        hidden: Width of the hidden layers.
        depth: Number of hidden layers.
    """

    def __init__(
        self, n_in: int, n_out: int, hidden: int = 128, depth: int = 2
    ) -> None:
        """Build the network."""
        super().__init__()
        layers: list[nn.Module] = []
        width = n_in
        for _ in range(depth):
            layers += [nn.Linear(width, hidden), nn.ReLU()]
            width = hidden
        layers.append(nn.Linear(width, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return unmasked logits."""
        return self.net(x)


@dataclass
class NumpyPolicy:
    """Picklable NumPy copy of a trained MLP used for fast rollouts.

    Attributes:
        weights: List of (W, b) pairs.
    """

    weights: list[tuple[np.ndarray, np.ndarray]]

    @classmethod
    def from_torch(cls, model: MLP) -> NumpyPolicy:
        """Extract the weights of a torch MLP."""
        linear = [m for m in model.net if isinstance(m, nn.Linear)]
        return cls(
            [
                (m.weight.detach().numpy().T.copy(), m.bias.detach().numpy().copy())
                for m in linear
            ]
        )

    def logits(self, x: np.ndarray) -> np.ndarray:
        """Return the logits of a feature vector or matrix."""
        h = x
        for w, b in self.weights[:-1]:
            h = np.maximum(h @ w + b, 0.0)
        w, b = self.weights[-1]
        return h @ w + b

    def act(self, market: Market, state: State) -> int:
        """Return the most likely feasible action in a state."""
        out = self.logits(features(market, state))
        out = np.where(feasible_mask(market, state), out, NEG_INF)
        return int(np.argmax(out))


@dataclass
class TrainStats:
    """Summary of a training run.

    Attributes:
        n_samples: Number of training samples.
        epochs: Number of epochs actually run.
        val_loss: Best validation loss.
        val_acc: Validation accuracy of the best model.
        seconds: Training wall-clock time.
    """

    n_samples: int
    epochs: int
    val_loss: float
    val_acc: float
    seconds: float


def train_policy(
    x: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    seed: int = 0,
    hidden: int = 128,
    depth: int = 2,
    lr: float = 1e-3,
    batch_size: int = 512,
    max_epochs: int = 60,
    patience: int = 8,
) -> tuple[NumpyPolicy, TrainStats]:
    """Train an MLP policy by (soft) behavioural cloning.

    Args:
        x: Features, shape (N, n_features).
        target: Target action distributions, shape (N, n_actions).
        mask: Feasibility masks, shape (N, n_actions).
        seed: Random seed (data split and initialisation).
        hidden: Width of hidden layers.
        depth: Number of hidden layers.
        lr: Adam learning rate.
        batch_size: Mini-batch size.
        max_epochs: Maximum number of epochs.
        patience: Early-stopping patience in epochs.

    Returns:
        The best policy on the 10% hold-out set and training statistics.
    """
    start = time.perf_counter()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    perm = rng.permutation(n)
    n_val = max(1, n // 10)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    xt = torch.from_numpy(x.astype(np.float32))
    yt = torch.from_numpy(target.astype(np.float32))
    mt = torch.from_numpy(mask.astype(bool))

    model = MLP(x.shape[1], target.shape[1], hidden, depth)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def loss_fn(idx: torch.Tensor) -> torch.Tensor:
        logits = model(xt[idx]).masked_fill(~mt[idx], NEG_INF)
        logp = torch.log_softmax(logits, dim=1)
        return -(yt[idx] * logp).sum(dim=1).mean()

    best_loss = float("inf")
    best_state = None
    best_acc = 0.0
    wait = 0
    epoch = 0
    val_t = torch.from_numpy(val_idx)
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.from_numpy(rng.permutation(tr_idx))
        for b in range(0, order.numel(), batch_size):
            opt.zero_grad()
            loss = loss_fn(order[b : b + batch_size])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(val_t))
            logits = model(xt[val_t]).masked_fill(~mt[val_t], NEG_INF)
            acc = float((logits.argmax(1) == yt[val_t].argmax(1)).float().mean())
        if val_loss < best_loss - 1e-5:
            best_loss, best_acc, wait = val_loss, acc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    stats = TrainStats(n, epoch, best_loss, best_acc, time.perf_counter() - start)
    return NumpyPolicy.from_torch(model), stats
