"""Learned (imitation) policy.

The policy is a multilayer perceptron mapping state features to logits over
the actions; infeasible actions are masked. It is trained with a (soft)
cross-entropy loss against expert labels and evaluated with a fast NumPy
forward pass, so that it can be shipped to worker processes cheaply.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


NEG_INF = -1e9


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
        sample: Whether ``act`` samples from the predicted distribution (as in
            the physician study) instead of taking its mode.
    """

    weights: list[tuple[np.ndarray, np.ndarray]]
    sample: bool = False

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

    def act(
        self,
        x: np.ndarray,
        mask: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> int:
        """Return a feasible action.

        Args:
            x: Feature vector of the state.
            mask: Boolean feasibility mask over actions.
            rng: Generator used when ``sample`` is True.

        Returns:
            The sampled action if ``sample`` is True, else the most likely one.

        Raises:
            ValueError: If sampling is requested without a generator.
        """
        out = np.where(mask, self.logits(x), NEG_INF)
        if not self.sample:
            return int(np.argmax(out))
        if rng is None:
            raise ValueError("a sampling policy needs a random generator")
        prob = np.exp(out - out.max())
        return int(rng.choice(prob.size, p=prob / prob.sum()))

    def with_sampling(self, sample: bool) -> NumpyPolicy:
        """Return a copy sharing the weights with another action rule."""
        return NumpyPolicy(self.weights, sample)


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
    batch_size: int = 256,
    max_epochs: int = 100,
    patience: int | None = None,
    val_fraction: float = 0.2,
    init: NumpyPolicy | None = None,
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
        patience: Early-stopping patience in epochs (None trains for
            ``max_epochs`` epochs, as in the physician study).
        val_fraction: Fraction of the samples held out to select the epoch.
        init: Policy whose weights initialise the network (warm start).

    Returns:
        The policy with the best hold-out loss and training statistics.
    """
    start = time.perf_counter()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_fraction))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    xt = torch.from_numpy(x.astype(np.float32))
    yt = torch.from_numpy(target.astype(np.float32))
    mt = torch.from_numpy(mask.astype(bool))

    model = MLP(x.shape[1], target.shape[1], hidden, depth)
    if init is not None:
        linear = [m for m in model.net if isinstance(m, nn.Linear)]
        with torch.no_grad():
            for m, (w, b) in zip(linear, init.weights, strict=True):
                m.weight.copy_(torch.from_numpy(w.T))
                m.bias.copy_(torch.from_numpy(b))
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
            if patience is not None and wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    stats = TrainStats(n, epoch, best_loss, best_acc, time.perf_counter() - start)
    return NumpyPolicy.from_torch(model), stats
