"""Checkpoints of a learning run, to resume it after an interruption.

A checkpoint is a folder with ``state.npz`` (labelled data and policy
weights) and ``state.json`` (iteration, learning curve, best policy row,
timers and the fingerprint of the run). Both files are written to temporary
names and renamed, ``state.json`` last; a random token stored in both makes
sure they belong to the same save, so a job killed while saving leaves the
previous checkpoint intact.

All random draws of a run are derived from the episode seeds and the
iteration number, so a resumed run gives the same results as an
uninterrupted one.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from udagger.policy import NumpyPolicy

_NPZ = "state.npz"
_JSON = "state.json"


def _policy_arrays(prefix: str, policy: NumpyPolicy | None) -> dict[str, np.ndarray]:
    if policy is None:
        return {}
    arrays = {}
    for i, (w, b) in enumerate(policy.weights):
        arrays[f"{prefix}/w{i}"] = w
        arrays[f"{prefix}/b{i}"] = b
    return arrays


def _policy_from(arrays: Any, prefix: str, sample: bool) -> NumpyPolicy | None:
    weights = []
    while f"{prefix}/w{len(weights)}" in arrays:
        i = len(weights)
        weights.append((arrays[f"{prefix}/w{i}"], arrays[f"{prefix}/b{i}"]))
    return NumpyPolicy(weights, sample) if weights else None


def save(
    folder: Path,
    fingerprint: dict,
    state: dict,
    data: dict[str, np.ndarray],
    policy: NumpyPolicy | None = None,
    best: NumpyPolicy | None = None,
) -> None:
    """Write a checkpoint atomically.

    Args:
        folder: Checkpoint folder (created if needed).
        fingerprint: Settings of the run; a resume requires the same values.
        state: JSON-serialisable progress (iteration, curve, timers, ...).
        data: Labelled samples collected so far.
        policy: Current policy (warm start, next DAgger iteration).
        best: Best policy on validation so far.
    """
    folder.mkdir(parents=True, exist_ok=True)
    token = secrets.token_hex(8)
    arrays = {f"data/{k}": v for k, v in data.items()}
    arrays |= _policy_arrays("policy", policy)
    arrays |= _policy_arrays("best", best)
    arrays["token"] = np.array(token)
    tmp_npz = folder / f"{_NPZ}.tmp.npz"
    np.savez(tmp_npz, **arrays)
    meta = {
        "token": token,
        "fingerprint": fingerprint,
        "state": state,
        "policy_sample": None if policy is None else policy.sample,
        "best_sample": None if best is None else best.sample,
    }
    tmp_json = folder / f"{_JSON}.tmp"
    tmp_json.write_text(json.dumps(meta))
    os.replace(tmp_npz, folder / _NPZ)
    os.replace(tmp_json, folder / _JSON)


def load(folder: Path, fingerprint: dict) -> dict | None:
    """Read a checkpoint written by :func:`save`, if there is a valid one.

    Returns:
        ``None`` if there is no complete checkpoint, else a dict with
        ``state``, ``data``, ``policy`` and ``best``.

    Raises:
        ValueError: If the checkpoint belongs to a run with other settings.
    """
    if not (folder / _JSON).exists() or not (folder / _NPZ).exists():
        return None
    meta = json.loads((folder / _JSON).read_text())
    if meta["fingerprint"] != fingerprint:
        changed = sorted(
            k
            for k in set(meta["fingerprint"]) | set(fingerprint)
            if meta["fingerprint"].get(k) != fingerprint.get(k)
        )
        raise ValueError(
            f"checkpoint {folder} is from another run (differs in {changed})"
        )
    with np.load(folder / _NPZ) as arrays:
        if str(arrays["token"]) != meta["token"]:
            return None  # interrupted while saving: files of two saves
        data = {
            k.split("/", 1)[1]: arrays[k] for k in arrays.files if k.startswith("data/")
        }
        policy = _policy_from(arrays, "policy", bool(meta["policy_sample"]))
        best = _policy_from(arrays, "best", bool(meta["best_sample"]))
    return {"state": meta["state"], "data": data, "policy": policy, "best": best}


def remove(folder: Path) -> None:
    """Delete a checkpoint folder (after a completed run)."""
    shutil.rmtree(folder, ignore_errors=True)
