"""Checkpoints of a learning run, to resume it after an interruption.

A checkpoint folder holds:

* ``data_<i>.npz``: the samples labelled since the previous save (chunks are
  written once and never modified, so a save only writes new data);
* ``policies_<token>.npz``: the current and best policies of one save;
* ``state.json``: iteration, learning curve, timers, the fingerprint of the
  run, the chunk files and the policy file of the latest save.

A save writes its files first and replaces ``state.json`` last, in a single
atomic rename; files that the new ``state.json`` no longer references are
deleted afterwards. A job killed at any point therefore leaves a consistent
checkpoint: the previous one, or the new one.

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

import numpy as np

from udagger.policy import NumpyPolicy

_JSON = "state.json"


def _policy_arrays(prefix: str, policy: NumpyPolicy | None) -> dict[str, np.ndarray]:
    if policy is None:
        return {}
    arrays = {}
    for i, (w, b) in enumerate(policy.weights):
        arrays[f"{prefix}/w{i}"] = w
        arrays[f"{prefix}/b{i}"] = b
    return arrays


def _policy_from(
    arrays: np.lib.npyio.NpzFile, prefix: str, sample: bool
) -> NumpyPolicy | None:
    weights = []
    while f"{prefix}/w{len(weights)}" in arrays:
        i = len(weights)
        weights.append((arrays[f"{prefix}/w{i}"], arrays[f"{prefix}/b{i}"]))
    return NumpyPolicy(weights, sample) if weights else None


def _read_meta(folder: Path) -> dict | None:
    path = folder / _JSON
    return json.loads(path.read_text()) if path.exists() else None


def save(
    folder: Path,
    fingerprint: dict,
    state: dict,
    new_data: dict[str, np.ndarray],
    policy: NumpyPolicy | None = None,
    best: NumpyPolicy | None = None,
) -> None:
    """Add a checkpoint: the samples labelled since the last save and the policies.

    Args:
        folder: Checkpoint folder (created if needed).
        fingerprint: Settings of the run; a resume requires the same values.
        state: JSON-serialisable progress (iteration, curve, timers, ...).
        new_data: Samples labelled since the previous save (may be empty).
        policy: Current policy (warm start, next DAgger iteration).
        best: Best policy on validation so far.
    """
    folder.mkdir(parents=True, exist_ok=True)
    previous = _read_meta(folder)
    chunks = list(previous["chunks"]) if previous else []
    if new_data and len(next(iter(new_data.values()))) > 0:
        name = f"data_{len(chunks):05d}.npz"
        tmp = folder / f"{name}.tmp.npz"
        np.savez(tmp, **new_data)
        os.replace(tmp, folder / name)
        chunks.append(name)
    token = secrets.token_hex(8)
    policies = f"policies_{token}.npz"
    arrays = _policy_arrays("policy", policy) | _policy_arrays("best", best)
    tmp = folder / f"{policies}.tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, folder / policies)
    meta = {
        "fingerprint": fingerprint,
        "state": state,
        "chunks": chunks,
        "policies": policies,
        "policy_sample": None if policy is None else policy.sample,
        "best_sample": None if best is None else best.sample,
    }
    tmp_json = folder / f"{_JSON}.tmp"
    tmp_json.write_text(json.dumps(meta))
    os.replace(tmp_json, folder / _JSON)
    keep = {_JSON, policies, *chunks}
    for path in folder.iterdir():
        if path.name not in keep:
            path.unlink(missing_ok=True)


def load(folder: Path, fingerprint: dict) -> dict | None:
    """Read the checkpoint of a run, if there is one.

    Returns:
        ``None`` if there is no checkpoint, else a dict with ``state``,
        ``data`` (all samples saved so far), ``policy`` and ``best``.

    Raises:
        ValueError: If the checkpoint belongs to a run with other settings.
    """
    meta = _read_meta(folder)
    if meta is None:
        return None
    if meta["fingerprint"] != fingerprint:
        old = meta["fingerprint"]
        changed = sorted(
            k for k in set(old) | set(fingerprint) if old.get(k) != fingerprint.get(k)
        )
        raise ValueError(
            f"checkpoint {folder} is from another run (differs in {changed})"
        )
    parts: dict[str, list[np.ndarray]] = {}
    for name in meta["chunks"]:
        with np.load(folder / name) as chunk:
            for key in chunk.files:
                parts.setdefault(key, []).append(chunk[key])
    data = {k: np.concatenate(v) for k, v in parts.items()}
    with np.load(folder / meta["policies"]) as arrays:
        policy = _policy_from(arrays, "policy", bool(meta["policy_sample"]))
        best = _policy_from(arrays, "best", bool(meta["best_sample"]))
    return {"state": meta["state"], "data": data, "policy": policy, "best": best}


def remove(folder: Path) -> None:
    """Delete a checkpoint folder (after a completed run)."""
    shutil.rmtree(folder, ignore_errors=True)
