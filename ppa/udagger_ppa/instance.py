"""Instance parameters, episode seeds and arguments of the PPA models.

The defaults are those of the original study (``src/utility.py``, tag ``ppa``): 4
physicians with 20 appointments, N(25, 5) priority-1 and N(75, 12)
priority-2 requests, rejection costs 200 and 50, and a non-preferred
assignment penalty of 10% of the rejection cost.
"""

from __future__ import annotations

import atexit
import shutil
import tempfile
from dataclasses import dataclass

# Episode seeds, as in the original evaluation: test 5001 + n, validation
# 6001 + n; training episodes of replicate r are 10**6 * (r + 1) + i.
TEST_SEED = 5001
VALID_SEED = 6001
SPLIT_SIZE = VALID_SEED - TEST_SEED

_TMP: list[str] = []


@dataclass(frozen=True)
class Instance:
    """Parameters of the clinic and of the request distribution.

    Attributes:
        n1: Mean number of priority-1 requests.
        sigma1: Standard deviation of the number of priority-1 requests.
        n2: Mean number of priority-2 requests.
        sigma2: Standard deviation of the number of priority-2 requests.
        n_phys: Number of physicians.
        n_slots: Number of slots (session length = slots x 30 minutes).
        cap_per_phys: Maximum number of appointments per physician.
        var: High-variance duration setting of the original generator.
        k_min: Minimum number of eligible physicians per patient.
        k_max: Maximum number of eligible physicians per patient.
        physician_weights: Draw eligibility from Dirichlet preferences.
        c_miss_1: Rejection cost of a priority-1 patient.
        c_miss_2: Rejection cost of a priority-2 patient.
        c_overtime: Overtime cost per minute.
        c_np_per: Non-preferred assignment penalty (% of rejection cost).
        release_cap: Maximum number of accepted patients (None: no cap).
    """

    n1: int = 25
    sigma1: float = 5.0
    n2: int = 75
    sigma2: float = 12.0
    n_phys: int = 4
    n_slots: int = 8
    cap_per_phys: int = 20
    var: int = 1
    k_min: int = 1
    k_max: int = 4
    physician_weights: int = 1
    c_miss_1: int = 200
    c_miss_2: int = 50
    c_overtime: int = 0
    c_np_per: int = 10
    release_cap: int | None = None

    @property
    def n_actions(self) -> int:
        """One action per physician plus rejection (the last action)."""
        return self.n_phys + 1


def _tmp_dir() -> str:
    """Folder for the files the original classes write (one per process)."""
    if not _TMP:
        _TMP.append(tempfile.mkdtemp(prefix="ppa_"))
        atexit.register(shutil.rmtree, _TMP[0], ignore_errors=True)
    return _TMP[0]


def model_kwargs(inst: Instance, seed: int, time_limit: float, mipgap: float) -> dict:
    """Constructor arguments of the simulator and of the MIP models."""
    return {
        "n_patients1": inst.n1,
        "sigma1": inst.sigma1,
        "n_patients2": inst.n2,
        "sigma2": inst.sigma2,
        "n_phys": inst.n_phys,
        "n_slots": inst.n_slots,
        "k_max": inst.k_max,
        "k_min": inst.k_min,
        "c_miss_1": inst.c_miss_1,
        "c_miss_2": inst.c_miss_2,
        "c_overtime": inst.c_overtime,
        "c_np_per": inst.c_np_per,
        "physician_weights": inst.physician_weights,
        "release_cap": inst.release_cap,
        "cap_per_phys": inst.cap_per_phys,
        "time_limit": time_limit,
        "mipgap": mipgap,
        "var": inst.var,
        "seed": seed,
        "out_dir": _tmp_dir(),
        "suffix": False,
    }


def split_seeds(split: str, n: int, run_seed: int = 0) -> list[int]:
    """Return the seeds of the first ``n`` episodes of a split.

    Raises:
        ValueError: If the split is unknown or test and validation episodes
            would overlap.
    """
    if split == "train":
        return [10**6 * (run_seed + 1) + i for i in range(n)]
    if split not in ("test", "valid"):
        raise ValueError(f"unknown split {split!r}")
    if n > SPLIT_SIZE:
        raise ValueError(f"at most {SPLIT_SIZE} test or validation episodes")
    base = TEST_SEED if split == "test" else VALID_SEED
    return [base + i for i in range(n)]
