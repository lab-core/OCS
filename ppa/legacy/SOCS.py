"""Compatibility module: the code moved to ``udagger_ppa.stochastic``.

The original scripts of this folder import ``SOCS``; the package folder
(``ppa/``) is added to the import path so that they keep working.
"""

import sys
from pathlib import Path

_PPA = str(Path(__file__).resolve().parents[1])
if _PPA not in sys.path:
    sys.path.insert(0, _PPA)

# Re-export the moved module; the import path above must be set first.
from udagger_ppa.stochastic import *  # noqa: E402,F401,F403
from udagger_ppa.stochastic import SAppointmentScheduler  # noqa: E402,F401
