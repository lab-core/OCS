"""The physician-to-patient assignment problem as an ``ildagger`` example.

The simulator, the state features and the optimisation experts are those of
the original study (``ppa/src``), used unchanged; this package only adapts
them to the ``ildagger`` interfaces. The original modules use flat imports
(``from OCS import ...``), so ``ppa/src`` is added to the import path.
"""

import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
