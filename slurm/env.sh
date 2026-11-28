# Environment of the experiments on an Alliance cluster (Rorqual).
# One virtual environment serves both examples: numpy, scipy, pandas, torch,
# gurobipy, pyomo and matplotlib (see README.md). Override UDAGGER_VENV to
# use another one.
module load StdEnv/2023 python/3.11 gurobi/13.0.0
source "${UDAGGER_VENV:-$HOME/udagger_env}/bin/activate"
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$ROOT:$ROOT/adwords:$ROOT/ppa" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
