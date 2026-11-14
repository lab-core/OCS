# Environment of the PPA rerun: Python 3.11 and the same Gurobi as adwords/.
module load StdEnv/2023 python/3.11 gurobi/13.0.0
source "${PPA_VENV:-$HOME/ppa_env}/bin/activate"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
