# Common environment for the AdWords experiments on an Alliance cluster.
# Submit jobs from the adwords/ directory; override IL4CO_VENV or ADWORDS_OUT
# to change the virtual environment or the result directory.
module load StdEnv/2023 python/3.11 gurobi/13.0.0
source "${IL4CO_VENV:-$HOME/il4co_env}/bin/activate"
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
# The repository root holds the udagger package.
export PYTHONPATH="$PWD/..:$PWD" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export ADWORDS_OUT="${ADWORDS_OUT:-$HOME/scratch/il4co/adwords}"
