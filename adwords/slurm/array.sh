#!/bin/bash
# Run line $SLURM_ARRAY_TASK_ID (1-based) of the grid file given as $1.
#SBATCH --account=def-legraina_cpu
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2G
#SBATCH --job-name=adw
source slurm/env.sh
ARGS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$1")
echo "task ${SLURM_ARRAY_TASK_ID}: ${ARGS}"
python -m ilco_adwords.cli --workers "${SLURM_CPUS_PER_TASK}" ${ARGS}
