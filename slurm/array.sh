#!/bin/bash
# Run line $SLURM_ARRAY_TASK_ID of a grid file with the command line of an
# example: sbatch [options] slurm/array.sh APP GRID, where APP is adwords or
# ppa and GRID a file written by APP/slurm/make_grid.py. Each line holds the
# options of `python -m udagger_APP.cli`; results go to $UDAGGER_OUT/APP.
#SBATCH --account=def-legraina_cpu
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2G
#SBATCH --job-name=udagger
set -euo pipefail
APP=$1
GRID=$2
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source slurm/env.sh
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${GRID}")
OUT=${UDAGGER_OUT:-$HOME/scratch/udagger}/${APP}
echo "task ${SLURM_ARRAY_TASK_ID}: ${LINE}"
# Global options go before the sub-command written in the grid line.
# UDAGGER_WORKERS overrides the number of worker processes (e.g. half the
# cores for PPA, whose MIPs use 2 threads each).
WORKERS=${UDAGGER_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}
python -m "udagger_${APP}.cli" --workers "${WORKERS}" --out "${OUT}" ${LINE}
