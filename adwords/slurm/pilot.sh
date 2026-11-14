#!/bin/bash
#SBATCH --account=def-legraina_cpu
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2G
#SBATCH --job-name=adw-pilot
#SBATCH --output=%x-%j.out
# Short run to check timings before launching the full grid.
export ADWORDS_OUT="${ADWORDS_OUT:-$HOME/scratch/il4co/adwords_pilot}"
source slurm/env.sh
W=32
time python -m ilco_adwords.cli --workers $W hindsight
time python -m ilco_adwords.cli --workers $W --n-test 200 expert --expert saa --scenarios 30
time python -m ilco_adwords.cli --workers $W learn --expert full --mode oneshot --episodes 200 --sizes 50 100
time python -m ilco_adwords.cli --workers $W learn --expert saa --scenarios 5 --mode dagger --episodes 200 --retrain-every 5
