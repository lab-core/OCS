#!/bin/bash
# Run line $SLURM_ARRAY_TASK_ID of the grid file $1 (submit from ppa/slurm/).
# A line sets: kind (learn|expert|greedy|hindsight), mode (oneshot|dagger),
# expert (expert options), mg (MIP gap) and rep (replicate).
#SBATCH --account=def-legraina_cpu
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=ppa
set -euo pipefail
source env.sh
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$1")
echo "task ${SLURM_ARRAY_TASK_ID}: ${LINE}"
kind=learn mode=dagger expert="" mg=0.02 rep=0 tl=30
eval "${LINE}"
CPUS=${SLURM_CPUS_PER_TASK:-2}
cd ../src
# Instance and learning options of the IJOO study (bash/batch_learn.sh).
COMMON="--N1 25 --s1 5 --N2 75 --s2 12 --P 4 --I 8 --L 20 --adapter flat \
--var 1 --kmin 1 --kmax 4 --pw 1 --cnp 10 --co 0 --use_weight 0 --vanilla 1 --gated 0 --evolve_expert 0 --time_limit ${tl} --mipgap ${mg}"
OUT=../data/evaluation

if [ "$kind" != learn ]; then
  tag=${kind}_$(echo "${expert}" | tr -d ' -')_mg${mg}
  python Evaluate.py --target "$kind" --split test --workers $((CPUS / 2)) \
    --out "${OUT}/${tag}.csv" ${COMMON} ${expert}
  exit 0
fi

seed=$((1000000 * (rep + 1)))
# 400 iterations of 10 episodes in both modes. One-shot keeps the expert in
# control (lambda = 1, so beta stays 1) and trains once at the end (um = 400),
# instead of the original single iteration of 4,000 parallel episodes, which
# builds all 4,000 simulators and their scenario models at once.
li=400 runs=10
if [ "$mode" = oneshot ]; then lm=1 um=400; else lm=0.8 um=1; fi
python Learn_Main.py ${COMMON} ${expert} --lm ${lm} --um ${um} \
  --learn_iter ${li} --runs ${runs} --resume 0 --seed ${seed} --seed_in_name 1 \
  --workers ${runs}

dir=$(ls -d ../data/Data_Dagger_${lm}*_flat_0/Instances_*/NN_iter${li}_runs${runs}_um${um}_*_mg${mg}_nc5_seed${seed}/ | \
  grep -F -- "$(python - ${expert} <<'PY'
import sys
from utility import build_config, parse_args
c = build_config(parse_args())
print(f"_new{c.new}_stoch{c.stoch}_sc{c.scenarios}_")
PY
)")
echo "folder: ${dir}"
# One-shot trains a single model, saved at the last iteration.
best=$((li - 1))
if [ "$mode" = dagger ]; then
  python Evaluate.py --target policy --split valid --workers "${CPUS}" \
    --policy_dir "${dir}" --models $(seq 0 $((li - 1))) \
    --out "${dir}eval/valid_random.csv" ${COMMON} ${expert}
  best=$(python -c "import pandas as pd; d=pd.read_csv('${dir}eval/valid_random.csv'); print(d.groupby('model').total_cost.mean().idxmin())")
fi
echo "best model: ${best}"
for rule in random max; do
  python Evaluate.py --target policy --split test --workers "${CPUS}" \
    --policy_dir "${dir}" --models "${best}" --action_type "${rule}" \
    --out "${dir}eval/test_${rule}.csv" ${COMMON} ${expert}
done
