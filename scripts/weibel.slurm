#!/bin/bash
#SBATCH --job-name=weibel
#SBATCH --account=cse
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
#SBATCH --time=1:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x_%N_%A_%a.out

SCORE_METHODS=(blob sbtm)
C_VALUES=(0 0.0008)

n=100000
M=100
dv=3
dt=0.1

tid=$SLURM_ARRAY_TASK_ID

idx_score=$(( tid % 2 ))
idx_C=$(( tid / 2 ))

score_method=${SCORE_METHODS[$idx_score]}
C=${C_VALUES[$idx_C]}

source /gscratch/amath/vilin/conda/etc/profile.d/conda.sh
conda activate vlsbtm

srun python3 experiments/vlasov-landau-weibel.py \
  --n "${n}" \
  --M "${M}" \
  --dt "${dt}" \
  --dv "${dv}" \
  --C "${C}" \
  --score_method "${score_method}" \
  --wandb_project weibel \
  --wandb_run_name "n${n}_M${M}_dt${dt}_dv${dv}_C${C}_${score_method}" \
