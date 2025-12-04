#!/bin/bash
#SBATCH --job-name=landau_damping
#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=110:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/%x_%N_%A_%a.out

DV=3
C=0.4
M=100
N_VALUES=(500000 2000000 3000000)
SCORE_METHODS=(sbtm kde)

tid=$SLURM_ARRAY_TASK_ID

idx_sm=$(( tid % 2 ))
idx_n=$(( tid / 2 ))

score_method=${SCORE_METHODS[$idx_sm]}
n=${N_VALUES[$idx_n]}

source /gscratch/amath/vilin/conda/etc/profile.d/conda.sh
conda activate vlsbtm

srun python3 experiments/vlasov-landau-damping.py \
  --n "${n}" \
  --M "${M}" \
  --dt 0.02 \
  --dv "${DV}" \
  --C "${C}" \
  --score_method "${score_method}" \
  --wandb_project vlasov_landau_damping \
  --wandb_run_name "n${n}_M${M}_dt0.02_dv${DV}_C${C}_${score_method}"
