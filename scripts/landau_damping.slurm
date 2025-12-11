#!/bin/bash
#SBATCH --job-name=landau_damping
#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=72:00:00
#SBATCH --array=0-23
#SBATCH --output=logs/%x_%N_%A_%a.out

DV=3
M=100
N_VALUES=(500000 1000000 3000000)        # 3
SCORE_METHODS=(sbtm blob)                # 2
C_VALUES=(0 0.1 0.4 1)                  # 4

tid=$SLURM_ARRAY_TASK_ID

idx_c=$(( tid % 4 ))
rem1=$(( tid / 4 ))
idx_sm=$(( rem1 % 2 ))
idx_n=$(( rem1 / 2 ))

C=${C_VALUES[$idx_c]}
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
  --wandb_run_name "alpha_0.5_T_50_n${n}_M${M}_dt0.02_dv${DV}_C${C}_${score_method}" \
  --alpha 0.5 \
  --final_time 50.0
