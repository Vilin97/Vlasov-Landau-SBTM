#!/bin/bash
#SBATCH --job-name=landau_damping
#SBATCH --account=naske
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=5-00:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/%x_%N_%A_%a.out

DV=3
M=100
N_VALUES=(1000000 3000000)        # 2
SCORE_METHODS=(blob)              # 1
C_VALUES=(0 0.1 0.4 1)            # 4

tid=$SLURM_ARRAY_TASK_ID

idx_c=$(( tid % 4 ))
rem=$(( tid / 4 ))
idx_sm=$(( rem % 1 ))
idx_n=$(( rem / 1 ))

C=${C_VALUES[$idx_c]}
score_method=${SCORE_METHODS[$idx_sm]}
n=${N_VALUES[$idx_n]}

source /gpfs/projects/mathai/vilin/miniconda3/etc/profile.d/conda.sh
conda activate vlsbtm

srun python3 experiments/vlasov-landau-damping.py \
  --n "$n" \
  --M "$M" \
  --dt 0.02 \
  --dv "$DV" \
  --C "$C" \
  --score_method "$score_method" \
  --wandb_project vlasov_landau_damping \
  --wandb_run_name "alpha0.5_T50_n${n}_M${M}_dt0.02_dv${DV}_C${C}_${score_method}" \
  --alpha 0.5 \
  --final_time 50.0
