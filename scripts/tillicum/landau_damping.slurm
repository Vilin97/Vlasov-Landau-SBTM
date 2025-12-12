#!/bin/bash
#SBATCH --job-name=landau_damping
#SBATCH --account=naske
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --array=0-0
#SBATCH --output=logs/%x_%N_%A_%a.out

DV=3
M=100
ALPHA=0.1
FINAL_TIME=15.0

N_VALUES=(3000000)
SCORE_METHODS=(blob)
C_VALUES=(0.4)

tid=$SLURM_ARRAY_TASK_ID

idx_c=0
rem=0
idx_sm=0
idx_n=0

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
  --wandb_run_name "alpha${ALPHA}_T${FINAL_TIME}_n${n}_M${M}_dt0.02_dv${DV}_C${C}_${score_method}" \
  --alpha "$ALPHA" \
  --final_time "$FINAL_TIME"