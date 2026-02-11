#!/bin/bash
#SBATCH --job-name=ld_sweep_n
#SBATCH --account=naske
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=9:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/%x_%N_%A_%a.out

N_VALUES=(500000 1000000 3000000)
SCORE_METHODS=(blob sbtm)

C=0.4
M=100
DT=0.02
DV=3
ALPHA=0.1
T=15.0

tid=$SLURM_ARRAY_TASK_ID

num_n=${#N_VALUES[@]}
num_sm=${#SCORE_METHODS[@]}

idx_n=$(( tid % num_n ))
idx_sm=$(( tid / num_n ))

N=${N_VALUES[$idx_n]}
score_method=${SCORE_METHODS[$idx_sm]}

source /gpfs/projects/mathai/vilin/miniconda3/etc/profile.d/conda.sh
conda activate vlsbtm

srun python3 experiments/vlasov-landau-damping.py \
  --n "$N" \
  --M "$M" \
  --dt "$DT" \
  --dv "$DV" \
  --C "$C" \
  --score_method "$score_method" \
  --wandb_project vlasov_landau_damping \
  --wandb_run_name "sweepN_alpha${ALPHA}_T${T}_n${N}_M${M}_dt${DT}_dv${DV}_C${C}_${score_method}" \
  --alpha "$ALPHA" \
  --final_time "$T"
