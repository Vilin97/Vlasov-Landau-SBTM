#!/bin/bash
#SBATCH --job-name=landau_damping
#SBATCH --account=naske
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=6:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x_%N_%A_%a.out

DV_VALUES=(3)                 # 2
SCORE_METHODS=(blob)       # 2
C_VALUES=(0.01 0.05 0.1)      # 4
N_VALUES=(1000000)             # 1

M=100
DT=0.01
ALPHA=0.5
T=50.0

tid=$SLURM_ARRAY_TASK_ID

num_c=${#C_VALUES[@]}
num_sm=${#SCORE_METHODS[@]}
num_dv=${#DV_VALUES[@]}
num_n=${#N_VALUES[@]}

# tid in [0, num_c*num_sm*num_dv*num_n - 1]
idx_c=$(( tid % num_c ))
rem=$(( tid / num_c ))

idx_sm=$(( rem % num_sm ))
rem=$(( rem / num_sm ))

idx_dv=$(( rem % num_dv ))
idx_n=$(( rem / num_dv ))

C=${C_VALUES[$idx_c]}
score_method=${SCORE_METHODS[$idx_sm]}
DV=${DV_VALUES[$idx_dv]}
n=${N_VALUES[$idx_n]}

source /gpfs/projects/mathai/vilin/miniconda3/etc/profile.d/conda.sh
conda activate vlsbtm

srun python3 experiments/vlasov-landau-damping.py \
  --n "$n" \
  --M "$M" \
  --dt "$DT" \
  --dv "$DV" \
  --C "$C" \
  --score_method "$score_method" \
  --wandb_project vlasov_landau_damping \
  --wandb_run_name "alpha${ALPHA}_T${T}_n${n}_M${M}_dt${DT}_dv${DV}_C${C}_${score_method}" \
  --alpha "$ALPHA" \
  --final_time "$T"
