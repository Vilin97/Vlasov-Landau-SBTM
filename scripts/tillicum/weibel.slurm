#!/bin/bash
#SBATCH --job-name=weibel
#SBATCH --account=naske
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x_%N_%A_%a.out

DVS=(3)
NS=(3000000)
SCORE_METHODS=(blob sbtm)     # 2
CS=(0.0008 0.0016)            # 2

tid=$SLURM_ARRAY_TASK_ID

idx_sm=$(( tid % 2 ))
idx_c=$(( tid / 2 ))

dv=${DVS[0]}
n=${NS[0]}
score_method=${SCORE_METHODS[$idx_sm]}
C=${CS[$idx_c]}

M=100
dt=0.1

source /gpfs/projects/mathai/vilin/miniconda3/etc/profile.d/conda.sh
conda activate vlsbtm

srun python3 experiments/vlasov-landau-weibel.py \
  --n "$n" \
  --M "$M" \
  --dt "$dt" \
  --dv "$dv" \
  --C "$C" \
  --score_method "$score_method" \
  --wandb_project weibel \
  --wandb_run_name "n${n}_M${M}_dt${dt}_dv${dv}_C${C}_${score_method}"
