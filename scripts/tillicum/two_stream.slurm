#!/bin/bash
#SBATCH --job-name=two_stream
#SBATCH --account=naske
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=7:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/%x_%N_%A_%a.out

# 5 jobs: 500k/blob, 500k/sbtm, 1M/blob, 1M/sbtm, 3M/sbtm
NS=(500000 500000 1000000 1000000 3000000)
SCORE_METHODS=(blob sbtm blob sbtm sbtm)

C=0.24
M=100

tid=$SLURM_ARRAY_TASK_ID

dv=3
score_method=${SCORE_METHODS[$tid]}
n=${NS[$tid]}

source /gpfs/projects/mathai/vilin/miniconda3/etc/profile.d/conda.sh
conda activate vlsbtm

srun python3 experiments/vlasov-landau-two-stream.py \
  --n "$n" \
  --M "$M" \
  --dt 0.05 \
  --dv "$dv" \
  --C "$C" \
  --score_method "$score_method" \
  --wandb_project vlasov_landau_two_stream \
  --wandb_run_name "n${n}_M${M}_dt0.05_dv${dv}_C${C}_${score_method}"
