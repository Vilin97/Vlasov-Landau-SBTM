#!/bin/bash
#SBATCH --job-name=two_stream
#SBATCH --account=naske
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --array=0-0
#SBATCH --output=logs/%x_%N_%A_%a.out

DVS=(3)
SCORE_METHODS=(kde)
NS=(3000000)

C=0.32
M=100

tid=$SLURM_ARRAY_TASK_ID

dv=${DVS[0]}
score_method=${SCORE_METHODS[0]}
n=${NS[0]}

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
