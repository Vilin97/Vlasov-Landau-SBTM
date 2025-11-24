#!/bin/bash
#SBATCH --job-name=two_stream
#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=24:00:00
#SBATCH --array=0-0
#SBATCH --output=logs/%x_%N_%A_%a.out

DVS=(2 3)
SCORE_METHODS=(kde sbtm)
C_VALUES=(0 0.02 0.08 0.32)

tid=$SLURM_ARRAY_TASK_ID
idx_C=$(( tid % 4 ))
idx_sm=$(( (tid / 4) % 2 ))
idx_dv=$(( tid / 8 ))

dv=${DVS[$idx_dv]}
score_method=${SCORE_METHODS[$idx_sm]}
C=${C_VALUES[$idx_C]}

source /gscratch/amath/vilin/conda/etc/profile.d/conda.sh
conda activate vlsbtm

srun python3 experiments/vlasov-landau-two-stream.py \
  --n 1_000_000 \
  --M 100 \
  --dt 0.05 \
  --dv "${dv}" \
  --C "${C}" \
  --score_method "${score_method}" \
  --wandb_project vlasov_landau_two_stream \
  --wandb_run_name "n1e6_M100_dt0.05_dv${dv}_C${C}_${score_method}"
