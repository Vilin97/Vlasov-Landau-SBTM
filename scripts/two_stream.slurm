#!/bin/bash
#SBATCH --job-name=two_stream
#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=96:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/%x_%N_%A_%a.out   # A=array job ID, a=array index

# Possible C values
# C_VALUES=(0 0.01 0.02 0.04 0.08)
C_VALUES=(0.16 0.32 0.64 1.28 2.56)
C=${C_VALUES[$SLURM_ARRAY_TASK_ID]}

source /gscratch/amath/vilin/conda/etc/profile.d/conda.sh
conda activate vlsbtm

srun python3 experiments/vlasov-landau-two-stream.py \
  --n 1_000_000 \
  --M 100 \
  --dt 0.05 \
  --gpu 0 \
  --dv 2 \
  --final_time 50.0 \
  --C $C \
  --alpha 0.005 \
  --k 0.2 \
  --c 2.4 \
  --score_method kde \
  --wandb_project vlasov_landau_two_stream \
  --wandb_run_name "n1e6_M100_dt0.05_C${C}_kde" \
  --wandb_mode online
