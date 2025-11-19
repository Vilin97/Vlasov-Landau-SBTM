#!/bin/bash
#SBATCH --job-name=landau_damping
#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x_%N_%A_%a.out   # A=array job ID, a=array index

# Collision strength C values
C_VALUES=(0 0.01 0.1 1.0)
C=${C_VALUES[$SLURM_ARRAY_TASK_ID]}

# sbtm_num_batch_steps values
# SBTM_STEPS_VALUES=(0 10 100 1000)
# SBTM_STEPS=${SBTM_STEPS_VALUES[$SLURM_ARRAY_TASK_ID]}

source /gscratch/amath/vilin/conda/etc/profile.d/conda.sh
conda activate vlsbtm

srun python3 experiments/vlasov-landau-damping.py \
  --n 1_000_000 \
  --M 100 \
  --dt 0.02 \
  --gpu 0 \
  --dv 2 \
  --final_time 15.0 \
  --C ${C} \
  --alpha 0.1 \
  --score_method sbtm \
  --sbtm_num_batch_steps 10 \
  --wandb_project vlasov_landau_damping \
  --wandb_run_name "n1e6_M100_dt0.02_C${C}_sbtm_steps10" \
  --wandb_mode online
