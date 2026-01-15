#!/bin/bash
#SBATCH --job-name=weibel_sweep_sbtm
#SBATCH --account=amath
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/%x_%N_%A_%a.out

DV=3
N=1000000
CS=(0.0002 0.0004)
SCORE_METHOD=sbtm

tid=$SLURM_ARRAY_TASK_ID
C=${CS[$tid]}

M=100
dt=0.1

source /gscratch/amath/vilin/conda/etc/profile.d/conda.sh
conda activate vlsbtm

srun python3 experiments/vlasov-landau-weibel.py \
    --n "$N" \
    --M "$M" \
    --dt "$dt" \
    --dv "$DV" \
    --C "$C" \
    --score_method "$SCORE_METHOD" \
    --wandb_project weibel \
    --wandb_run_name "n${N}_M${M}_dt${dt}_dv${DV}_C${C}_${SCORE_METHOD}"
