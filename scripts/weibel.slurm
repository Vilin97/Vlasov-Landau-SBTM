#!/bin/bash
#SBATCH --job-name=weibel_sweep
#SBATCH --account=amath
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x_%N_%A_%a.out

DVS=(2 3)                    # 2
NS=(500000 1000000)          # 2
C=0.0001                     # fixed
SCORE_METHOD=sbtm

tid=$SLURM_ARRAY_TASK_ID

idx_dv=$(( tid % 2 ))
idx_n=$(( tid / 2 ))

dv=${DVS[$idx_dv]}
n=${NS[$idx_n]}

M=100
dt=0.1

source /gscratch/amath/vilin/conda/etc/profile.d/conda.sh
conda activate vlsbtm

srun python3 experiments/vlasov-landau-weibel.py \
  --n "$n" \
  --M "$M" \
  --dt "$dt" \
  --dv "$dv" \
  --C "$C" \
  --score_method "$SCORE_METHOD" \
  --wandb_project weibel \
  --wandb_run_name "n${n}_dv${dv}_C${C}_${SCORE_METHOD}"
