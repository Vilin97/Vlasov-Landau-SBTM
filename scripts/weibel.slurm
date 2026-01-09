#!/bin/bash
#SBATCH --job-name=weibel_sweep_sbtm
#SBATCH --account=cse
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-19
#SBATCH --output=logs/%x_%N_%A_%a.out

DVS=(2 3)                                   # 2
NS=(500000 1000000)                         # 2
CS=(0.0001 0.0002 0.0004 0.0008 0.0016)     # 5 (no C=0)
SCORE_METHOD=sbtm

tid=$SLURM_ARRAY_TASK_ID

combo=$(( tid % 4 ))        # 0..3 cycles (dv,n) first
idx_c=$(( tid / 4 ))        # 0..4 (C outer)

idx_n=$(( combo % 2 ))      # 0..1
idx_dv=$(( combo / 2 ))     # 0..1

dv=${DVS[$idx_dv]}
n=${NS[$idx_n]}
C=${CS[$idx_c]}

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
  --wandb_run_name "n${n}_M${M}_dt${dt}_dv${dv}_C${C}_${SCORE_METHOD}"
