#!/bin/bash
#SBATCH --job-name=weibel_sweep
#SBATCH --account=amath
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-47
#SBATCH --output=logs/%x_%N_%A_%a.out

DVS=(2 3)                                   # 2
NS=(500000 1000000)                          # 2
SCORE_METHODS=(blob sbtm)                    # 2
CS=(0 0.0001 0.0002 0.0004 0.0008 0.0016)    # 6

tid=$SLURM_ARRAY_TASK_ID

idx_c=$(( tid % 6 ))
rem=$(( tid / 6 ))
idx_sm=$(( rem % 2 ))
rem=$(( rem / 2 ))
idx_n=$(( rem % 2 ))
idx_dv=$(( rem / 2 ))

dv=${DVS[$idx_dv]}
n=${NS[$idx_n]}
score_method=${SCORE_METHODS[$idx_sm]}
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
  --score_method "$score_method" \
  --wandb_project weibel \
  --wandb_run_name "n${n}_M${M}_dt${dt}_dv${dv}_C${C}_${score_method}"
