#!/bin/bash
#SBATCH --job-name=weibel
#SBATCH --account=naske
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00
#SBATCH --array=0
#SBATCH --output=logs/%x_%N_%A_%a.out

DVS=(3)
NS=(3000000)
CS=(0.0016)   # 1 value, sbtm only

tid=$SLURM_ARRAY_TASK_ID

dv=${DVS[0]}
n=${NS[0]}
C=${CS[$tid]}
score_method=sbtm

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
