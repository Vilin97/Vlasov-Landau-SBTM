#!/bin/bash
#SBATCH --job-name=weibel
#SBATCH --account=naske
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=11:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/%x_%N_%A_%a.out

DVS=(2 3)
NS=(1000000)
CS=(0 0.0001 0.0002 0.0004 0.0008)
SCORE_METHODS=(blob sbtm)

tid=$SLURM_ARRAY_TASK_ID

# Each C value runs with both blob and sbtm
# Array indices 0-4: blob with C values 0, 0.0001, 0.0002, 0.0004, 0.0008
# Array indices 5-9: sbtm with C values 0, 0.0001, 0.0002, 0.0004, 0.0008
method_idx=$((tid / 5))
c_idx=$((tid % 5))

dv=${DVS[0]}
n=${NS[0]}
C=${CS[$c_idx]}
score_method=${SCORE_METHODS[$method_idx]}

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
