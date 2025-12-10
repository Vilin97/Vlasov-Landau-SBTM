#!/bin/bash
#SBATCH --job-name=two_stream
#SBATCH --account=amath
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/%x_%N_%A_%a.out

SCORE_METHODS=(sbtm kde)

n=1000000
dv=3
C=0.32
M=100
dt=0.01

tid=$SLURM_ARRAY_TASK_ID
score_method=${SCORE_METHODS[$tid]}

source /gscratch/amath/vilin/conda/etc/profile.d/conda.sh
conda activate vlsbtm

srun python3 experiments/vlasov-landau-two-stream.py \
  --n "${n}" \
  --M "${M}" \
  --dt "${dt}" \
  --dv "${dv}" \
  --C "${C}" \
  --score_method "${score_method}" \
  --wandb_project vlasov_landau_two_stream \
  --wandb_run_name "n${n}_M${M}_dt${dt}_dv${dv}_C${C}_${score_method}" \
  --sbtm_num_batch_steps 20 # reduce number of batch steps by 5x because dt is 5x smaller
