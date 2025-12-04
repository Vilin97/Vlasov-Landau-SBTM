#!/bin/bash
#SBATCH --job-name=two_stream
#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=110:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/%x_%N_%A_%a.out

DVS=(3)
SCORE_METHODS=(sbtm kde)
NS=(500000 2000000 3000000)

C=0.32
M=100

tid=$SLURM_ARRAY_TASK_ID
idx_sm=$(( tid % 2 ))
idx_n=$(( tid / 2 ))

dv=${DVS[0]}
score_method=${SCORE_METHODS[$idx_sm]}
n=${NS[$idx_n]}

source /gscratch/amath/vilin/conda/etc/profile.d/conda.sh
conda activate vlsbtm

srun python3 experiments/vlasov-landau-two-stream.py \
  --n "${n}" \
  --M "${M}" \
  --dt 0.05 \
  --dv "${dv}" \
  --C "${C}" \
  --score_method "${score_method}" \
  --wandb_project vlasov_landau_two_stream \
  --wandb_run_name "n${n}_M${M}_dt0.05_dv${dv}_C${C}_${score_method}"
