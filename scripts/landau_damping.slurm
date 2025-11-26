#!/bin/bash
#SBATCH --job-name=landau_damping
#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=24:00:00
#SBATCH --array=0-59
#SBATCH --output=logs/%x_%N_%A_%a.out

DVS=(2 3)
SCORE_METHODS=(kde sbtm)
C_VALUES=(0.1 0.4 1.0)
N_PARTICLES=(100000 200000 400000 600000 800000)
M_VALUES=(10 25 40 60 80)

tid=$SLURM_ARRAY_TASK_ID

idx_C=$(( tid % 3 ))
idx_sm=$(( (tid / 3) % 2 ))
idx_dv=$(( (tid / 6) % 2 ))
idx_nm=$(( tid / 12 ))

dv=${DVS[$idx_dv]}
score_method=${SCORE_METHODS[$idx_sm]}
C=${C_VALUES[$idx_C]}
n=${N_PARTICLES[$idx_nm]}
M=${M_VALUES[$idx_nm]}

source /gscratch/amath/vilin/conda/etc/profile.d/conda.sh
conda activate vlsbtm

srun python3 experiments/vlasov-landau-damping.py \
  --n "${n}" \
  --M "${M}" \
  --dt 0.02 \
  --dv "${dv}" \
  --C "${C}" \
  --score_method "${score_method}" \
  --wandb_project vlasov_landau_damping \
  --wandb_run_name "n${n}_M${M}_dt0.02_dv${dv}_C${C}_${score_method}"
