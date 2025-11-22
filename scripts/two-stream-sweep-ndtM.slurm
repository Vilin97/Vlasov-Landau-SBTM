#!/bin/bash
#SBATCH --job-name=two_stream_ndtM
#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=1:00:00
#SBATCH --array=0-47
#SBATCH --output=logs/%x_%N_%A_%a.out

source /gscratch/amath/vilin/conda/etc/profile.d/conda.sh
conda activate vlsbtm

N_VALUES=(1000000 10000000 100000000)   # 1e6, 1e7, 1e8
DT_VALUES=(0.05 0.02 0.01 0.005)
M_VALUES=(40 100 400 1000)

cid=${SLURM_ARRAY_TASK_ID}

# decode indices: M (4), dt (4), n (3)
m_idx=$(( cid % 4 ))
cid=$(( cid / 4 ))        # now 0..11
dt_idx=$(( cid % 4 ))
cid=$(( cid / 4 ))        # now 0..2
n_idx=$cid

N=${N_VALUES[$n_idx]}
DT=${DT_VALUES[$dt_idx]}
M=${M_VALUES[$m_idx]}

run_name="ndtM_n${N}_M${M}_dt${DT}"

srun python3 experiments/two-stream-sweep.py \
  --n "${N}" \
  --M "${M}" \
  --dt "${DT}" \
  --gpu 0 \
  --dv 2 \
  --final_time 50.0 \
  --alpha 0.005 \
  --k 0.2 \
  --c 2.4 \
  --wandb_project two_stream_ndtM \
  --wandb_run_name "${run_name}" \
  --wandb_mode online
