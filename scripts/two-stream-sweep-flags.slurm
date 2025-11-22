#!/bin/bash
#SBATCH --job-name=two_stream_flags
#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=1:00:00
#SBATCH --array=0-127
#SBATCH --output=logs/%x_%N_%A_%a.out

source /gscratch/amath/vilin/conda/etc/profile.d/conda.sh
conda activate vlsbtm

# fixed base parameters
N=1000000
M=100
DT=0.05

cid=${SLURM_ARRAY_TASK_ID}

# 7 binary flags
b0=$(( cid & 1 )); cid=$(( cid >> 1 ))  # init_v_mirror
b1=$(( cid & 1 )); cid=$(( cid >> 1 ))  # init_x_symmetric
b2=$(( cid & 1 )); cid=$(( cid >> 1 ))  # step_x_use_vnew
b3=$(( cid & 1 )); cid=$(( cid >> 1 ))  # step_v_use_Enew
b4=$(( cid & 1 )); cid=$(( cid >> 1 ))  # step_E_use_xnew
b5=$(( cid & 1 )); cid=$(( cid >> 1 ))  # step_E_use_vnew
b6=$(( cid & 1 ))                       # step_E_zero_mean

FLAG_ARGS=""
(( b0 == 1 )) && FLAG_ARGS+=" --init_v_mirror"
(( b1 == 1 )) && FLAG_ARGS+=" --init_x_symmetric"
(( b2 == 1 )) && FLAG_ARGS+=" --step_x_use_vnew"
(( b3 == 1 )) && FLAG_ARGS+=" --step_v_use_Enew"
(( b4 == 1 )) && FLAG_ARGS+=" --step_E_use_xnew"
(( b5 == 1 )) && FLAG_ARGS+=" --step_E_use_vnew"
(( b6 == 1 )) && FLAG_ARGS+=" --step_E_zero_mean"

run_name="flags_n${N}_M${M}_dt${DT}_ivm${b0}_ixs${b1}_xvn${b2}_vEn${b3}_Exn${b4}_Evn${b5}_E0m${b6}"

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
  ${FLAG_ARGS} \
  --wandb_project two_stream_flags \
  --wandb_run_name "${run_name}" \
  --wandb_mode online
