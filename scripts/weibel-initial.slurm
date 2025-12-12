#!/bin/bash
#SBATCH --job-name=weibel_init_sweep
#SBATCH --account=krishna
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --array=0-23
#SBATCH --output=logs/%x_%N_%A_%a.out

WIDTHS=(256 512)          # 2
NUMS=(2 3)                # 2
LRS=(2e-4 1e-4 5e-5)      # 3
BSS=(20000 40000)         # 2

tid=$SLURM_ARRAY_TASK_ID

idx_w=$(( tid % 2 ))
idx_n=$(( (tid / 2) % 2 ))
idx_lr=$(( (tid / 4) % 3 ))
idx_bs=$(( tid / 12 ))

hidden_width=${WIDTHS[$idx_w]}
hidden_num=${NUMS[$idx_n]}
lr=${LRS[$idx_lr]}
batch_size=${BSS[$idx_bs]}

source /gscratch/amath/vilin/conda/etc/profile.d/conda.sh
conda activate vlsbtm

srun python3 experiments/train-weibel-initial.py \
    --example weibel \
    --n 1000000 \
    --dv 2 \
    --beta 1e-2 \
    --c 0.3 \
    --k 0.2 \
    --hidden_width "$hidden_width" \
    --hidden_num "$hidden_num" \
    --lr "$lr" \
    --max_epochs 100000 \
    --batch_size "$batch_size" \
    --abs_tol 1e-4 \
    --gpu 0 \
    --wandb_project weibel_initial_sweep \
    --wandb_run_name "hw${hidden_width}_hn${hidden_num}_lr${lr}_bs${batch_size}"
