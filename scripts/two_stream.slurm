#!/bin/bash
#SBATCH --job-name=two_stream
#SBATCH --account=amath       # e.g. amath
#SBATCH --partition=gpu-rtx6k        # or your GPU partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/two_stream_%j.out
#SBATCH --error=logs/two_stream_%j.err

# modules / env
conda activate vlsbtm

cd /mmfs1/gscratch/amath/vilin/Vlasov-Landau-SBTM/scripts  # repo root

# run
srun python ../experiments/vlasov-landau-two-stream.py \
  --n 1000000 \
  --M 100 \
  --dt 0.05 \
  --gpu 0 \
  --dv 2 \
  --final_time 50.0 \
  --C 0.08 \
  --alpha 0.005 \
  --k 0.2 \
  --c 2.4 \
  --score_method kde \
  --wandb_project vlasov_landau_two_stream \
  --wandb_run_name "n1e6_M100_dt0.05_C0.08_kde" \
  --wandb_mode online
