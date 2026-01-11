#!/bin/bash -l

#SBATCH --job-name=train_ssm
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --account=env540
#SBATCH --output=slurm_logs/train_ssm_%j.out
#SBATCH --error=slurm_logs/train_ssm_%j.err

module purge
module load gcc cuda python

export PYTHONPATH="/home/dinesh/IPEO-2025:$PYTHONPATH"
source .venv/bin/activate

RUN_NAME=$1

srun python train/train_ssm.py \
    --train "/scratch/izar/dinesh/geoplant/training_set.npz" \
    --test "/scratch/izar/dinesh/geoplant/validation_set.npz" \
    --state-space-dim 64 \
    --batch-size 64 \
    --epochs 200 \
    --learning-rate 1e-3 \
    --wandb \
    --threshold 0.85 \
    --checkpoint-dir "/scratch/izar/dinesh/checkpoints" \
    --wandb-run-name=$1 \
    --state-change-reg-weight 0.01 \
    --reg-loss-weight 0.01 \
