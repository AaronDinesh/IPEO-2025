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

source .venv/bin/activate

RUN_NAME=$1

srun python train/train_ssm.py \
    --data-root "/scratch/izar/dinesh/geoplant" \
    --state-dim 256 \
    --batch-size 8 \
    --epochs 30 \
    --lr 1e-4 \
    --val-ratio 0.2 \
    --lambda-reg 0.05 \
    --wandb-mode online \
    --num-workers 4 \
    --image-size 224 \
    --use-sampler \
    --checkpoint-dir "/scratch/izar/dinesh/checkpoints" \
    --run-name=$1
