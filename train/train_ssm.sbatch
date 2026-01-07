#!/bin/bash -l

#SBATCH --job-name=train_ssm
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=env540
#SBATCH --account=env540
#SBATCH --output=slurm_logs/train_ssm_%j.out
#SBATCH --error=slurm_logs/train_ssm_%j.err

module purge
module load gcc cuda python

source .venv/bin/activate

mkdir -p slurm_logs

DATA_ROOT=${DATA_ROOT:-data}

srun python train/train_ssm.py \
    --data-root "${DATA_ROOT}" \
    --species-count 342 \
    --state-dim 256 \
    --batch-size 8 \
    --epochs 30 \
    --lr 1e-4 \
    --val-ratio 0.2 \
    --lambda-reg 0.05 \
    --pos-weight-cap 50.0 \
    --cb-beta 0.999 \
    --focal-gamma 2.0 \
    --wandb-mode online \
    --num-workers 4 \
    --image-size 224
