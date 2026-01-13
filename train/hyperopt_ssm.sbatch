#!/bin/bash -l

#SBATCH --job-name=hyperopt_ssm
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --account=env540
#SBATCH --output=slurm_logs/hyperopt_ssm_%j.out
#SBATCH --error=slurm_logs/hyperopt_ssm_%j.err

module purge
module load gcc cuda python

export PYTHONPATH="/home/dinesh/IPEO-2025:$PYTHONPATH"
source .venv/bin/activate

RUN_NAME=${1:-hyperopt-ssm}

srun python train/train_ssm.py \
    --train "/scratch/izar/dinesh/geoplant/training_set.npz" \
    --test "/scratch/izar/dinesh/geoplant/validation_set.npz" \
    --state-space-dim 64 \
    --batch-size 64 \
    --epochs 200 \
    --learning-rate 3e-4 \
    --wandb \
    --wandb-group "hyperopt-ssm-auprc" \
    --wandb-run-name="${RUN_NAME}" \
    --threshold 0.42016854191620334 \
    --checkpoint-dir "/scratch/izar/dinesh/checkpoints" \
    --state-change-reg-weight 0.860930481913434 \
    --state-change-threshold 1.4123737432567047 \
    --reg-loss-weight 0.01 \
    --hyperopt \
    --hyperopt-max-evals 20 \
    --hyperopt-metric "test/auprc_macro" \
    --hyperopt-disable-notifications \
    --hyperopt-seed 42 \
    --torch-compile
