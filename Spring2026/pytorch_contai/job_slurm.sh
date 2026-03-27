#!/bin/bash

#SBATCH --job-name=pytorch_job   # Job name

#SBATCH --account=compute2-ris   # Replace with respective account

#SBATCH --output=output_%j.log   # Output file

#SBATCH --error=error_%j.log     # Error file

#SBATCH --ntasks=1               # Number of tasks (adjust if needed)

#SBATCH --time=01:00:00          # Time limit (hh:mm:ss)

#SBATCH --partition=general-gpu  # Partition/queue to submit to

#SBATCH --gres=gpu:1             # Number of GPUs (adjust if needed)

#SBATCH --mem=16G                # Memory limit (adjust if needed)

#SBATCH --container-image='nvcr.io#nvidia/pytorch:25.05-py3'

#SBATCH --container-mounts=/storage1/fs1/test/active

# Load additional modules as required

# Run your PyTorch script inside the container using Pyxis

python train.py
