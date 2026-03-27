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

# Load Apptainer module

module load ris apptainer/1.3.4        # Replace with appropriate Apptainer version

# Define the path to the pulled NVIDIA container

CONTAINER_PATH=/storage1/fs1/test/Active/pytorch:22.10-py3.sif  # Adjust path as needed

# Run your PyTorch script inside the container

apptainer exec --nv --bind=/storage1/fs1/test/Active  $CONTAINER_PATH python train.py
