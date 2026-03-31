#!/bin/bash
#SBATCH --job-name=ddnet
#SBATCH --nodes 1
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8384                # total memory per node (4 GB per cpu-core is default)
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-node 1             #GPU per node
#SBATCH --partition=general-gpu # slurm partition
#SBATCH --time=24:30:00          # time limit
#SBATCH -A compute2-ris           # account name
#SBATCH --reservation workshop2026

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "SLURM_NNODES=$SLURM_NNODES"
echo "SLURMTMPDIR=$SLURMTMPDIR"

: "${NEXP:=1}"

module load ris shared
module load apptainer

export imagefile=/storage1/fs1/ayush/Active/containers/pytorch_25_05.sif
export BASE="apptainer  exec --nv --writable-tmpfs --bind=/storage1/fs1/ayush/Active${TMPFS} ${imagefile} "
export CMD="python train.py"

srun  --unbuffered --wait=120 --kill-on-bad-exit=0 --cpu-bind=none $BASE $CMD
