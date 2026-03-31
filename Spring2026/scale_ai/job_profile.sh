#!/bin/bash
#SBATCH --job-name=ddnet
#SBATCH --nodes 2
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8384                # total memory per node (4 GB per cpu-core is default)
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-node 2             #GPU per node
#SBATCH --partition=general-gpu # slurm partition
#SBATCH --time=24:30:00          # time limit
#SBATCH -A compute2-ris           # account name
#SBATCH --reservation workshop2026


export MASTER_PORT=$port
echo "master port: $MASTER_PORT"
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE=$WORLD_SIZE"
echo "slurm job: $SLURM_JOBID"
#expor job_id=$SLURM_JOBID

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "SLURM_NNODES=$SLURM_NNODES"
echo "SLURMTMPDIR=$SLURMTMPDIR"

: "${NEXP:=1}"

module load ris shared
module load apptainer

export imagefile=/storage1/fs1/ayush/Active/containers/pytorch_25_5.sif
export BASE="apptainer  exec --nv --writable-tmpfs --bind=/scratch,/storage1/fs1/ayush/Active${TMPFS} ${imagefile} "
export CMD="python train.py"

echo "cuda home: ${CUDA_HOME}"
srun --wait=120 --kill-on-bad-exit=0 --cpu-bind=none $BASE \
nsys profile --trace='cuda','cublas','cudnn','osrt','nvtx','cudnn','cusparse','mpi' --stats='true' --sample=none --export=sqlite -f true -o profile.${SLURM_PROCID} ${cmd}
