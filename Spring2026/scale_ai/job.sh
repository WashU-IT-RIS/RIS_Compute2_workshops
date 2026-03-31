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

export imagefile=/storage1/fs1/ayush/Active/containers/pytorch_25_5.sif
export BASE="apptainer  exec --nv --writable-tmpfs --bind=/scratch,/storage1/fs1/ayush/Active${TMPFS} ${imagefile} "
export CMD="python train.py"

srun  --unbuffered --wait=120 --kill-on-bad-exit=0 --cpu-bind=none $BASE $CMD

if [ "$enable_profile" = "true" ];then
#  module load CUDA/11.7.0
  echo "cuda home: ${CUDA_HOME}"
  srun --wait=120 --kill-on-bad-exit=0 --cpu-bind=none $BASE dlprof --output_path=${SLURM_JOBID} --nsys_base_name=nsys_${SLURM_PROCID} --profile_name=dlpro_${SLURM_PROCID} --mode=pytorch --nsys_opts="-t osrt,cuda,nvtx,cudnn,cublas,cusparse,mpi, --cuda-memory-usage=true" -f true --reports=all --delay 60 --duration 120 ${CMD}
else
  for _experiment_index in $(seq 1 "${NEXP}"); do
    (
  	echo "Beginning trial ${_experiment_index} of ${NEXP}"
  	srun  --unbuffered --wait=120 --kill-on-bad-exit=0 --cpu-bind=none $BASE $CMD
    )
  done
fi