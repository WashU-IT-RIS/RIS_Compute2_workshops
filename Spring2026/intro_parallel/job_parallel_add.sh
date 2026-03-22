#!/bin/bash
#SBATCH --job-name=pthread_test
#SBATCH --output=res.out
#SBATCH --nodes=1               # Pthreads only work on ONE node
#SBATCH --cpus-per-task=8       # Number of cores to use
#SBATCH --time=00:05:00
#SBATCH --partition general-cpu
#SBATCH --account compute2-ris   #please change this to your account

# Load GCC 11 (the module name varies by cluster)
module load ris shared 
module load gcc/13.1.0
if [ -f "./parallel_add" ]; then
    echo "Removing existing executable..."
    rm -f ./parallel_add
fi
# Compile with the pthread flag
gcc -O3 parallel_add.c -o parallel_add -lpthread

# Run the program, passing the number of CPUs allocated by Slurm
./parallel_add $SLURM_NPROC