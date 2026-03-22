#!/bin/bash
#SBATCH --job-name=omp_bench
#SBATCH --output=omp_results.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=4G
#SBATCH --time=00:10:00

EXE_NAME="./omp_mat"

# 1. Cleanup
if [ -f "$EXE_NAME" ]; then
    rm -f "$EXE_NAME"
fi

# 2. Setup
module load ris shared 
module load gcc/13.1.0

# 3. Compilation (CRITICAL: use -fopenmp instead of -lpthread)
gcc -O3 -fopenmp omp_mat.c -o omp_mat

# 4. Execution
if [ -f "$EXE_NAME" ]; then
    # OpenMP automatically looks for the OMP_NUM_THREADS variable
    # which Slurm sets based on --cpus-per-task
    export OMP_NUM_THREADS=$SLURM_NPROCS
    $EXE_NAME
else
    echo "Error: Compilation failed."
    exit 1
fi

# 5. Cleanup
# rm -f "$EXE_NAME"