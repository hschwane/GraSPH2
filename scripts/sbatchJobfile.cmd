#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./tjob.out.%j
#SBATCH -e ./tjob.err.%j
# Initial working directory:
#SBATCH -D ./
#
#SBATCH -J test_slurm
# Queue:
#SBATCH --partition=gpu1        # If using only 1 GPU of a shared node
# Node feature:
#SBATCH --constraint="gpu"
# Specify number of GPUs to use:
#SBATCH --gres=gpu:1            # If using only 1 GPU of a shared node
#SBATCH --mem=61000             # Memory is necessary if using only 1 GPU
#
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1      # If using both GPUs of a node
#SBATCH --cpus-per-task=4
#
#SBATCH --mail-type=END
#SBATCH --mail-user=%u@rzg.mpg.de
#
# wall clock limit:
#SBATCH --time=24:00:00

module load cuda 

export OMP_NUM_THREADS=4
export OMP_PLACES=cores 

# Run the program:
srun ../build/GraSPH2 > prog.out
