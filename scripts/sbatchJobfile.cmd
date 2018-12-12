#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./tjob.out.%j
#SBATCH -e ./tjob.err.%j
# Initial working directory:
#SBATCH -D ./
#
#SBATCH -J GraSPH2
# Queue:
#SBATCH --partition=gpu1
# Node feature:
#SBATCH --constraint="gpu"
# Specify number of GPUs to use:
#SBATCH --gres=gpu:1
#SBATCH --mem=61000
#
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
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
