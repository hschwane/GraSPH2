#!/bin/bash

########################
# loads all modules and set pathes neded to compile and run  GraSPH2 and mpUtils on the Cobra HPC System at MPCDF.
# will be updated when Cobra configuration changes
# You can call it manually or add " source /path/to/this/scrip/loadCobraModules.sh " to your bashrc file  (~/.bashrc).
# When adding to the bashrc remember to log out and log in again for the modules to be loaded
########################

module load cmake/3.13
module load cuda/10.0
module load gcc/6
module load git
module load hdf5-serial
module unload hdf5-serial
module load hdf5-serial

export LIBRARY_PATH=$LIBRARY_PATH:/cobra/u/system/soft/SLE_12_SP4/packages/skylake/cuda/10.0.130/lib64
source /mpcdf/soft/SLE_12_SP4/packages/x86_64/intel_parallel_studio/2019.4/compilers_and_libraries_2019.4.243/linux/tbb/bin/tbbvars.sh intel64
