/*
 * mpUtils
 * mpCuda.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */
#ifndef MPUTILS_MPCUDA_H
#define MPUTILS_MPCUDA_H

// only include this file in *.cu files
//--------------------
#ifndef __CUDACC__
    #error "Only use the cudaUtils.h if compiling *.cu files with nvcc!"
#endif
//--------------------

// includes
//--------------------

// cuda api
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <device_launch_parameters.h>

// cuda stuff from the framework
#include "Cuda/cudaUtils.h"
#include "../external/cuda/helper_math.h"
#include "Cuda/Matrix.h"
#include "Cuda/clionCudaHelper.h"
//--------------------

#endif //MPUTILS_MPCUDA_H
