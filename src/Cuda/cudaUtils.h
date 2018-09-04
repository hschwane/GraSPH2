/*
 * mpUtils
 * cudaUtils.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */
#ifndef MPUTILS_CUDAUTILS_H
#define MPUTILS_CUDAUTILS_H


// only include this file in *.cu files
//--------------------
#ifndef __CUDACC__
    #error "Only use the cudaUtils.h if compiling *.cu files with nvcc!"
#endif
//--------------------

// includes
//--------------------
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <device_launch_parameters.h>

#include <cstdio>
#include <cstdlib>
#include "../../external/cuda/helper_math.h"
#include "../Log/Log.h"
#include "../Range.h"
//--------------------

// make clion understand some cuda specific stuff
//--------------------
#ifdef __JETBRAINS_IDE__
#include "math.h"
#define __CUDACC__ 1
#define __host__
#define __device__
#define __global__
#define __noinline__
#define __forceinline__
#define __shared__
#define __constant__
#define __managed__
#define __restrict__
// CUDA Synchronization
inline void __syncthreads() {};
inline void __threadfence_block() {};
inline void __threadfence() {};
inline void __threadfence_system();
inline int __syncthreads_count(int predicate) {return predicate;};
inline int __syncthreads_and(int predicate) {return predicate;};
inline int __syncthreads_or(int predicate) {return predicate;};
template<class T> inline T __clz(const T val) { return val; }
template<class T> inline T __ldg(const T* address){return *address;};
// CUDA TYPES
typedef unsigned short uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;
typedef unsigned long long ulonglong;
typedef long long longlong;
#endif
//--------------------

// some defines
//--------------------
// wrap cuda function calls in this to check for errors
#define assert_cuda(CODE) mpu::_cudaAssert((CODE),MPU_FILEPOS)
// use this to define a function as usable on host and device
#ifndef CUDAHOSTDEV
#define CUDAHOSTDEV __host__ __device__
#endif
//--------------------

namespace mpu {

inline void _cudaAssert(cudaError_t code, std::string &&filepos)
{
    if(code != cudaSuccess)
    {
        std::string message("Cuda error: " + std::string(cudaGetErrorString(code)));

        if(!(mpu::Log::noGlobal()))
        {
            if(mpu::Log::getGlobal().getLogLevel() >= mpu::LogLvl::FATAL_ERROR)
                mpu::Log::getGlobal()(mpu::LogLvl::FATAL_ERROR, std::move(filepos), "cuda") << message;
            mpu::Log::getGlobal().flush();
        }

        throw std::runtime_error("Cuda error: " + message);
    }
}

class Managed
{
public:
    void *operator new(size_t len)
    {
        void *ptr;
        assert_cuda(cudaMallocManaged(&ptr, len));
        assert_cuda(cudaDeviceSynchronize());
        return ptr;
    }

    void operator delete(void *ptr)
    {
        assert_cuda(cudaDeviceSynchronize());
        assert_cuda(cudaFree(ptr));
    }
};

/**
 * @brief generates a Range to be used in a for each loop inside a kernel to decouple the grid size from the data size
 *          indices will run in the range [0,problemSize)
 * @param problemSize the size of the data to process
 * @return a mpu::Range object to be used inside a for each loop
 */
inline __device__ Range<int> gridStrideRange(int problemSize)
{
    return Range<int>(blockIdx.x * blockDim.x + threadIdx.x, problemSize, gridDim.x * blockDim.x);
}


/**
 * @brief generates a Range to be used in a for each loop inside a kernel to decouple the grid size from the data size
 *          indices will run in the range [firstElement,problemSize)
 * @param firstElement the first element of the dataset to process
 * @param problemSize the size of the data to process
 * @return a mpu::Range object to be used inside a for each loop
 */
inline __device__ Range<int> gridStrideRange(int firstElement, int problemSize)
{
    firstElement += blockIdx.x * blockDim.x + threadIdx.x;
    return Range<int>(firstElement, problemSize, gridDim.x * blockDim.x);
}

}

#endif //MPUTILS_CUDAUTILS_H
