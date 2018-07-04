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

// includes
//--------------------
// most of those are mainly needed for clion and not actual compiling
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cstdlib>
#include "../Log/Log.h"
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

#define assert_cuda(CODE) _cudaAssert((CODE),MPU_FILEPOS)

inline void _cudaAssert(cudaError_t code, std::string&& filepos)
{
    if (code != cudaSuccess)
    {
        std::string message("Cuda error: " + std::string(cudaGetErrorString(code)));

        if (!(mpu::Log::noGlobal()))
        {
            if(mpu::Log::getGlobal().getLogLevel() >= mpu::LogLvl::FATAL_ERROR)
                mpu::Log::getGlobal()(mpu::LogLvl::FATAL_ERROR, std::move(filepos), "cuda") << message;
            mpu::Log::getGlobal().flush();
        }

        throw std::runtime_error("Cuda error: " + message);
    }
}


// some converting functions
template <typename d1, typename d2>
__host__ __device__
d1 toDim2(d2 rhs)
{
    return {rhs.x, rhs.y};
};

template <typename d1, typename d2>
__host__ __device__
d1 toDim3(d2 rhs)
{
    return {rhs.x, rhs.y, rhs.z};
};

#endif //MPUTILS_CUDAUTILS_H
