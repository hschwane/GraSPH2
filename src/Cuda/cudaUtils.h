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
#include "clionCudaHelper.h"
#include "../../external/cuda/helper_math.h"
#include "../Log/Log.h"
#include "../Range.h"
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
