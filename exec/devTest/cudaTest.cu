/*
 * mpUtils
 * cudaTest.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */

#include "cudaTest.h"
#include <Cuda/cudaUtils.h>

class Managed
{
public:
    void *operator new(size_t len) {
        void *ptr;
        assert_cuda(cudaMallocManaged(&ptr, len));
        assert_cuda(cudaDeviceSynchronize());
        return ptr;
    }

    void operator delete(void *ptr) {
        assert_cuda(cudaDeviceSynchronize());
        assert_cuda(cudaFree(ptr));
    }
};

class Array : public Managed
{
public:
    Array(size_t n) : length(n) {cudaMallocManaged(&data,n* sizeof(float));}
    ~Array() {cudaFree(data);}

    __host__ __device__
    float& operator[](int pos) {return data[pos];}

    const size_t length;
private:
    float* data;
};

// CUDA kernel to add elements of two arrays
__global__
void add(size_t n, Array& x, Array& y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;


    for (int i = index; i < n; i += stride)
    {
        y[i] = x[i] + y[i];
    }
}

void testCuda()
{
    size_t N = 1<<28;
    auto x=std::make_unique<Array>(N);
    auto y=std::make_unique<Array>(N);

    assert_cuda( cudaPeekAtLastError() );

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        (*x)[i] = 1.0f;
        (*y)[i] = 2.0f;
    }

    // Launch kernel on 1M elements on the GPU
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    add<<<numBlocks, blockSize>>>(N, *x, *y);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs((*y)[i]-3.0f));
    logINFO("TEST") << "Max error: " << maxError << std::endl;

    return;
}