/*
 * mpUtils
 * main.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail: hendrik.schwanekamp@gmx.net
 *
 * mpUtils = my personal Utillities
 * A utility library for my personal c++ projects
 *
 * Copyright 2016 Hendrik Schwanekamp
 *
 */

#include <mpUtils.h>
#include <Cuda/cudaUtils.h>

#include "Particles.h"

__global__ void test(Particles* from, Particles* to)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;



    auto p = from->loadParticle<M>(index);

    to->storeParticle(p,index);
}

int main()
{

    mpu::Log myLog( mpu::LogLvl::ALL, mpu::ConsoleSink());

    logINFO("pfSPH") << "Welcome to planetformSPH!";

    Particles* pb1 = new Particles(100);
    Particles* pb2 = new Particles(100);

    Particle<M> p;
    p.mass = 10.0f;
    pb1->storeParticle(p,10);

    assert_cuda(cudaDeviceSynchronize());
    pb1->copyToDevice();
    assert_cuda(cudaDeviceSynchronize());

    test<<<1,100>>>(pb1,pb2);

    assert_cuda(cudaDeviceSynchronize());
    pb2->copyFromDevice();
    assert_cuda(cudaDeviceSynchronize());
    p = pb2->loadParticle<M>(10);

    logINFO("test") << pb2->loadParticle<M>(10).mass;
    logINFO("test") << pb1->loadParticle<M>(10).mass;
    return 0;
}