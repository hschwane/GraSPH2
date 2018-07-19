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
#include <cuda_gl_interop.h>
#include <thrust/random.h>

#include "particles/Particles.h"
#include "frontends/frontendInterface.h"
#include <Cuda/cudaUtils.h>
#include <crt/math_functions.hpp>

#if 0

constexpr int BLOCK_SIZE = 256;
constexpr int PARTICLES = 1<<15;

int NUM_BLOCKS = (PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE;

__global__ void generate2DNBSystem(Particles particles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx>=particles.size())
    {
        printf("wrong dispatch parameters for particle count!");
        return;
    }

    thrust::random::default_random_engine rng;
    rng.discard(idx);
    thrust::random::uniform_real_distribution<float> dist(-1.0f,1.0f);

    Particle<POSM,VEL,ACC> p;

    p.pos.x = dist(rng);
    p.pos.y = dist(rng);
    p.pos.z = 0.0f;
    p.mass = 1.0f/particles.size();

    p.vel = cross(p.pos,{0.0f,0.0f, 0.75f});

    particles.storeParticle(p,idx);
}

__global__ void nbodyForces(Particles particles, f1_t eps2, const int numTiles)
{
    SharedParticles<BLOCK_SIZE,SHARED_POSM> shared;

    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

    const auto pi = particles.loadParticle<POSM,VEL>(idx);
    Particle<ACC> piacc;

    for (int tile = 0; tile < numTiles; tile++)
    {
        shared.copyFromGlobal(threadIdx.x, tile*blockDim.x+threadIdx.x, particles);
        __syncthreads();

        for(int j = 0; j<blockDim.x;j++)
        {
            auto pj = shared.loadParticle<POSM>(j);
            f3_t r = pi.pos-pj.pos;
            f1_t distSqr = dot(r,r) + eps2;

            f1_t invDist = rsqrt(distSqr);
            f1_t invDistCube =  invDist * invDist * invDist;
            piacc.acc -= r * pj.mass * invDistCube;

        }
        __syncthreads();
    }
    piacc.acc -= pi.vel * 0.1;
    particles.storeParticle(piacc,idx);
}


__global__ void integrateLeapfrog(Particles particles, f1_t dt, bool not_first_step)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx>=particles.size())
    {
        printf("wrong dispatch parameters for particle count!");
        return;
    }

    auto pi = particles.loadParticle<POSM,VEL,ACC>(idx);

    //   calculate velocity a_t
    pi.vel  = pi.vel + pi.acc * (dt*0.5f);

    // we could now change delta t here

    f1_t next_dt = dt;

    // calculate velocity a_t+1/2
    pi.vel = pi.vel + pi.acc * (dt*0.5f) * not_first_step;

    // calculate position r_t+1
    pi.pos = pi.pos + pi.vel * dt;

    particles.storeParticle(pi,idx);
}

int main()
{

    mpu::Log myLog( mpu::LogLvl::ALL, mpu::ConsoleSink());

    logINFO("pfSPH") << "Welcome to planetformSPH!";
    assert_cuda(cudaSetDevice(0));

    // handle frontend
    fnd::initializeFrontend();
    bool simShouldRun = true;
    fnd::setPauseHandler([&simShouldRun](bool pause){simShouldRun = !pause;});

    // generate 100 particles
    Particles pb(PARTICLES);

    // register position and velocity buffer with cuda
#if defined(FRONTEND_OPENGL)
    pb.registerGLPositionBuffer(fnd::getPositionBuffer(pb.size()));
    pb.registerGLVelocityBuffer(fnd::getVelocityBuffer(pb.size()));
    pb.mapRegisteredBuffers();
#endif

    generate2DNBSystem<<<NUM_BLOCKS,BLOCK_SIZE>>>(pb.createDeviceClone());
    assert_cuda(cudaGetLastError());
    assert_cuda(cudaDeviceSynchronize());

    nbodyForces<<<NUM_BLOCKS,BLOCK_SIZE>>>(pb.createDeviceClone(),0.01f, PARTICLES/ BLOCK_SIZE);
    assert_cuda(cudaGetLastError());
    integrateLeapfrog<<<NUM_BLOCKS,BLOCK_SIZE>>>(std::move(pb.createDeviceClone()),0.005f,false);
    assert_cuda(cudaGetLastError());

    pb.unmapRegisteredBuffes(); // used for frontend stuff
    mpu::DeltaTimer dt;
    while(fnd::handleFrontend(dt.getDeltaTime()))
    {
        if(simShouldRun)
        {
            pb.mapRegisteredBuffers(); // used for frontend stuff

            nbodyForces<<<NUM_BLOCKS,BLOCK_SIZE>>>(pb.createDeviceClone(),0.00001f, PARTICLES/ BLOCK_SIZE);
            assert_cuda(cudaGetLastError());
            integrateLeapfrog<<<NUM_BLOCKS,BLOCK_SIZE>>>(std::move(pb.createDeviceClone()),0.0025f,true);
            assert_cuda(cudaGetLastError());

            pb.unmapRegisteredBuffes(); // used for frontend stuff
            assert_cuda(cudaDeviceSynchronize());
        }
    }

    pb.unregisterBuffers(); // probably not needed since it is done in destructor
    return 0;
}

#endif




__global__ void test(DEV_MASS a, DEV_MASS b)
{
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    Particle<MASS> p;
    a.loadParticle(idx,p);
    b.storeParticle(idx,p);
}

int main()
{

    HOST_MASS host(100);
    DEV_MASS dev1(100);
    DEV_MASS dev2(100);

    Particle<MASS> p(10.0f);
    host.storeParticle(10,p);

    dev1 = host;

    test<<<1,100>>>(dev1.createDeviceCopy(), dev2.createDeviceCopy());

    host = dev2;

    Particle<MASS> p2;
    host.loadParticle(10, p2);

    std::cout << p.mass;
}