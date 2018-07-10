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

#include "Particles.h"
#include "frontends/frontendInterface.h"
#include <Cuda/cudaUtils.h>
#include <crt/math_functions.hpp>

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

__device__ void interaction(const Particle<POSM,VEL>& bi, const Particle<POSM>& bj, f3_t& ai, f1_t eps2)
{
    f3_t r;

    // r_ij  [3 FLOPS]
    r.x = bj.pos.x - bi.pos.x;
    r.y = bj.pos.y - bi.pos.y;
    r.z = bj.pos.z - bi.pos.z;

    // distSqr = dot(r_ij, r_ij) + EPS^2  [6 FLOPS]
    f1_t distSqr = r.x * r.x + r.y * r.y + r.z * r.z;
    distSqr += eps2;

    // invDistCube =1/distSqr^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    f1_t invDist = rsqrt(distSqr);
    f1_t invDistCube =  invDist * invDist * invDist;

    // s = m_j * invDistCube [1 FLOP]
    f1_t s = bj.mass * invDistCube;

    // a_i =  a_i + s * r_ij [6 FLOPS]
    ai.x += r.x * s;
    ai.y += r.y * s;
    ai.z += r.z * s;
}

__global__ void nbodyForces(Particles particles, f1_t eps2, const int numTiles)
{
    SharedParticles<BLOCK_SIZE,SHARED_POSM> shared;

    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

    const auto pi = particles.loadParticle<POSM,VEL>(idx);
    Particle<ACC> piacc;

//    int numTiles = particles.size() / BLOCK_SIZE;
    for (int tile = 0; tile < numTiles; tile++)
    {
        shared.copyFromGlobal(threadIdx.x, tile*blockDim.x+threadIdx.x, particles);
        __syncthreads();

        for(int j = 0; j<blockDim.x;j++)
        {
            interaction(pi,shared.loadParticle<POSM>(j),piacc.acc,eps2);
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
