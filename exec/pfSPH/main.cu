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

constexpr int FORCES_BLOCK_SIZE = 256;
constexpr int PARTICLES = 16384;

template <typename T>
__host__ __device__
const T operator+(const T& lhs, const T& rhs)
{
    return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
}

template <typename T>
__host__ __device__
const T operator-(const T& lhs, const T& rhs)
{
    return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
}

template <typename T, typename SC>
__host__ __device__
const T operator*(const T& lhs, const SC& rhs)
{
    return {lhs.x * rhs, lhs.y * rhs, lhs.z * rhs};
}

__global__ void generate2DNBSystem(Particles particles)
{
    int indexX = blockIdx.x * blockDim.x + threadIdx.x;
    int strideX = blockDim.x * gridDim.x;

    thrust::random::default_random_engine rng;
    rng.discard(indexX);
    thrust::random::uniform_real_distribution<float> dist(-1.0f,1.0f);

    Particle<POSM,VEL,ACC> p;
    p.mass = 1.0f/particles.size();

    for(int i= indexX; i < particles.size(); i+=strideX)
    {
        p.pos.x = dist(rng);
        p.pos.y = dist(rng);
        p.pos.z = 0.0f;
        particles.storeParticle(p,i);
    }
}


__global__ void nbodyForces(Particles particles, f1_t eps2)
{
    SharedParticles<FORCES_BLOCK_SIZE,SHARED_POSM,SHARED_VEL> shared;

    const unsigned indexX = blockIdx.x * blockDim.x + threadIdx.x;
    int strideX = blockDim.x * gridDim.x;

    for(int i= indexX; i < particles.size(); i+=strideX)
    {
        const auto pi = particles.loadParticle<POSM,VEL>(i);
        Particle<ACC> piacc;

        for(int k= indexX; k < particles.size(); k += blockDim.x)
        {
            shared.copyFromGlobal(threadIdx.x,k,particles);
            __syncthreads();

            for(int j = 0; j<blockDim.x;j++)
            {
                auto pj = shared.loadParticle<POSM,VEL>(j);

                const f3_t rij = pi.pos - pj.pos;
                const f1_t r2 = rij.x * rij.x + rij.y * rij.y + rij.z * rij.z;
                const f1_t r2e = r2 + eps2;
                const f1_t s = pj.mass/(r2e*sqrt(r2e));
                piacc.acc = piacc.acc - rij * s;

//                f3_t velij = pi.vel - pj.vel;
            }
        }
        particles.storeParticle(piacc,i);
    }
}


__global__ void integrateLeapfrog(Particles particles, f1_t dt, bool not_first_step)
{
    int indexX = blockIdx.x * blockDim.x + threadIdx.x;
    int strideX = blockDim.x * gridDim.x;

    for(int i= indexX; i < particles.size(); i+=strideX)
    {
        auto pi = particles.loadParticle<POSM,VEL,ACC>(i);

        //   calculate velocity a_t
        pi.vel  = pi.vel + pi.acc * (dt*0.5f);

        // we could now change delta t here

        f1_t next_dt = dt;

        // calculate velocity a_t+1/2
        pi.vel = pi.vel + pi.acc * (dt*0.5f) * not_first_step;

        // calculate position r_t+1
        pi.pos = pi.pos + pi.vel * dt;

        particles.storeParticle(pi,i);
    }
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

    generate2DNBSystem<<<(PARTICLES+FORCES_BLOCK_SIZE-1)/FORCES_BLOCK_SIZE,FORCES_BLOCK_SIZE>>>(pb.createDeviceClone());
    assert_cuda(cudaGetLastError());
    assert_cuda(cudaDeviceSynchronize());


    pb.unmapRegisteredBuffes(); // used for frontend stuff
    mpu::DeltaTimer dt;
    while(fnd::handleFrontend(dt.getDeltaTime()))
    {
        if(simShouldRun)
        {
            pb.mapRegisteredBuffers(); // used for frontend stuff
            // run simulation here
            nbodyForces<<<(PARTICLES+FORCES_BLOCK_SIZE-1)/FORCES_BLOCK_SIZE,FORCES_BLOCK_SIZE>>>(pb.createDeviceClone(),0.01f);
            integrateLeapfrog<<<(PARTICLES+FORCES_BLOCK_SIZE-1)/FORCES_BLOCK_SIZE,FORCES_BLOCK_SIZE>>>(std::move(pb.createDeviceClone()),0.01f,true);
            assert_cuda(cudaGetLastError());
//            assert_cuda(cudaDeviceSynchronize());
            pb.unmapRegisteredBuffes(); // used for frontend stuff
            mpu::sleep_ms(1);
        }
    }

    pb.unregisterBuffers(); // probably not needed since it is done in destructor
    return 0;
}
