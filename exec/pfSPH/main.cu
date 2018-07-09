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
#include <cuda_gl_interop.h>

#include "Particles.h"
#include "frontends/frontendInterface.h"

template <typename T>
__host__ __device__
const T operator+(const T& lhs, const T& rhs)
{
    return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
}

template <typename T, typename SC>
__host__ __device__
const T operator*(const T& lhs, const SC& rhs)
{
    return {lhs.x * rhs, lhs.y * rhs, lhs.z * rhs};
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
    Particles* __pb = new Particles(10);
    Particles& pb = *__pb;

    // register position and velocity buffer with cuda
#if defined(FRONTEND_OPENGL)
    pb.registerGLPositionBuffer(fnd::getPositionBuffer(pb.size()));
    pb.registerGLVelocityBuffer(fnd::getVelocityBuffer(pb.size()));
    pb.mapRegisteredBuffers();
#endif

    Particle<POS,VEL> p;
    p.pos = {0.5f,0.5f,0.0f};
    p.vel = {1.0,0.,0.0};
    pb.storeParticle(p,0);

    p.pos = {-0.5f,-0.5f,0.0f};
    p.vel = {0.0,1.0,0.0};
    pb.storeParticle(p,1);

    p.pos = {0.5f,-0.5f,0.0f};
    p.vel = {1.0,1.0,0.0};
    pb.storeParticle(p,2);

    pb.copyToDevice();
    assert_cuda(cudaDeviceSynchronize());

    pb.unmapRegisteredBuffes(); // used for frontend stuff
    mpu::DeltaTimer dt;
    while(fnd::handleFrontend(dt.getDeltaTime()))
    {
        if(simShouldRun)
        {
            pb.mapRegisteredBuffers(); // used for frontend stuff
            // run simulation here
            integrateLeapfrog<<<1,100>>>(pb.createDeviceClone(),0.01f,true);
            assert_cuda(cudaGetLastError());
//            assert_cuda(cudaDeviceSynchronize());
            mpu::sleep_ms(1);
            pb.unmapRegisteredBuffes(); // used for frontend stuff
        }
    }

    pb.unregisterBuffers(); // probably not needed since it is done in destructor
    return 0;
}
