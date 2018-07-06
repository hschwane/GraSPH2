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

__global__ void test(Particles* from, Particles* to)
{
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
//    int stride = blockDim.x * gridDim.x;

    SharedParticles<100,SHARED_POSM> sp;
    sp.copyFromGlobal(threadIdx.x, index, *from);

    auto p = sp.loadParticle<POSM>(index);

    to->storeParticle(p,index);
}

int main()
{

    mpu::Log myLog( mpu::LogLvl::ALL, mpu::ConsoleSink());

    logINFO("pfSPH") << "Welcome to planetformSPH!";
    assert_cuda(cudaSetDevice(0));

    // handle frontend
    fnd::initializeFrontend();
    bool simShouldRun = false;
    fnd::setPauseHandler([&simShouldRun](bool pause){simShouldRun = !pause;});

    // generate 100 particles
    std::unique_ptr<Particles> pb(new Particles(100));

    // register position and velocity buffer with cuda
#if defined(FRONTEND_OPENGL)
    pb->registerGLPositionBuffer(fnd::getPositionBuffer(pb->size()));
    pb->registerGLVelocityBuffer(fnd::getVelocityBuffer(pb->size()));
    pb->mapRegisteredBuffers();
#endif

    Particle<POS,VEL> p;
    p.pos = {0.5f,0.5f,0.0f};
    p.vel = {1.0,0.,0.0};
    pb->storeParticle(p,9);

    p.pos = {-0.5f,-0.5f,0.0f};
    p.vel = {0.0,1.0,0.0};
    pb->storeParticle(p,10);

    p.pos = {0.5f,-0.5f,0.0f};
    p.vel = {0.0,0.0,1.0};
    pb->storeParticle(p,11);

    pb->copyToDevice();
    assert_cuda(cudaDeviceSynchronize());

    pb->unmapRegisteredBuffes(); // used for frontend stuff
    mpu::DeltaTimer dt;
    while(fnd::handleFrontend(dt.getDeltaTime()))
    {
        if(simShouldRun)
        {
            pb->mapRegisteredBuffers(); // used for frontend stuff
            // run simulation here
            pb->unmapRegisteredBuffes(); // used for frontend stuff
        }
    }

    pb->unregisterBuffers(); // probably not needed since it is done in destructor
    return 0;
}
