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

int main()
{

    mpu::Log myLog( mpu::LogLvl::ALL, mpu::ConsoleSink());

    logINFO("pfSPH") << "Welcome to planetformSPH!";

    Particles pb(1000);

    auto p = pb.loadParticle<POSM,VEL,ACC>(1);

    p.mass = 100;
    p.vel = {5,2,0};

    pb.copyToDevice();
    pb.copyFromDevice();
    assert_cuda(cudaDeviceSynchronize());

    pb.storeParticle(p,1);
    pb.storeParticle(p,2);


    logINFO("test") << pb.loadParticle<VEL>(2).vel.x;
    return 0;
}