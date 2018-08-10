/*
 * mpUtils
 * algorithms.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */
#ifndef MPUTILS_ALGORITHMS_H
#define MPUTILS_ALGORITHMS_H

// includes
//--------------------
#include <nvfunctional>
#include "Particles.h"
#include <mpUtils.h>
//--------------------

//-------------------------------------------------------------------
// template functions to execute some common algorithms on particles

/**
 * @brief initalize all particles in the particle buffer pb using the function f
 * @tparam particlesT the type of the Particles object which manages the Particles
 * @tparam callableT the type of the callable f
 * @param pb a reference to the Particles object which contains the particles
 * @param f something callable that takes an index and returns a Particle which can then be loaded into pb
 */
template <typename particlesT, typename callableT>
__device__ void initializeEach(particlesT& pb, const callableT& f)
{
    for(const auto &i : mpu::gridStrideRange(pb.size()))
    {
        pb.storeParticle(i, f(i));
    }
}

/**
 * @brief execute a function on every particle
 * @tparam particlesT the type of the Particles object which manages the Particles
 * @tparam rpT the type of particle to be stored after the execution of f
 * @tparam PArgs the bases of the particle that is passed to f
 * @param pb a reference to the Particles object which contains the particles
 * @param f  something callable that excepts a Particle<PArgs...> as an argument, does some computation and returns
 *          a Particle object of type rpT which is then stored in the buffer
 */
template <typename particlesT, typename rpT, typename... PArgs>
__device__ void doForEach( particlesT& pb, nvstd::function< rpT(Particle<PArgs...>) > f)
{
    for(const auto &i : mpu::gridStrideRange(pb.size()))
    {
        pb.storeParticle(i, f(pb.template loadParticle<PArgs...>(i)));
    }
}

#endif //MPUTILS_ALGORITHMS_H
