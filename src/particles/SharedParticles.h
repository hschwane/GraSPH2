/*
 * mpUtils
 * SharedParticles.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */
#ifndef MPUTILS_SHAREDPARTICLES_H
#define MPUTILS_SHAREDPARTICLES_H

// includes
//--------------------
#include "particle_tmp_utils.h"
#include "Particle.h"
#include <mpUtils/mpUtils.h>
#include <mpUtils/mpCuda.h>
//--------------------

//-------------------------------------------------------------------
/**
 * class SharedParticles
 *
 * handles particles in shared memory on the device
 *
 * usage:
 * Specify how many particles should be stored in shared memory and what attributes should be defined for them.
 * Beware that you have to take care of synchronization yourself.
 * Supported particle attributes are:
 * SHARED_POSM
 * SHARED_VEL
 * SHARED_ACC
 *
 */
template <size_t n, template <size_t> class... TArgs>
class SharedParticles : public TArgs<n>...
{
public:
    __device__ SharedParticles() : TArgs<n>()... {}
    SharedParticles(const SharedParticles&)=delete;
    SharedParticles& operator=(const SharedParticles&)=delete;

    template<typename... particleArgs>
    __device__
    Particle<particleArgs...> loadParticle(size_t id) //!< get a particle object with the requested members
    {
        Particle<particleArgs...> p{};
        int t[] = {0, ((void)TArgs<n>::loadParticle(id,p),1)...}; // call load particle functions of all the base classes
        (void)t[0]; // silence compiler warning abut t being unused
        return p;
    }

    template<typename... particleArgs>
    __device__
    void storeParticle(size_t id, const Particle<particleArgs...>& p) //!< set the attributes of particle id according to the particle object
    {
        int t[] = {0, ((void)TArgs<n>::storeParticle(id,p),1)...};
        (void)t[0]; // silence compiler warning abut t being unused
    }

    __device__ size_t size() {return n;}
};

#endif //MPUTILS_SHAREDPARTICLES_H
