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
#include "particle_buffer_impl.h"
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
 * See particle_buffer_impl.h for all possible attributes.
 *
 */
template <size_t n, template <size_t> class... TArgs>
class SharedParticles : public TArgs<n>...
{
public:
    static_assert( mpu::conjunction_v< std::is_base_of<Shared_base, TArgs<0>>...>,
                   "Only use the SharedParticles class with instantiations of Shared_base! See file particle_buffer_impl.h for possible bases."); //!< check if only valid bases are used for the particle
    static_assert( checkOrder_v<std::tuple<TArgs<0>...>,shared_base_order >,
                   "Use particle Attributes in correct order without duplicates. See dref_base_order in particle_buffer_impl.h.");

    // types
    using attributes = std::tuple<TArgs<n>...>;
    using particleType = merge_particles_t < typename TArgs<n>::particleType ... >;

    __device__ SharedParticles() : TArgs<n>()... {}
    SharedParticles(const SharedParticles&)=delete;
    SharedParticles& operator=(const SharedParticles&)=delete;

    template<typename... particleArgs, std::enable_if_t< (sizeof...(particleArgs) > 0),int> =0>
    __device__ auto loadParticle(size_t id) const //!< get a particle object with the requested members
    {
        Particle<particleArgs...> p{};
        int t[] = {0, ((void)TArgs<n>::loadParticle(id,p),1)...}; // call load particle functions of all the base classes
        (void)t[0]; // silence compiler warning abut t being unused
        return p;
    }

    template<typename... particleArgs, std::enable_if_t< (sizeof...(particleArgs) == 0),int> =0>
    __device__ auto loadParticle(size_t id) const //!< default if no attributes are specified, returns a particle with all attributes stored by this buffer
    {
        particleType p;
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
