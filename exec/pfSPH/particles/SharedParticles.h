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
#include <mpUtils.h>
#include "ext_base_cast.h"
#include "Particle.h"
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

//    __device__ void copyFromGlobal(size_t shared_id, size_t global_id, const Particles& global); //!< load particle from global to shared memory
//    __device__ void copyToGlobal(size_t shared_id, size_t global_id, const Particles& global); //!< store a particle in global memory

    template<typename... particleArgs>
    __device__
            Particle<particleArgs...> loadParticle(size_t id) //!< get a particle object with the requested members
    {
        Particle<particleArgs...> p{};
        int t[] = {0, ((void)TArgs<n>::loadParticle(id,p),1)...}; // call load particle functions of all the base classes
        return p;
    }

    template<typename... particleArgs>
    __device__
    void storeParticle(const Particle<particleArgs...>& p,size_t id) //!< set the attributes of particle id according to the particle object
    {
        int t[] = {0, ((void)TArgs<n>::storeParticle(id,p),1)...};
    }

    __device__ size_t size() {return n;}
};

//-------------------------------------------------------------------
/**
 * @brief Class Template that holds particle attributes in shared memory
 * @tparam n number of particles to store
 * @tparam T type of the attribute
 * @tparam lsFunctor a struct that contains the following functions:
 *          - public static Particle< ...>load(const T& v) the attribute value v should be copied into the returned particle.
 *          - public template<typename U> static void store(T & v, const U& p) which should be specialized for different
 *              particle base classes and copy the attribute of the base p into the shared memory position referenced by v
 *          Reference implementations of such structs can be found in Particles.h.
 */
template <size_t n, typename T, typename lsFunctor>
class SHARED_BASE
{
public:
    __device__ SHARED_BASE() { __shared__ static T mem[n]; m_data = mem;}

    template<typename ... Args>
    __device__ void loadParticle(size_t id, Particle<Args ...> & p) {p = lsFunctor::load(m_data[id]);}
    template<typename ... Args>
    __device__ void storeParticle(size_t id, const Particle<Args ...> & p)
    {
        int i[] = {0, ((void)lsFunctor::template store(m_data[id], ext_base_cast<Args>(p)),1)...};
        (void)i[0]; // silence compiler warning abut i being unused
    }

private:
    T * m_data;
};

#endif //MPUTILS_SHAREDPARTICLES_H
