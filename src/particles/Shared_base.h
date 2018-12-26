/*
 * GraSPH2
 * Shared_base.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */
#ifndef GRASPH2_SHARED_BASE_H
#define GRASPH2_SHARED_BASE_H

// includes
//--------------------
#include <mpUtils/mpUtils.h>
#include <mpUtils/mpCuda.h>
#include "Particle.h"
//--------------------

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

#endif //GRASPH2_SHARED_BASE_H
