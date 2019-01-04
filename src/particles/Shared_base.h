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

// forward declarations
//--------------------
//!< class to identify particle buffer implementations
class pb_impl;
//--------------------

//!< class to identify classes that hold attributes of particles in shared memory
class Shared_base {};

//-------------------------------------------------------------------
/**
 * class template SHARED_BASE
 *
 * @brief Class Template that holds particle attributes in shared memory
 * @tparam n number of particles to store
 * @tparam implementation a struct that contains the following functions and typedefs:
 *          - using type = the type of the internal data
 *          - static constexpr type defaultValue = the default value for that particle buffer
 *          - using particleType = the type of particle that can be loaded from or stored in this buffer
 *          - public static Particle< ...>load(const T& v) the attribute value v should be copied into the returned particle.
 *          - public template<typename U> static void store(T & v, const U& p) which should be specialized for different
 *              particle base classes and copy the attribute of the base p into the shared memory position referenced by v
 *          Reference implementations of such structs can be found in particle_buffer_impl.h.
 */
template <size_t n, typename implementation>
class SHARED_BASE : Shared_base
{
public:
    static_assert( std::is_base_of<pb_impl,implementation>::value, "Implementation needs to be a subclass of pb_impl. See particle_buffer_impl.h");
    using impl = implementation;
    using type = typename impl::type;
    using particleType = typename impl::particleType;

    __device__ SHARED_BASE() { __shared__ static type mem[n]; m_data = mem;}

    template<typename ... Args>
    __device__ void loadParticle(size_t id, Particle<Args ...> & p) const {p = impl::load(m_data[id]);}
    template<typename ... Args>
    __device__ void storeParticle(size_t id, const Particle<Args ...> & p)
    {
        int i[] = {0, ((void)impl::template store(m_data[id], ext_base_cast<Args>(p)),1)...};
        (void)i[0]; // silence compiler warning abut i being unused
    }

private:
    type * m_data;
};

// include forward declared classes
//--------------------
#include "particle_buffer_impl.h"
//--------------------

#endif //GRASPH2_SHARED_BASE_H
