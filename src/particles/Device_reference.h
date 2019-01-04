/*
 * GraSPH2
 * Device_reference.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */
#ifndef GRASPH2_DEVICE_REFERENCE_H
#define GRASPH2_DEVICE_REFERENCE_H


// includes
//--------------------
#include <mpUtils/mpUtils.h>
#include <mpUtils/mpCuda.h>
#include <thrust/swap.h>
#include "Particle.h"
#include "Device_base.h"
//--------------------

//!< class to identify classes that hold attributes of particles in host memory
class device_reference_flag {};

//-------------------------------------------------------------------
/**
 * class template DEVICE_REFERENCE
 *
 * @brief A class template that references the data stored in a DEVICE_BASE object and allows to load and store Particles from device code.
 *          Simply construct a DEVICE_REFERENCE from your DEVICE_BASE and pass it to the kernel. Best used inside a DeviceParticleReference.
 * @tparam implementation a struct that contains the following functions and typedefs:
 *          - using type = the type of the internal data
 *          - static constexpr type defaultValue = the default value for that particle buffer
 *          - using particleType = the type of particle that can be loaded from or stored in this buffer
 *          - public static Particle< ...>load(const T& v) the attribute value v should be copied into the returned particle.
 *          - public template<typename U> static void store(T & v, const U& p) which should be specialized for different
 *              particle base classes and copy the attribute of the base p into the shared memory position referenced by v
 *          Reference implementations of such structs can be found in particle_buffer_impl.h.
 */
template <typename implementation>
class DEVICE_REFERENCE : device_reference_flag
{
public:
    static_assert( std::is_base_of<pb_impl,implementation>::value, "Implementation needs to be a subclass of pb_impl. See particle_buffer_impl.h");

    template<typename nbft, std::enable_if_t< std::is_same<nbft,no_baseclass_flag>::value || std::is_same<nbft,size_t>::value, int> _null = 0 >
    CUDAHOSTDEV explicit DEVICE_REFERENCE(nbft v) { static_assert(mpu::always_false_v<nbft>,
                      "Cannot construct DeviceParticleReference from this DeviceParticleReference or DeviceParticleBuffer, because the latter is missing at least one attribute!"); }

    // types
    using impl = implementation;
    using type = typename impl::type;
    using particleType = typename impl::particleType;
    using deviceType = DEVICE_BASE<implementation>;
    using bind_ref_to_t = deviceType;

    // construction only from a compatible device base in host code
    DEVICE_REFERENCE(const deviceType & other); //!< construct from a compatible device base

    // default copy and move construction, no assignment since this is a reference
    CUDAHOSTDEV DEVICE_REFERENCE(const DEVICE_REFERENCE & other)= default;
    CUDAHOSTDEV DEVICE_REFERENCE( DEVICE_REFERENCE&& other) noexcept = default;
    CUDAHOSTDEV DEVICE_REFERENCE& operator=(DEVICE_REFERENCE&& other) = delete;
    CUDAHOSTDEV DEVICE_REFERENCE& operator=(DEVICE_REFERENCE other) = delete;

    // particle handling
    template<typename ... Args>
    __device__ void loadParticle(size_t id, Particle<Args ...> & p) const; //!< get a particle object with the requested members
    template<typename ... Args>
    __device__ void storeParticle(size_t id, const Particle<Args ...> & p); //!< set the attributes of particle id according to the particle object

    __device__ void initialize(size_t id); //!< initialises the particle ad position id

public:

    type* const m_data; //!< pointer to the actual data
};

//-------------------------------------------------------------------
// function definitions for DEVICE_BASE class

template<typename implementation>
DEVICE_REFERENCE<implementation>::DEVICE_REFERENCE(const DEVICE_REFERENCE::deviceType &other) : m_data(other.m_data)
{
}

template <typename implementation>
template<typename... Args>
__device__ void DEVICE_REFERENCE<implementation>::loadParticle(size_t id, Particle<Args ...> &p) const
{
    p = impl::load(m_data[id]);
}

template <typename implementation>
template<typename... Args>
__device__ void DEVICE_REFERENCE<implementation>::storeParticle(size_t id, const Particle<Args ...> &p)
{
    int i[] = {0, ((void)impl::template store(m_data[id], ext_base_cast<Args>(p)),1)...};
    (void)i[0]; // silence compiler warning abut i being unused
}

template<typename implementation>
__device__ void DEVICE_REFERENCE<implementation>::initialize(size_t id)
{
    m_data[id] = type{impl::defaultValue};
}

#endif //GRASPH2_DEVICE_REFERENCE_H
