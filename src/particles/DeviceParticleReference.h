/*
 * GraSPH2
 * DeviceParticleReference.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */
#ifndef GRASPH2_DEVICEPARTICLEREFERENCE_H
#define GRASPH2_DEVICEPARTICLEREFERENCE_H

// includes
//--------------------
#include <mpUtils/mpUtils.h>
#include <mpUtils/mpCuda.h>
#include "particle_tmp_utils.h"
#include "Particle.h"
#include "particle_buffer_impl.h"
//--------------------

// forward declarations
//--------------------
//!< class template that manages the storage of a lot of particles in device memory.
template <typename ... Args>
class DeviceParticleBuffer;
//--------------------

//-------------------------------------------------------------------
/**
 * class template DeviceParticleReference
 *
 * References a DeviceParticleBuffer to load and store particles inside a global or device function.
 *
 */
template <typename ... Args>
class DeviceParticleReference : public Args...
{
public:
    static_assert( mpu::conjunction_v< std::is_base_of<device_reference_flag,Args>...>,
                   "Only use the DeviceParticleReference class with instantiations of Device_reference! See file particle_buffer_impl.h for possible bases."); //!< check if only valid bases are used for the particle
    static_assert( checkOrder_v<std::tuple<Args...>,dref_base_order>,
                   "Use particle Attributes in correct order without duplicates. See dref_base_order in particle_buffer_impl.h.");

    // construction only from a compatible DeviceParticleBuffer in host code
    template <typename... TArgs>
    DeviceParticleReference(const DeviceParticleBuffer<TArgs...>& other);  //!< construct this from a DeviceParticleBuffer

    // default copy and move construction, no assignment since this is a reference
    template <typename... TArgs>
    CUDAHOSTDEV DeviceParticleReference(const DeviceParticleReference<TArgs...>& other);  //!< construct this from a particle reference with different attributes
    CUDAHOSTDEV DeviceParticleReference(const DeviceParticleReference & other) = default;
    CUDAHOSTDEV DeviceParticleReference( DeviceParticleReference&& other) noexcept = default;
    CUDAHOSTDEV DeviceParticleReference& operator=(DeviceParticleReference&& other) = delete;
    CUDAHOSTDEV DeviceParticleReference& operator=(DeviceParticleReference other) = delete;

    // particle handling
    template<typename... particleArgs>
    CUDAHOSTDEV Particle<particleArgs...> loadParticle(size_t id) const; //!< get a particle object with the requested members
    template<typename... particleArgs>
    CUDAHOSTDEV void storeParticle(size_t id, const Particle<particleArgs...>& p); //!< set the attributes of particle id according to the particle object

    // status checks
    CUDAHOSTDEV size_t size() const {return m_numParticles;} //!< return the number of particles in this buffer

private:
    const size_t m_numParticles; //!< the number of particles stored in this buffer
};

//-------------------------------------------------------------------
// function definitions for ParticleBuffer class

template<typename... Args>
template<typename... TArgs>
DeviceParticleReference<Args...>::DeviceParticleReference(const DeviceParticleBuffer<TArgs...> &other)
        : m_numParticles(other.size()),
          Args(ext_base_cast<Args>(other))... {}

template<typename... Args>
template<typename... TArgs>
CUDAHOSTDEV DeviceParticleReference<Args...>::DeviceParticleReference(const DeviceParticleReference<TArgs...> &other)
        : m_numParticles(other.size()),
          Args(ext_base_cast<Args>(other))... {}

template<typename... Args>
template<typename... particleArgs>
Particle<particleArgs...> DeviceParticleReference<Args...>::loadParticle(size_t id) const
{
    Particle<particleArgs...> p{};
    int t[] = {0, ((void)Args::loadParticle(id,p),1)...}; // call load particle functions of all the base classes
    (void)t[0]; // silence compiler warning abut t being unused
    return p;
}

template<typename... Args>
template<typename... particleArgs>
void DeviceParticleReference<Args...>::storeParticle(size_t id, const Particle<particleArgs...> &p)
{
    int t[] = {0, ((void)Args::storeParticle(id,p),1)...};
    (void)t[0]; // silence compiler warning abut t being unused
}

// include forward declared classes
//--------------------
#include "DeviceParticleBuffer.h"
//--------------------

#endif //GRASPH2_DEVICEPARTICLEREFERENCE_H
