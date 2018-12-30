/*
 * GraSPH2
 * HostParticleBuffer.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */
#ifndef GRASPH2_HOSTPARTICLEBUFFER_H
#define GRASPH2_HOSTPARTICLEBUFFER_H

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
 * class template ParticleBuffer
 *
 * Manages the storage of a lot of particles in host memory.
 *
 * usage:
 * The required number of particles should be passed to the constructor.
 * You can manipulate the particles by calling loadParticle() and storeParticle().
 * Specify what particles attributes you want by setting the corresponding template parameters.
 * See particle_buffer_impl.h for possible attributes.
 *
 * It is possible to use pinned host memory for faster memory transfer. Use the functions pinMemory() and unpinMemory()
 * In case of move assignment or construction the type of memory used (pinned vs non pinned) will
 * be inherited by the new object. In case of copy construction or assignment only data is transferred.
 *
 */
template <typename ... Args>
class HostParticleBuffer : public Args...
{
public:
    static_assert( mpu::conjunction_v< std::is_base_of<host_base_flag,Args>...>,
                   "Only use the HostParticleBuffer class with instantiations of Host_base! See file particle_buffer_impl.h for possible bases."); //!< check if only valid bases are used for the particle
    static_assert( checkOrder_v<std::tuple<Args...>,host_base_order>,
                   "Use particle Attributes in correct order without duplicates. See host_base_order in particle_buffer_impl.h.");

    // types
    using attributes = std::tuple<Args...>;
    using particleType = merge_particles_t < typename Args::particleType ... >;
    using deviceType =  mpu::instantiate_from_tuple_t<DeviceParticleBuffer, reorderd_t< std::tuple<typename Args::deviceType ...>, device_base_order>>;

    // constructors
    HostParticleBuffer() : m_numParticles(0), Args()... {} //!< default constructor
    explicit HostParticleBuffer(size_t n) : m_numParticles(n), Args(n)... {} //!< construct particle buffer which can contain n particles

    // conversions between different instantiations of this template
    template <typename... TArgs>
    HostParticleBuffer(const HostParticleBuffer<TArgs...>& other);  //!< construct this from a particle buffer with different attributes
    template <typename... TArgs>
    HostParticleBuffer& operator=(const HostParticleBuffer<TArgs...> &b); //!< assignment between particles buffers with different attributes

    // assign and construct from a DeviceParticleBuffer
    template <typename... TArgs>
    explicit HostParticleBuffer(const DeviceParticleBuffer<TArgs...>& other);  //!< construct this from a DeviceParticleBuffer
    template <typename... TArgs>
    HostParticleBuffer& operator=(const DeviceParticleBuffer<TArgs...> &b); //!< assignment from a DeviceParticleBuffer

    // particle handling
    template<typename... particleArgs>
    Particle<particleArgs...> loadParticle(size_t id) const; //!< get a particle object with the requested members
    template<typename... particleArgs>
    void storeParticle(size_t id, const Particle<particleArgs...>& p); //!< set the attributes of particle id according to the particle object
    void initialize(); //!< set all particle attributes to there default values

    // using pinned memory
    void pinMemory();   //!< pin the used memory for faster transfer with gpu (takes some time)
    void unpinMemory(); //!< unpin the used memory

    // generate other buffer
    auto getDeviceBuffer(); //!< generates a device particle buffer and copies all data to it

    // status checks
    bool isPinned() const {return std::tuple_element<0,std::tuple<Args...>>::type::isPinned();} //!<  check if pinned memory is used
    size_t size() const {return m_numParticles;} //!< return the number of particles in this buffer

private:
    size_t m_numParticles; //!< the number of particles stored in this buffer
};

//-------------------------------------------------------------------
// function definitions for ParticleBuffer class

template<typename... Args>
template<typename... TArgs>
HostParticleBuffer<Args...>::HostParticleBuffer(const HostParticleBuffer<TArgs...> &other)
        : m_numParticles(other.size()),
          Args(ext_base_cast<Args>(other))... {}

template<typename... Args>
template<typename... TArgs>
HostParticleBuffer<Args...> & HostParticleBuffer<Args...>::operator=(const HostParticleBuffer<TArgs...> &b)
{
    int t[] = {0, ((void)Args::operator=(ext_base_cast<Args>(b)),1)...};
    (void)t[0]; // silence compiler warning abut t being unused
    return *this;
}

template<typename... Args>
template<typename... TArgs>
HostParticleBuffer<Args...>::HostParticleBuffer(const DeviceParticleBuffer<TArgs...> &other)
        : m_numParticles(other.size()),
          Args(ext_base_cast<Args>(other))... {}

template<typename... Args>
template<typename... TArgs>
HostParticleBuffer<Args...> &HostParticleBuffer<Args...>::operator=(const DeviceParticleBuffer<TArgs...> &b)
{
    int t[] = {0, ((void)Args::operator=(ext_base_cast<Args>(b)),1)...};
    (void)t[0]; // silence compiler warning abut t being unused
    return *this;
}

template<typename... Args>
template<typename... particleArgs>
Particle<particleArgs...> HostParticleBuffer<Args...>::loadParticle(size_t id) const
{
    Particle<particleArgs...> p{};
    int t[] = {0, ((void)Args::loadParticle(id,p),1)...}; // call load particle functions of all the base classes
    (void)t[0]; // silence compiler warning abut t being unused
    return p;
}

template<typename... Args>
template<typename... particleArgs>
void HostParticleBuffer<Args...>::storeParticle(size_t id, const Particle<particleArgs...> &p)
{
    int t[] = {0, ((void)Args::storeParticle(id,p),1)...};
    (void)t[0]; // silence compiler warning abut t being unused
}

template<typename... Args>
void HostParticleBuffer<Args...>::initialize()
{
    int t[] = {0, ((void)Args::initialize(),1)...};
    (void)t[0]; // silence compiler warning abut t being unused
}

template<typename... Args>
void HostParticleBuffer<Args...>::pinMemory()
{
    int t[] = {0, ((void)Args::pinMemory(),1)...};
    (void)t[0]; // silence compiler warning abut t being unused
}

template<typename... Args>
void HostParticleBuffer<Args...>::unpinMemory()
{
    int t[] = {0, ((void)Args::unpinMemory(),1)...};
    (void)t[0]; // silence compiler warning abut t being unused
}

template<typename... Args>
auto HostParticleBuffer<Args...>::getDeviceBuffer()
{
    return deviceType(*this);
}

// include forward declared classes
//--------------------
#include "DeviceParticleBuffer.h"
//--------------------

#endif //GRASPH2_HOSTPARTICLEBUFFER_H
