/*
 * GraSPH2
 * DeviceParticleBuffer.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */
#ifndef GRASPH2_DEVICEPARTICLEBUFFER_H
#define GRASPH2_DEVICEPARTICLEBUFFER_H

// includes
//--------------------
#include <mpUtils/mpUtils.h>
#include <mpUtils/mpCuda.h>
#include "particle_tmp_utils.h"
#include "Particle.h"
#include "particle_buffer_impl.h"
#include "HostParticleBuffer.h"
//--------------------

//-------------------------------------------------------------------
/**
 * class template ParticleBuffer
 *
 * Manages the storage of a lot of particles in global device memory or host memory.
 *
 * usage:
 * The required number of particles should be passed to the constructor.
 * You can manipulate the particles by calling loadParticle() and storeParticle().
 * Specify what particles attributes you want by setting the corresponding template parameters.
 * To access the particles object on the device define a __global__ use a DeviceParticleReference.
 * You can assign DeviceParticleBuffer to each other as well as to a HostParticleBuffer and vice versa.
 * Which will result in the required memory transfer.
 *
 * openGL interop
 * Use the register function to register an openGL VBO to a specific device base. Then use the map and unmap functions
 * to map the VBOs into cuda address space and make them usable in kernel calls. Make sure the openGL buffer has the
 * correct size. Assignment from or to a host to a mapped device base will copy the data to or from the VBO.
 * Move Assignment of another DeviceParticleBuffer as well as copy assignment from a DeviceParticleBuffer of different size
 * will unregister the VBO.
 *
 */
template <typename ... Args>
class DeviceParticleBuffer : public Args...
{
public:
    static_assert( mpu::conjunction_v< std::is_base_of<device_base_flag,Args>...>,
                   "Only use the DeviceParticleBuffer class with instantiations of Device_base! See file particle_buffer_impl.h for possible bases."); //!< check if only valid bases are used for the particle
    static_assert( checkOrder_v<std::tuple<Args...>,device_base_order>,
                   "Use particle Attributes in correct order without duplicates. See host_base_order in particle_buffer_impl.h.");

    // constructors
    DeviceParticleBuffer() : m_numParticles(0), Args()... {} //!< default constructor
    explicit DeviceParticleBuffer(size_t n) : m_numParticles(n), Args(n)... {} //!< construct particle buffer which can contain n particles

    // conversions between different instantiations of this template
    template <typename... TArgs>
    DeviceParticleBuffer(const DeviceParticleBuffer<TArgs...>& other);  //!< construct this from a particle buffer with different attributes
    template <typename... TArgs>
    DeviceParticleBuffer& operator=(const DeviceParticleBuffer<TArgs...> &b); //!< assignment between particles buffers with different attributes

    // assign and construct from a HostParticleBuffer
    template <typename... TArgs>
    DeviceParticleBuffer(const HostParticleBuffer<TArgs...>& other);  //!< construct this from a HostParticleBuffer
    template <typename... TArgs>
    DeviceParticleBuffer& operator=(const HostParticleBuffer<TArgs...> &b); //!< assignment from a HostParticleBuffer

    // particle handling
    template<typename... particleArgs>
    Particle<particleArgs...> loadParticle(size_t id) const; //!< get a particle object with the requested members
    template<typename... particleArgs>
    void storeParticle(size_t id, const Particle<particleArgs...>& p); //!< set the attributes of particle id according to the particle object
    void initialize(); //!< set all particle attributes to there default values

    // graphics interop
    template <class base>
    void registerGLGraphicsResource(uint32_t resourceID, cudaGraphicsMapFlags flag = cudaGraphicsMapFlagsNone); //!< register an openGL vbo to be used for the particle attribute "base" instead of the internal storage
    void mapGraphicsResource(); //!< if an opengl buffer is used in any base class, map the buffer to enable cuda usage
    void unmapGraphicsResource(); //!< unmap the opengl buffer of all bases so it can be used by openGL again

    // status checks
    size_t size() const {return m_numParticles;} //!< return the number of particles in this buffer
    bool isRegistered() const {return isRegisteredImpl;} //!< check if any internal attribute uses an openGL VBO instead of native cuda memory

private:

    // recursive implementation for isRegistered()
    template <typename first, typename... other, std::enable_if_t< (sizeof...(other)>0), int> _null =0 >
    bool isRegisteredImpl()
    {
        return first::isRegistered || isRegisteredImpl<other...>();
    }

    template <typename first>
    bool isRegisteredImpl()
    {
        return first::isRegistered;
    }

    size_t m_numParticles; //!< the number of particles stored in this buffer
};

//-------------------------------------------------------------------
// helper functions

//namespace detail {
//    template<typename ... Args>
//    __global__ void _initializeParticles()
//    {
//        for(const auto &i : mpu::gridStrideRange(particles.size()))
//        {
//            int t[] = {0, ((void)static_cast<Args*>(&particles)->initialize(i),1)...};
//            (void)t[0]; // silence compiler warning abut t being unused
//        }
//    }
//}

//-------------------------------------------------------------------
// function definitions for ParticleBuffer class

template<typename... Args>
template<typename... TArgs>
DeviceParticleBuffer<Args...>::DeviceParticleBuffer(const DeviceParticleBuffer<TArgs...> &other)
        : m_numParticles(other.size()),
          Args(ext_base_cast<Args>(other))... {}

template<typename... Args>
template<typename... TArgs>
DeviceParticleBuffer<Args...> & DeviceParticleBuffer<Args...>::operator=(const DeviceParticleBuffer<TArgs...> &b)
{
    int t[] = {0, ((void)Args::operator=(ext_base_cast<Args>(b)),1)...};
    (void)t[0]; // silence compiler warning abut t being unused
    return *this;
}

template<typename... Args>
template<typename... TArgs>
DeviceParticleBuffer<Args...>::DeviceParticleBuffer(const HostParticleBuffer<TArgs...> &other)
        : m_numParticles(other.size()),
          Args(ext_base_cast<Args>(other))... {}

template<typename... Args>
template<typename... TArgs>
DeviceParticleBuffer<Args...> &DeviceParticleBuffer<Args...>::operator=(const HostParticleBuffer<TArgs...> &b)
{
    int t[] = {0, ((void)Args::operator=(ext_base_cast<Args>(b)),1)...};
    (void)t[0]; // silence compiler warning abut t being unused
    return *this;
}

template<typename... Args>
template<typename... particleArgs>
Particle<particleArgs...> DeviceParticleBuffer<Args...>::loadParticle(size_t id) const
{
    Particle<particleArgs...> p{};
    int t[] = {0, ((void)Args::loadParticle(id,p),1)...}; // call load particle functions of all the base classes
    (void)t[0]; // silence compiler warning abut t being unused
    return p;
}

template<typename... Args>
template<typename... particleArgs>
void DeviceParticleBuffer<Args...>::storeParticle(size_t id, const Particle<particleArgs...> &p)
{
    int t[] = {0, ((void)Args::storeParticle(id,p),1)...};
    (void)t[0]; // silence compiler warning abut t being unused
}

template<typename... Args>
void DeviceParticleBuffer<Args...>::mapGraphicsResource()
{
    int t[] = {0, ((void)Args::mapGraphicsResource(),1)...};
    (void)t[0]; // silence compiler warning abut t being unused
}

template<typename... Args>
void DeviceParticleBuffer<Args...>::unmapGraphicsResource()
{
    int t[] = {0, ((void)Args::unmapGraphicsResource(),1)...};
    (void)t[0]; // silence compiler warning abut t being unused
}

template<typename... Args>
template<class base>
void DeviceParticleBuffer<Args...>::registerGLGraphicsResource(uint32_t resourceID, cudaGraphicsMapFlags flag)
{
    static_assert( mpu::disjunction_v< std::is_same<base,Args>...> , "The attribute you try to map to openGL is not part of this Buffer!");
    static_cast<base*>(this)->registerGLGraphicsResource(resourceID,flag);
}

template<typename... Args>
void DeviceParticleBuffer<Args...>::initialize()
{
//    detail::_initializeParticles<<<(size()+255)/256, 256>>>(createDeviceCopy());
}

#endif //GRASPH2_DEVICEPARTICLEBUFFER_H
