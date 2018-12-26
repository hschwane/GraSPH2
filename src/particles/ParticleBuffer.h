/*
 * mpUtils
 * GlobalParticles.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */
#ifndef GRASPH2_PARTICLE_BUFFER_H
#define GRASPH2_PARTICLE_BUFFER_H

// includes
//--------------------
#include <thrust/swap.h>
#include <tuple>
#include <cuda_gl_interop.h>
#include <mpUtils/mpUtils.h>
#include <mpUtils/mpCuda.h>
#include "particle_tmp_utils.h"
#include "Particle.h"
//--------------------

//-------------------------------------------------------------------
/**
 * class template ParticleBuffer
 *
 * Manages the storage of a lot of particles in Global Device Memory or Host Memory.
 *
 * usage:
 * The required number of particles should be passed to the constructor.
 * You can manipulate the particles by calling loadParticle() and storeParticle(). Only attributes from HOST bases will be returned
 * when calling from host code and DEV bases when calling from device code.
 * Specify what particles attributes you want by setting the corresponding template parameters.
 * To access the particles object on the device define a __global__ function that takes a Particles object which has only DEV bases by Value.
 * Then call the function and pass a shallow copy created with the createDeviceCopy() function.
 * If you assign a Particles object with HOST based to another one with DEV bases or vice versa data will be copied automatically
 * between host and device using. Data copying will also work for HOST-HOST and DEV-DEV assignment.
 *
 * Do NOT mix host and device bases.
 *
 * openGL interop
 * Use the register function to register an openGL VBO to a specific device base. Then use the map and unmap functions
 * to map the VBOs into cuda address space and make them usable. In kernel calls. Make sure the openGL buffer has the
 * correct size. Assignment from a host to a mapped device base will copy the data to the VBO, assignment from another
 * device will however unmap the the VBO and allocate new cuda memory.
 *
 * Supported particle attributes are:
 * HOST_POSM
 * HOST_VEL
 * HOST_ACC
 *
 * DEV_POSM
 * DEV_VEL
 * DEV_ACC
 *
 */
template <typename ... Args>
class ParticleBuffer : public Args...
{
public:
    // constructors
    CUDAHOSTDEV ParticleBuffer() : m_numParticles(0), m_isDeviceCopy(false), Args()... {} //!< default constructor
    explicit ParticleBuffer(size_t n) : m_numParticles(n), m_isDeviceCopy(false), Args(n)... {} //!< construct particle buffer which can contain n particles

    // shallow copying
    CUDAHOSTDEV auto createDeviceCopy() const; //!< creates a shallow copy which only include device bases to be used on the device

    // conversions
    template <typename... TArgs>
    ParticleBuffer(const ParticleBuffer<TArgs...>& other);  //!< construct this from a particle buffer with different attributes
    template <typename... TArgs>
    ParticleBuffer& operator=(const ParticleBuffer<TArgs...> &b); //!< assignment between particles buffers with different atributes

    // particle handling
    template<typename... particleArgs>
    CUDAHOSTDEV Particle<particleArgs...> loadParticle(size_t id) const; //!< get a particle object with the requested members
    template<typename... particleArgs>
    CUDAHOSTDEV void storeParticle(size_t id, const Particle<particleArgs...>& p); //!< set the attributes of particle id according to the particle object
    void initialize(); //!< set all particle attributes to there default values

    // graphics interop
    template <class base>
    void registerGLGraphicsResource(uint32_t resourceID, cudaGraphicsMapFlags flag = cudaGraphicsMapFlagsNone); //!< register an openGL vbo to be used for the particle attribute "base" instead of the internal storage
    void mapGraphicsResource(); //!< if an opengl buffer is used in any base class, map the buffer to enable cuda usage
    void unmapGraphicsResource(); //!< unmap the opengl buffer of all bases so it can be used by openGL again

    // status checks
    CUDAHOSTDEV size_t size() const {return m_numParticles;} //!< return the number of particles in this buffer
    CUDAHOSTDEV bool isDeviceCopy() const {return m_isDeviceCopy;} //!< check if the particle object is a shallow copy only containing device pointers

private:
    CUDAHOSTDEV ParticleBuffer(const ParticleBuffer& other, bool devCopy) //!< copy constructor that constructs a Particles object by making shallow copies of the bases
            : m_numParticles(other.m_numParticles),
              m_isDeviceCopy(true),
              Args(ext_base_cast<Args>(other).createDeviceCopy())... {}

    template <class T>
    using create_dev_copy_t = decltype(std::declval<T>().createDeviceCopy()); //!< helper for copy condition

    template <class T>
    using copy_condition = mpu::is_detected<create_dev_copy_t,T>; //!< check if T has member "createDeviceCopy"

    template <class T>
    using inv_copy_condition = mpu::negation<mpu::is_detected<create_dev_copy_t,T>>; //!< check if T has no member "createDeviceCopy"

    template <typename...Ts>
    ParticleBuffer<Ts...> deviceCopyHelper(std::tuple<Ts...>&&) const; //!< returns a particles object hat was constructed using Particles<Ts...>(*this,true)

    bool m_isDeviceCopy; //!< if this is a device copy no memory will be freed on destruction
    size_t m_numParticles; //!< the number of particles stored in this buffer
};

//-------------------------------------------------------------------
// helper functions

namespace detail {
    template<typename ... Args>
    __global__ void _initializeParticles(ParticleBuffer<Args...> particles)
    {
        for(const auto &i : mpu::gridStrideRange(particles.size()))
        {
            int t[] = {0, ((void)static_cast<Args*>(&particles)->initialize(i),1)...};
            (void)t[0]; // silence compiler warning abut t being unused
        }
    }
}

//-------------------------------------------------------------------
// function definitions for ParticleBuffer class

template<typename... Args>
template<typename... TArgs>
ParticleBuffer<Args...>::ParticleBuffer(const ParticleBuffer<TArgs...> &other)
                                                : m_numParticles(other.size()),
                                                  m_isDeviceCopy(other.isDeviceCopy()),
                                                  Args(ext_base_cast<Args>(other))... {}

template<typename... Args>
template<typename... TArgs>
ParticleBuffer<Args...> & ParticleBuffer<Args...>::operator=(const ParticleBuffer<TArgs...> &b)
{
    int t[] = {0, ((void)Args::operator=(ext_base_cast<Args>(b)),1)...};
    (void)t[0]; // silence compiler warning abut t being unused
    return *this;
}

template<typename... Args>
template<typename... particleArgs>
Particle<particleArgs...> ParticleBuffer<Args...>::loadParticle(size_t id) const
{
    Particle<particleArgs...> p{};
    int t[] = {0, ((void)Args::loadParticle(id,p),1)...}; // call load particle functions of all the base classes
    (void)t[0]; // silence compiler warning abut t being unused
    return p;
}

template<typename... Args>
template<typename... particleArgs>
void ParticleBuffer<Args...>::storeParticle(size_t id, const Particle<particleArgs...> &p)
{
    int t[] = {0, ((void)Args::storeParticle(id,p),1)...};
    (void)t[0]; // silence compiler warning abut t being unused
}

template<typename... Args>
template <typename...Ts>
ParticleBuffer<Ts...> ParticleBuffer<Args...>::deviceCopyHelper(std::tuple<Ts...>&&) const
{
    static_assert(sizeof...(Ts)>0,"You can not create a device copy if there is no device base.");
    return ParticleBuffer<Ts...>(*this,true);
};

template<typename... Args>
auto ParticleBuffer<Args...>::createDeviceCopy() const
{
#if 1 // defined(__CUDA_ARCH__)
    return ParticleBuffer<Args...>(*this,true);
#else
    return deviceCopyHelper(mpu::remove_t<copy_condition,Args...>());
#endif
}

template<typename... Args>
void ParticleBuffer<Args...>::mapGraphicsResource()
{
    int t[] = {0, ((void)Args::mapGraphicsResource(),1)...};
    (void)t[0]; // silence compiler warning abut t being unused
}

template<typename... Args>
void ParticleBuffer<Args...>::unmapGraphicsResource()
{
    int t[] = {0, ((void)Args::unmapGraphicsResource(),1)...};
    (void)t[0]; // silence compiler warning abut t being unused
}

template<typename... Args>
template<class base>
void ParticleBuffer<Args...>::registerGLGraphicsResource(uint32_t resourceID, cudaGraphicsMapFlags flag)
{
    static_cast<base*>(this)->registerGLGraphicsResource(resourceID,flag);
}

template<typename... Args>
void ParticleBuffer<Args...>::initialize()
{
    for(int i = 0; i < size(); ++i)
    {
        int t[] = {0, ((void)Args::initialize(i),1)...};
        (void)t[0]; // silence compiler warning abut t being unused
    }

    detail::_initializeParticles<<<(size()+255)/256, 256>>>(createDeviceCopy());
}

#endif //GRASPH2_PARTICLE_BUFFER_H
