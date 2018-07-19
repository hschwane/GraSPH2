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
#ifndef MPUTILS_GLOBALPARTICLES_H
#define MPUTILS_GLOBALPARTICLES_H

// includes
//--------------------
#include <thrust/swap.h>
#include <mpUtils.h>
#include "ext_base_cast.h"
#include "Particle.h"
//--------------------

//-------------------------------------------------------------------
/**
 * class Particles
 *
 * usage:
 * Manages the storage of a lot of particles in Global Device Memory or Host Memory.
 * you can manipulate the particles by calling loadParticle() and storeParticle()
 *
 *
 */
class Particles
{
public:
    Particles();
    Particles(size_t n);
    ~Particles();

    void free(); //!< free all memory
    void reallocate(size_t n); //!< reallocate all particle data and initialize to zero
    void copyToDevice();    //!< initiate copy of data from host to device
    void copyFromDevice();  //!< initiate copying of data from host to device

    void registerGLPositionBuffer(uint32_t posBufferID); //!< register an openGL buffer to be used for positions (optional)
    void registerGLVelocityBuffer(uint32_t velBufferID); //!< register an openGL buffer to be used for positions (optional)
    void mapRegisteredBuffers();    //!< map the registered buffers before performing any cuda operations
    void unmapRegisteredBuffes();     //!< unmap the registered buffers before performing any openGL operations
    void unregisterBuffers(); //!< unregister all external buffers

//    template<typename... Args>
//    __host__ __device__
//    Particle<Args...> loadParticle(size_t id) const; //!< get a particle object with the requested members
//
//    template<typename... Args>
//    __host__ __device__
//    void storeParticle(const Particle<Args...>& p,size_t id); //!< set the attributes of particle id according to the particle object

    __host__ __device__ size_t size() {return m_numParticles;} //!< return the number of particles

    Particles createDeviceClone() const; //!< create a copy which only holds device pointer and can be moved to the device

private:
    void allocate(size_t n);

    // allow this to be copied to the device without freeing memory on destruction
    bool m_isDeviceCopy;
    size_t m_numParticles;

    // external maped resources
    cudaGraphicsResource* VBO_CUDA[2];
    bool registeredPosBuffer;
    bool registeredVelBuffer;
};

//-------------------------------------------------------------------
// Template class to hold members of the Particles class
template <typename T, typename lsFunctor>
class DEVICE_BASE;

template <typename T, typename lsFunctor>
class HOST_BASE
{
public:
    __host__ HOST_BASE() : m_size(0), m_data(nullptr) {}
    __host__ explicit HOST_BASE(size_t n) :  m_size(n), m_data(m_size ? new T[m_size] : nullptr) {}
    __host__ HOST_BASE(const HOST_BASE & other) : HOST_BASE(other.m_size) {std::copy(other.m_data,other.m_data+other.m_size,m_data);}
    __host__ HOST_BASE( HOST_BASE&& other) noexcept : HOST_BASE() {swap(*this,other);}

    __host__ ~HOST_BASE() {delete[] m_data;}

    __host__ HOST_BASE& operator=(HOST_BASE other) {swap(*this,other); return *this;}

    __host__ friend void swap(HOST_BASE & first, HOST_BASE & second)
    {
        using thrust::swap;
        swap(first.m_data,second.m_data);
        swap(first.m_size,second.m_size);
    }

    template<typename ... Args>
    __host__ void loadParticle(size_t id, Particle<Args ...> & p) {p = lsFunctor::load(m_data[id]);}
    template<typename ... Args>
    __host__ void storeParticle(size_t id, const Particle<Args ...> & p) {int i[] = {0, ((void)lsFunctor::template store(m_data[id], ext_base_cast<Args>(p)),1)...};}

    template <typename Type, typename Functor> friend class DEVICE_BASE;
    using bind_ref_to_t = DEVICE_BASE<T,lsFunctor>;
private:
    size_t m_size;
    T* m_data;
};

template <typename T, typename lsFunctor>
class DEVICE_BASE
{
public:
    using host_type = HOST_BASE<T,lsFunctor>;

    CUDAHOSTDEV DEVICE_BASE() : m_size(0), m_data(nullptr), m_isDeviceCopy(false) {}
    __host__ explicit DEVICE_BASE(size_t n) :  m_size(n), m_data(nullptr), m_isDeviceCopy(false) {assert_cuda(cudaMalloc(&m_data, m_size*sizeof(T)));}
    __host__ DEVICE_BASE(const DEVICE_BASE & other) : DEVICE_BASE(other.m_size)
    {
        assert_cuda( cudaMemcpy(m_data, other.m_data, m_size, cudaMemcpyDeviceToDevice));
    }

    CUDAHOSTDEV DEVICE_BASE( DEVICE_BASE&& other) noexcept : DEVICE_BASE() {swap(*this,other);}

    ~DEVICE_BASE() {if(!m_isDeviceCopy) assert_cuda(cudaFree(m_data));}

    CUDAHOSTDEV DEVICE_BASE& operator=(DEVICE_BASE other) {swap(*this,other); return *this;}

    __host__ DEVICE_BASE(const host_type & other) : DEVICE_BASE(other.m_size)
    {
        assert_cuda( cudaMemcpy(m_data, other.m_data, m_size, cudaMemcpyHostToDevice));
    }

    __host__ operator host_type()
    {
        host_type host(m_size);
        logDEBUG("PARTICLES") << m_size;
        logDEBUG("PARTICLES") << host.m_size;
        logDEBUG("PARTICLES") << host.m_data;
        logDEBUG("PARTICLES") << m_data;
        assert_cuda( cudaMemcpy(host.m_data,m_data,m_size,cudaMemcpyDeviceToHost));
        return host;
    }

    CUDAHOSTDEV friend void swap(DEVICE_BASE & first, DEVICE_BASE & second)
    {
        using thrust::swap;
        swap(first.m_data,second.m_data);
        swap(first.m_size,second.m_size);
        swap(first.m_isDeviceCopy,second.m_isDeviceCopy);
    }

    __host__ DEVICE_BASE createDeviceCopy() {DEVICE_BASE b; b.m_data=m_data; b.m_size = m_size; b.m_isDeviceCopy=true; return std::move(b);}; //!< shallow copy to be moved to the device

    template<typename ... Args>
    __device__ void loadParticle(size_t id, Particle<Args ...> & p) {p = lsFunctor::load(m_data[id]);}
    template<typename ... Args>
    __device__ void storeParticle(size_t id, const Particle<Args ...> & p) {int i[] = {0, ((void)lsFunctor::template store(m_data[id], ext_base_cast<Args>(p)),1)...};}

    using bind_ref_to_t = host_type;
private:
    bool m_isDeviceCopy;
    size_t m_size;
    T* m_data;
};

#endif //MPUTILS_GLOBALPARTICLES_H
