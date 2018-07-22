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
template <typename ... Args>
class Particles : public Args...
{
public:
    CUDAHOSTDEV Particles() : m_numParticles(0), m_isDeviceCopy(false), Args(0)... {}
    explicit __host__ Particles(size_t n) : m_numParticles(0), m_isDeviceCopy(false), Args(n)... {}

    template <typename... TArgs>
    __host__ Particles(const Particles<TArgs...>& other)
            : m_numParticles(other.size()),
              m_isDeviceCopy(other.isDeviceCopy()),
              Args(ext_base_cast<Args>(other))... {}

    template <typename... TArgs>
    __host__ Particles& operator=(const Particles<TArgs...> &b)
    {
        int t[] = {0, ((void)Args::operator=(ext_base_cast<Args>(b)),1)...};
        (void)t[0]; // silence compiler warning abut t being unused
        return *this;
    }

    template<typename... particleArgs>
    CUDAHOSTDEV
    Particle<particleArgs...> loadParticle(size_t id) const //!< get a particle object with the requested members
    {
        Particle<particleArgs...> p{};
        int t[] = {0, ((void)Args::loadParticle(id,p),1)...}; // call load particle functions of all the base classes
        (void)t[0]; // silence compiler warning abut t being unused
        return p;
    }

    template<typename... particleArgs>
    CUDAHOSTDEV
    void storeParticle(size_t id, const Particle<particleArgs...>& p) //!< set the attributes of particle id according to the particle object
    {
        int t[] = {0, ((void)Args::storeParticle(id,p),1)...};
        (void)t[0]; // silence compiler warning abut t being unused
    }

    CUDAHOSTDEV size_t size() const {return m_numParticles;} //!< return the number of particles
    CUDAHOSTDEV bool isDeviceCopy() const {return m_isDeviceCopy;} //!< check if the particle object is a shallow copy only containing device pointers

    CUDAHOSTDEV Particles createDeviceCopy() const
    {
        return Particles(*this,true);
    }

private:
    CUDAHOSTDEV Particles(const Particles& other, bool devCopy)
            : m_numParticles(other.m_numParticles),
              m_isDeviceCopy(true),
              Args(ext_base_cast<Args>(other).createDeviceCopy())... {}

    // allow this to be copied to the device without freeing memory on destruction
    bool m_isDeviceCopy;
    size_t m_numParticles;
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
    __host__ void loadParticle(size_t id, Particle<Args ...> & p) const {p = lsFunctor::load(m_data[id]);}
    template<typename ... Args>
    __host__ void storeParticle(size_t id, const Particle<Args ...> & p) {int i[] = {0, ((void)lsFunctor::template store(m_data[id], ext_base_cast<Args>(p)),1)...};}

    __host__ size_t size() { return m_size;}

    template <typename Type, typename Functor> friend class DEVICE_BASE;
    using bind_ref_to_t = DEVICE_BASE<T,lsFunctor>;

protected:
    __host__ HOST_BASE & operator=(const size_t & f) {return *this;}

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

    __host__ operator host_type() const
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

    __host__ DEVICE_BASE createDeviceCopy() const {DEVICE_BASE b; b.m_data=m_data; b.m_size = m_size; b.m_isDeviceCopy=true; return std::move(b);}; //!< shallow copy to be moved to the device

    template<typename ... Args>
    __device__ void loadParticle(size_t id, Particle<Args ...> & p) const {p = lsFunctor::load(m_data[id]);}
    template<typename ... Args>
    __device__ void storeParticle(size_t id, const Particle<Args ...> & p) {int i[] = {0, ((void)lsFunctor::template store(m_data[id], ext_base_cast<Args>(p)),1)...};}

    CUDAHOSTDEV size_t size() { return m_size;}

    using bind_ref_to_t = host_type;

protected:
    __host__ DEVICE_BASE & operator=(const size_t & f) {return *this;}
private:
    bool m_isDeviceCopy;
    size_t m_size;
    T* m_data;
};

#endif //MPUTILS_GLOBALPARTICLES_H
