/*
 * GraSPH2
 * Host_base.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */
#ifndef GRASPH2_HOST_BASE_H
#define GRASPH2_HOST_BASE_H

// includes
//--------------------
#include <mpUtils/mpUtils.h>
#include <mpUtils/mpCuda.h>
#include <thrust/swap.h>
#include "Particle.h"
//--------------------

// forward declarations
//--------------------
template <typename T, typename lsFunctor>
class DEVICE_BASE;
//--------------------

//-------------------------------------------------------------------
/**
 * class template HOST_BASE
 *
 * @brief Class Template that holds particle attributes in host memory
 * @tparam T type of the attribute
 * @tparam lsFunctor a struct that contains the following functions:
 *          - public static Particle< ...>load(const T& v) the attribute value v should be copied into the returned particle.
 *          - public template<typename U> static void store(T & v, const U& p) which should be specialized for different
 *              particle base classes and copy the attribute of the base p into the shared memory position referenced by v
 *          Reference implementations of such structs can be found in Particles.h.
 */
template <typename T, typename lsFunctor>
class HOST_BASE
{
public:
    // constructors and destructor
    HOST_BASE() : m_size(0), m_data(nullptr) {}
    explicit HOST_BASE(size_t n) :  m_size(n), m_data(m_size ? new T[m_size] : nullptr) {}
    ~HOST_BASE() {delete[] m_data;}

    // copy swap idom for copy an move construction and assignment
    HOST_BASE(const HOST_BASE & other) : HOST_BASE(other.m_size) {std::copy(other.m_data,other.m_data+other.m_size,m_data);}
    HOST_BASE( HOST_BASE&& other) noexcept : HOST_BASE() {swap(*this,other);}
    HOST_BASE& operator=(HOST_BASE other) {swap(*this,other); return *this;}
    friend void swap(HOST_BASE & first, HOST_BASE & second)
    {
        using thrust::swap;
        swap(first.m_data,second.m_data);
        swap(first.m_size,second.m_size);
    }

    // particle handling
    template<typename ... Args>
    CUDAHOSTDEV void loadParticle(size_t id, Particle<Args ...> & p) const; //!< get a particle object with the requested members Note: while this is a host device function, calling it on the device will have no effect
    template<typename ... Args>
    CUDAHOSTDEV void storeParticle(size_t id, const Particle<Args ...> & p); //!< set the attributes of particle id according to the particle object  Note: while this is a host device function, calling it on the device will have no effect
    CUDAHOSTDEV void initialize(size_t i); //!< set all values to the default value

    void mapGraphicsResource() {} //!< does nothing, just for compatibility with the particles class
    void unmapGraphicsResource() {} //!< does nothing, just for compatibility with the particles class

    // status checks
    size_t size() const { return m_size;} //!< returns the number of particles

    // friends and types
    template <typename Type, typename Functor> friend class DEVICE_BASE; // be friends with the corresponding device base
    using bind_ref_to_t = DEVICE_BASE<T,lsFunctor>; //!< show to which device_base this can be assigned

protected:
    HOST_BASE & operator=(const size_t & f) {return *this;} //!< ignore assignments of size_t from the base class

public:
    size_t m_size; //!< the number of particles stored in this buffer
    T* m_data;  //!< the actual data
};

//-------------------------------------------------------------------
// function definitions for HOST_BASE class
template<typename T, typename lsFunctor>
template<typename... Args>
void HOST_BASE<T, lsFunctor>::loadParticle(size_t id, Particle<Args ...> &p) const
{
#if !defined(__CUDA_ARCH__)
    p = lsFunctor::load(m_data[id]);
#endif
}

template<typename T, typename lsFunctor>
template<typename... Args>
void HOST_BASE<T, lsFunctor>::storeParticle(size_t id, const Particle<Args ...> &p)
{
#if !defined(__CUDA_ARCH__)
    int i[] = {0, ((void)lsFunctor::template store(m_data[id], ext_base_cast<Args>(p)),1)...};
    (void)i[0]; // silence compiler warning abut i being unused
#endif
}

template<typename T, typename lsFunctor>
void HOST_BASE<T, lsFunctor>::initialize(size_t i)
{
#if !defined(__CUDA_ARCH__)
    m_data[i] = T{lsFunctor::defaultValue};
#endif
}

#endif //GRASPH2_HOST_BASE_H
