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
//!< class template that holds particle attributes in device memory
template <typename implementation>
class DEVICE_BASE;

//!< class to identify particle buffer implementations
class pb_impl;
//--------------------

//!< class to identify classes that hold attributes of particles in host memory
class host_base {};

//-------------------------------------------------------------------
/**
 * class template HOST_BASE
 *
 * @brief Class Template that holds particle attributes in host memory
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
class HOST_BASE : host_base
{
public:
    static_assert( std::is_base_of<pb_impl,implementation>::value, "Implementation needs to be a subclass of pb_impl. See particle_buffer_impl.h");

    // types
    using impl = implementation;
    using type = typename impl::type;
    using particleType = typename impl::particleType;
    using bind_ref_to_t = DEVICE_BASE<implementation>; //!< show to which device_base this can be assigned

    // constructors and destructor
    HOST_BASE() : m_size(0), m_data(nullptr) {}
    explicit HOST_BASE(size_t n) :  m_size(n), m_data(m_size ? new type[m_size] : nullptr) {}
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
    void loadParticle(size_t id, Particle<Args ...> & p) const; //!< get a particle object with the requested members Note: while this is a host device function, calling it on the device will have no effect
    template<typename ... Args>
    void storeParticle(size_t id, const Particle<Args ...> & p); //!< set the attributes of particle id according to the particle object  Note: while this is a host device function, calling it on the device will have no effect
    void initialize(size_t i); //!< set all values to the default value

    void mapGraphicsResource() {} //!< does nothing, just for compatibility with the particles class
    void unmapGraphicsResource() {} //!< does nothing, just for compatibility with the particles class

    // status checks
    size_t size() const { return m_size;} //!< returns the number of particles

    // friends
    friend class DEVICE_BASE<implementation>; // be friends with the corresponding device base

protected:
    HOST_BASE & operator=(const size_t & f) {return *this;} //!< ignore assignments of size_t from the base class

private:
    size_t m_size; //!< the number of particles stored in this buffer
    type* m_data;  //!< the actual data
};

//-------------------------------------------------------------------
// function definitions for HOST_BASE class
template<typename implementation>
template<typename... Args>
void HOST_BASE<implementation>::loadParticle(size_t id, Particle<Args ...> &p) const
{
    p = impl::load(m_data[id]);
}

template<typename implementation>
template<typename... Args>
void HOST_BASE<implementation>::storeParticle(size_t id, const Particle<Args ...> &p)
{
    int i[] = {0, ((void)impl::template store(m_data[id], ext_base_cast<Args>(p)),1)...};
    (void)i[0]; // silence compiler warning abut i being unused
}

template<typename implementation>
void HOST_BASE<implementation>::initialize(size_t i)
{
    m_data[i] = type{impl::defaultValue};
}

#endif //GRASPH2_HOST_BASE_H
