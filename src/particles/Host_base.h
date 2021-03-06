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
class host_base_flag {};

//-------------------------------------------------------------------
/**
 * class template HOST_BASE
 *
 * @brief Class Template that holds particle attributes in host memory
 *        You can use pinned host memory by calling pinMemory().
 *        In case of move assignment or construction the type of memory used (pinned vs non pinned) will
 *        be inherited by the new object. In case of copy construction or assignment from a buffer of the same size,
 *        only data is transferred.
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
class HOST_BASE : host_base_flag
{
public:
    static_assert( std::is_base_of<pb_impl,implementation>::value, "Implementation needs to be a subclass of pb_impl. See particle_buffer_impl.h");

    // types
    using impl = implementation;
    using type = typename impl::type;
    using particleType = typename impl::particleType;
    using deviceType = DEVICE_BASE<implementation>;
    using bind_ref_to_t = deviceType; //!< show to which device_base this can be assigned

    // constructors and destructor
    HOST_BASE();
    explicit HOST_BASE(size_t n);
    ~HOST_BASE();

    // copy swap idom for copy an move construction and move assignment
    HOST_BASE(const HOST_BASE & other, bool pin = false);
    HOST_BASE( HOST_BASE&& other) noexcept : HOST_BASE() {swap(*this,other);}
    HOST_BASE& operator=(HOST_BASE&& other) noexcept {swap(*this,other); return *this;}
    HOST_BASE& operator=(const HOST_BASE& other); // do not use swap here since that would screw up pinned memory
    friend void swap(HOST_BASE & first, HOST_BASE & second) noexcept
    {
        using thrust::swap;
        swap(first.m_data,second.m_data);
        swap(first.m_size,second.m_size);
        swap(first.m_isPinned,second.m_isPinned);
    }

    // construction and assignment from device_base
    explicit HOST_BASE(const deviceType & other, bool pin = false); //!< construct from a compatible device base
    HOST_BASE& operator=(const deviceType& other); //!< assign from compatible device base

    // particle handling
    template<typename ... Args>
    void loadParticle(size_t id, Particle<Args ...> & p) const; //!< get a particle object with the requested members Note: while this is a host device function, calling it on the device will have no effect
    template<typename ... Args>
    void storeParticle(size_t id, const Particle<Args ...> & p); //!< set the attributes of particle id according to the particle object  Note: while this is a host device function, calling it on the device will have no effect
    void initialize(); //!< set all values to the default value

    // use pinned memory
    void pinMemory();   //!< pin the used memory for faster transfer with gpu (takes some time)
    void unpinMemory(); //!< unpin the used memory

    // status checks
    size_t size() const { return m_size;} //!< returns the number of particles
    bool isPinned() const {return m_isPinned;} //!<  check if pinned memory is used

    // friends
    friend class DEVICE_BASE<implementation>; // be friends with the corresponding device base

protected:
    HOST_BASE & operator=(const size_t & f) {return *this;} //!< ignore assignments of size_t from the base class

private:
    size_t m_size; //!< the number of particles stored in this buffer
    type* m_data;  //!< the actual data
    bool m_isPinned; //!< whether or not pinned memory is used
};

//-------------------------------------------------------------------
// function definitions for HOST_BASE class
template<typename implementation>
HOST_BASE<implementation>::HOST_BASE() : m_size(0), m_data(nullptr), m_isPinned(false)
{
}

template<typename implementation>
HOST_BASE<implementation>::HOST_BASE(size_t n) :  m_size(n), m_data(m_size ? new type[m_size] : nullptr), m_isPinned(false)
{
}

template<typename implementation>
HOST_BASE<implementation>::~HOST_BASE()
{
    if(m_isPinned)
        unpinMemory();
    delete[] m_data;
}

template<typename implementation>
HOST_BASE<implementation>::HOST_BASE(const HOST_BASE &other, bool pin) : HOST_BASE(other.m_size)
{
    if(pin)
        pinMemory();
    std::copy(other.m_data,other.m_data+other.m_size,m_data);
}

template<typename implementation>
HOST_BASE<implementation> &HOST_BASE<implementation>::operator=(const HOST_BASE& other)
{
    if(&other != this)
    {
        if(other.size() == m_size)
            std::copy(other.m_data,other.m_data+other.m_size,m_data);
        else
        {
            HOST_BASE copy(other);
            swap(*this,copy);
        }
    }
    return *this;
}

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
void HOST_BASE<implementation>::initialize()
{
    #pragma omp parallel for
    for(int i = 0; i < m_size; i++)
        m_data[i] = type{impl::defaultValue};
}

template<typename implementation>
void HOST_BASE<implementation>::pinMemory()
{
    assert_cuda(cudaHostRegister(m_data, m_size * sizeof(type), cudaHostRegisterDefault));
    m_isPinned = true;
}

template<typename implementation>
void HOST_BASE<implementation>::unpinMemory()
{
    assert_cuda(cudaHostUnregister(m_data));
    m_isPinned = false;
}

template<typename implementation>
HOST_BASE<implementation>::HOST_BASE(const HOST_BASE::deviceType &other, bool pin) : HOST_BASE(other.m_size)
{
    if(pin)
        pinMemory();
    assert_cuda( cudaMemcpy(m_data, other.m_data, m_size*sizeof(type), cudaMemcpyDeviceToHost));
}

template<typename implementation>
HOST_BASE<implementation> &HOST_BASE<implementation>::operator=(const HOST_BASE::deviceType &other)
{
    if(size() != other.size())
    {
        HOST_BASE<implementation> base(other);
        swap(*this,base);
    }
    else
    {
        assert_cuda( cudaMemcpy(m_data, other.m_data, m_size*sizeof(type), cudaMemcpyDeviceToHost));
    }
    return *this;
}

// include forward declared classes
//--------------------
#include "Device_base.h"
#include "particle_buffer_impl.h"
//--------------------

#endif //GRASPH2_HOST_BASE_H
