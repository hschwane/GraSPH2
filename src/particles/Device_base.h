/*
 * GraSPH2
 * Device_base.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */
#ifndef GRASPH2_DEVICE_BASE_H
#define GRASPH2_DEVICE_BASE_H

// includes
//--------------------
#include <mpUtils/mpUtils.h>
#include <mpUtils/mpCuda.h>
#include <thrust/swap.h>
#include "Particle.h"
#include "Host_base.h"
//--------------------

//!< class to identify classes that hold attributes of particles in host memory
class device_base {};

//-------------------------------------------------------------------
/**
 * class template DEVICE_BASE
 *
 * @brief Class Template that holds particle attributes in device memory. You can use an openGL Buffer instead of the
 *              managed memory by casting the Particles object to the specific device base and calling the registerGLGraphics Resource.
 *              You can then use the map and unmap functions of the particles class to map or unmap the openGL Buffers of all its device bases.
 *              The resource will be automatically unregistered in the destructer. To unregister maually cast the Particles object to the base type
 *              and assign a new base object.
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
class DEVICE_BASE : device_base
{
public:
    static_assert( std::is_base_of<pb_impl,implementation>::value, "Implementation needs to be a subclass of pb_impl. See particle_buffer_impl.h");

    // types
    using impl = implementation;
    using type = typename impl::type;
    using particleType = typename impl::particleType;
    using host_type = HOST_BASE<implementation>; //!< the type of host base this device base can be converted to
    using bind_ref_to_t = host_type; //!< the type of host base this device base can be converted to

    //construction and destruction
    DEVICE_BASE() : m_size(0), m_data(nullptr), m_graphicsResource(nullptr) {}
    explicit DEVICE_BASE(size_t n);
    ~DEVICE_BASE();

    // copy swap idom where copy construction is only allowed on th host
    // to copy on the device use device copy
    DEVICE_BASE(const DEVICE_BASE & other);
    DEVICE_BASE( DEVICE_BASE&& other) noexcept : DEVICE_BASE() {swap(*this,other);}
    DEVICE_BASE& operator=(DEVICE_BASE&& other) noexcept {swap(*this,other); return *this;}
    DEVICE_BASE& operator=(DEVICE_BASE other) noexcept {swap(*this,other); return *this;}
    friend void swap(DEVICE_BASE & first, DEVICE_BASE & second) noexcept
    {
        using thrust::swap;
        swap(first.m_data,second.m_data);
        swap(first.m_size,second.m_size);
        swap(first.m_graphicsResource,second.m_graphicsResource);
    }

    // converting from and to a compatible host base
    DEVICE_BASE(const host_type & other); //!< construct from a compatible host base
    DEVICE_BASE& operator=(const host_type& other); //!< assign from compatible host base
    operator host_type() const; //!< convert to a compatible host base

    // mapping of graphics resources
    void registerGLGraphicsResource(uint32_t resourceID, cudaGraphicsMapFlags flag = cudaGraphicsMapFlagsNone); //!< register an openGL Buffer to be used instead of the allocated memory
    void mapGraphicsResource(); //!< if an opengl buffer is used map the buffer to enable cuda usage
    void unmapGraphicsResource(); //!< unmap the opengl buffer so it can be used by openGL again

    // particle handling
    template<typename ... Args>
    void loadParticle(size_t id, Particle<Args ...> & p) const; //!< get a particle object with the requested members
    template<typename ... Args>
    void storeParticle(size_t id, const Particle<Args ...> & p); //!< set the attributes of particle id according to the particle object

    // status checks
    size_t size() { return m_size;} //!< returns the number of particles

protected:
    DEVICE_BASE & operator=(const size_t & f) {return *this;} //!< ignore assignments of size_t from the base class

private:
    void unregisterGraphicsResource(); //!< unregister the opengl resource from cuda

    size_t m_size; //!< the umber of particles
    type* m_data; //!< the actual data
    cudaGraphicsResource* m_graphicsResource; //!< a cuda graphics resource that can be used as memory
};

//-------------------------------------------------------------------
// function definitions for DEVICE_BASE class
template <typename implementation>
DEVICE_BASE<implementation>::DEVICE_BASE(size_t n) :  m_size(n), m_data(nullptr), m_graphicsResource(nullptr)
{
    assert_cuda(cudaMalloc(&m_data, m_size*sizeof(type)));
}

template <typename implementation>
DEVICE_BASE<implementation>::DEVICE_BASE(const DEVICE_BASE &other) : DEVICE_BASE(other.m_size)
{
    assert_cuda( cudaMemcpy(m_data, other.m_data, m_size*sizeof(type), cudaMemcpyDeviceToDevice));
}

template <typename implementation>
DEVICE_BASE<implementation>::~DEVICE_BASE()
{
    if(m_graphicsResource)
        unregisterGraphicsResource();
    else
        assert_cuda(cudaFree(m_data));
}

template <typename implementation>
DEVICE_BASE<implementation>::DEVICE_BASE(const DEVICE_BASE::host_type &other) : DEVICE_BASE(other.m_size)
{
    assert_cuda( cudaMemcpy(m_data, other.m_data, m_size*sizeof(type), cudaMemcpyHostToDevice));
}

template <typename implementation>
DEVICE_BASE<implementation>::operator host_type() const
{
    host_type host(m_size);
    assert_cuda( cudaMemcpy(host.m_data,m_data,m_size*sizeof(type),cudaMemcpyDeviceToHost));
    return host;
}

template <typename implementation>
DEVICE_BASE<implementation> &DEVICE_BASE<implementation>::operator=(const DEVICE_BASE::host_type& host)
{
    if(size() != host.size())
    {
        DEVICE_BASE<implementation> base(host);
        swap(*this,base);
    }
    else
    {
        assert_cuda( cudaMemcpy(m_data, host.m_data, m_size*sizeof(type), cudaMemcpyHostToDevice));
    }
    return *this;
}

template <typename implementation>
template<typename... Args>
void DEVICE_BASE<implementation>::loadParticle(size_t id, Particle<Args ...> &p) const
{
#if defined(__CUDA_ARCH__)
    p = lsFunctor::load(m_data[id]);
#endif
}

template <typename implementation>
template<typename... Args>
void DEVICE_BASE<implementation>::storeParticle(size_t id, const Particle<Args ...> &p)
{
#if defined(__CUDA_ARCH__)
    int i[] = {0, ((void)lsFunctor::template store(m_data[id], ext_base_cast<Args>(p)),1)...};
    (void)i[0]; // silence compiler warning abut i being unused
#endif
}

template <typename implementation>
void DEVICE_BASE<implementation>::registerGLGraphicsResource(uint32_t resourceID, cudaGraphicsMapFlags flag)
{
    assert_cuda(cudaFree(m_data));
    m_data = nullptr;
    assert_cuda(cudaGraphicsGLRegisterBuffer(&m_graphicsResource, resourceID, flag));
}

template <typename implementation>
void DEVICE_BASE<implementation>::mapGraphicsResource()
{
    if(m_graphicsResource)
    {
        size_t mappedBufferSize;
        assert_cuda(cudaGraphicsMapResources(1, &m_graphicsResource));
        assert_cuda(cudaGraphicsResourceGetMappedPointer((void **)&m_data, &mappedBufferSize, m_graphicsResource));
        assert_true(mappedBufferSize == m_size * sizeof(type), "Paticles",
                    "opengl buffer size is not equal to particle number");
    }
}

template <typename implementation>
void DEVICE_BASE<implementation>::unmapGraphicsResource()
{
    if(m_graphicsResource)
    {
        assert_cuda(cudaGraphicsUnmapResources(1, &m_graphicsResource));
        m_data = nullptr;
    }
}

template <typename implementation>
void DEVICE_BASE<implementation>::unregisterGraphicsResource()
{
    assert_cuda(cudaGraphicsUnregisterResource(m_graphicsResource));
    m_data=nullptr;
    m_graphicsResource = nullptr;
}

//template <typename implementation>
//DEVICE_BASE<implementation> DEVICE_BASE<implementation>::createDeviceCopy() const
//{
//    DEVICE_BASE b;
//    b.m_data=m_data;
//    b.m_size = m_size;
//    b.m_isDeviceCopy=true;
//    b.m_graphicsResource=nullptr;
//    return b;
//};

#endif //GRASPH2_DEVICE_BASE_H
