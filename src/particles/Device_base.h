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
//--------------------

//-------------------------------------------------------------------
/**
 * class template DEVICE_BASE
 *
 * @brief Class Template that holds particle attributes in device memory. You can use an openGL Buffer instead of the
 *              managed memory by casting the Particles object to the specific device base and calling the registerGLGraphics Resource.
 *              You can then use the map and unmap functions of the particles class to map or unmap the openGL Buffers of all its device bases.
 *              The resource will be automatically unregistered in the destructer. To unregister maually cast the Particles object to the base type
 *              and assign a new base object.
 * @tparam T type of the attribute
 * @tparam lsFunctor a struct that contains the following functions:
 *          - public static Particle< ...>load(const T& v) the attribute value v should be copied into the returned particle.
 *          - public template<typename U> static void store(T & v, const U& p) which should be specialized for different
 *              particle base classes and copy the attribute of the base p into the shared memory position referenced by v
 *          Reference implementations of such structs can be found in Particles.h.
 */
template <typename T, typename lsFunctor>
class DEVICE_BASE
{
public:
    //construction and destruction
    CUDAHOSTDEV DEVICE_BASE() : m_size(0), m_data(nullptr), m_isDeviceCopy(false), m_graphicsResource(nullptr) {}
    explicit DEVICE_BASE(size_t n);
    CUDAHOSTDEV ~DEVICE_BASE();

    // copy swap idom where copy construction is only allowed on th host
    // to copy on the device use device copy
    DEVICE_BASE(const DEVICE_BASE & other);
    CUDAHOSTDEV DEVICE_BASE( DEVICE_BASE&& other) noexcept : DEVICE_BASE() {swap(*this,other);}
    CUDAHOSTDEV DEVICE_BASE& operator=(DEVICE_BASE&& other) noexcept {swap(*this,other); return *this;}
    DEVICE_BASE& operator=(DEVICE_BASE other) noexcept {swap(*this,other); return *this;}
    CUDAHOSTDEV friend void swap(DEVICE_BASE & first, DEVICE_BASE & second) noexcept
    {
        using thrust::swap;
        swap(first.m_data,second.m_data);
        swap(first.m_size,second.m_size);
        swap(first.m_isDeviceCopy,second.m_isDeviceCopy);
        swap(first.m_graphicsResource,second.m_graphicsResource);
    }

    // converting from and to a compatible host base
    using host_type = HOST_BASE<T,lsFunctor>; //!< the type of host base this device base can be converted to
    DEVICE_BASE(const host_type & other); //!< construct from a compatible host base
    DEVICE_BASE& operator=(const host_type& other); //!< assign from compatible host base
    operator host_type() const; //!< convert to a compatible host base

    // shallow copying
    CUDAHOSTDEV DEVICE_BASE createDeviceCopy() const; //!< create a shallow copy for usage on the device

    // mapping of graphics resources
    void registerGLGraphicsResource(uint32_t resourceID, cudaGraphicsMapFlags flag = cudaGraphicsMapFlagsNone); //!< register an openGL Buffer to be used instead of the allocated memory
    void mapGraphicsResource(); //!< if an opengl buffer is used map the buffer to enable cuda usage
    void unmapGraphicsResource(); //!< unmap the opengl buffer so it can be used by openGL again

    // particle handling
    template<typename ... Args>
    CUDAHOSTDEV void loadParticle(size_t id, Particle<Args ...> & p) const; //!< get a particle object with the requested members  Note: while this is a host device function, calling it on the host will have no effect
    template<typename ... Args>
    CUDAHOSTDEV void storeParticle(size_t id, const Particle<Args ...> & p); //!< set the attributes of particle id according to the particle object  Note: while this is a host device function, calling it on the device will have no effect
    CUDAHOSTDEV void initialize(size_t i); //!< set all values to the default value

    // status checks
    CUDAHOSTDEV size_t size() { return m_size;} //!< returns the number of particles

    // types
    using bind_ref_to_t = host_type; //!< the type of host base this device base can be converted to

protected:
    DEVICE_BASE & operator=(const size_t & f) {return *this;}

public:
    void unregisterGraphicsResource(); //!< unregister the opengl resource from cuda

    bool m_isDeviceCopy; //!< if this is a shallow copy no memory is freed on destruction
    size_t m_size; //!< the umber of particles
    T* m_data; //!< the actual data
    cudaGraphicsResource* m_graphicsResource; //!< a cuda graphics resource that can be used as memory
};

//-------------------------------------------------------------------
// function definitions for DEVICE_BASE class
template<typename T, typename lsFunctor>
DEVICE_BASE<T, lsFunctor>::DEVICE_BASE(size_t n) :  m_size(n), m_data(nullptr), m_isDeviceCopy(false), m_graphicsResource(nullptr)
{
    assert_cuda(cudaMalloc(&m_data, m_size*sizeof(T)));
}

template<typename T, typename lsFunctor>
DEVICE_BASE<T, lsFunctor>::DEVICE_BASE(const DEVICE_BASE &other) : DEVICE_BASE(other.m_size)
{
    assert_cuda( cudaMemcpy(m_data, other.m_data, m_size*sizeof(T), cudaMemcpyDeviceToDevice));
}

template<typename T, typename lsFunctor>
DEVICE_BASE<T, lsFunctor>::~DEVICE_BASE()
{
#if defined(__CUDA_ARCH__)
    #if !defined(NDEBUG)
        if(m_isDeviceCopy || m_graphicsResource)
        {
            printf("Destroying base that is not a device copy on the device, this is likely to be a bug!");
        }
    #endif
#else
    if(!m_isDeviceCopy && !m_graphicsResource) assert_cuda(cudaFree(m_data));
    if(m_graphicsResource) unregisterGraphicsResource();
#endif
}

template<typename T, typename lsFunctor>
DEVICE_BASE<T, lsFunctor>::DEVICE_BASE(const DEVICE_BASE::host_type &other) : DEVICE_BASE(other.m_size)
{
    assert_cuda( cudaMemcpy(m_data, other.m_data, m_size*sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T, typename lsFunctor>
DEVICE_BASE<T, lsFunctor>::operator host_type() const
{
    host_type host(m_size);
    assert_cuda( cudaMemcpy(host.m_data,m_data,m_size*sizeof(T),cudaMemcpyDeviceToHost));
    return host;
}

template<typename T, typename lsFunctor>
DEVICE_BASE<T,lsFunctor> &DEVICE_BASE<T, lsFunctor>::operator=(const DEVICE_BASE::host_type& host)
{
    if(size() != host.size())
    {
        DEVICE_BASE<T,lsFunctor> base(host);
        swap(*this,base);
    }
    else
    {
        assert_cuda( cudaMemcpy(m_data, host.m_data, m_size*sizeof(T), cudaMemcpyHostToDevice));
    }
    return *this;
}

template<typename T, typename lsFunctor>
DEVICE_BASE<T, lsFunctor> DEVICE_BASE<T, lsFunctor>::createDeviceCopy() const
{
    DEVICE_BASE b;
    b.m_data=m_data;
    b.m_size = m_size;
    b.m_isDeviceCopy=true;
    b.m_graphicsResource=nullptr;
    return b;
};

template<typename T, typename lsFunctor>
template<typename... Args>
__device__ void DEVICE_BASE<T, lsFunctor>::loadParticle(size_t id, Particle<Args ...> &p) const
{
#if defined(__CUDA_ARCH__)
    p = lsFunctor::load(m_data[id]);
#endif
}

template<typename T, typename lsFunctor>
template<typename... Args>
__device__ void DEVICE_BASE<T, lsFunctor>::storeParticle(size_t id, const Particle<Args ...> &p)
{
#if defined(__CUDA_ARCH__)
    int i[] = {0, ((void)lsFunctor::template store(m_data[id], ext_base_cast<Args>(p)),1)...};
    (void)i[0]; // silence compiler warning abut i being unused
#endif
}

template<typename T, typename lsFunctor>
void DEVICE_BASE<T, lsFunctor>::registerGLGraphicsResource(uint32_t resourceID, cudaGraphicsMapFlags flag)
{
    assert_cuda(cudaFree(m_data));
    m_data = nullptr;
    assert_cuda(cudaGraphicsGLRegisterBuffer(&m_graphicsResource, resourceID, flag));
}

template<typename T, typename lsFunctor>
void DEVICE_BASE<T, lsFunctor>::mapGraphicsResource()
{
    if(m_graphicsResource)
    {
        size_t mappedBufferSize;
        assert_cuda(cudaGraphicsMapResources(1, &m_graphicsResource));
        assert_cuda(cudaGraphicsResourceGetMappedPointer((void **)&m_data, &mappedBufferSize, m_graphicsResource));
        assert_true(mappedBufferSize == m_size * sizeof(T), "Paticles",
                    "opengl buffer size is not equal to particle number");
    }
}

template<typename T, typename lsFunctor>
void DEVICE_BASE<T, lsFunctor>::unmapGraphicsResource()
{
    if(m_graphicsResource)
    {
        assert_cuda(cudaGraphicsUnmapResources(1, &m_graphicsResource));
        m_data = nullptr;
    }
}

template<typename T, typename lsFunctor>
void DEVICE_BASE<T, lsFunctor>::unregisterGraphicsResource()
{
    assert_cuda(cudaGraphicsUnregisterResource(m_graphicsResource));
    m_data=nullptr;
    m_graphicsResource = nullptr;
}

template<typename T, typename lsFunctor>
void DEVICE_BASE<T, lsFunctor>::initialize(size_t i)
{
#if defined(__CUDA_ARCH__)
    m_data[i] = T{lsFunctor::defaultValue};
#endif
}

#endif //GRASPH2_DEVICE_BASE_H
