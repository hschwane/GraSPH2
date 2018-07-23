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
#include <tuple>
#include <mpUtils.h>
#include "ext_base_cast.h"
#include "Particle.h"
//--------------------

//-------------------------------------------------------------------
/**
 * class template Particles
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
 * between host and device using. Data copying will also work for HOST-HOST and DEV-DEV assignment. If you mix HOST and DEV Bases
 * in one Particles Object you can use the uploadData and downloadData member functions to control memory transfer.
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
class Particles : public Args...
{
public:
    // constructors
    CUDAHOSTDEV Particles() : m_numParticles(0), m_isDeviceCopy(false), Args(0)... {} //!< default constructor
    explicit Particles(size_t n) : m_numParticles(n), m_isDeviceCopy(false), Args(n)... {} //!< construct particle buffer which can contain n particles

    // shallow copying
    CUDAHOSTDEV auto createDeviceCopy() const; //!< creates a shallow copy which only include device bases to be used on the device

    // conversions
    template <typename... TArgs>
    Particles(const Particles<TArgs...>& other);  //!< construct this from a particle buffer with different attributes
    template <typename... TArgs>
    Particles& operator=(const Particles<TArgs...> &b); //!< assignment between particles buffers with different atributes

    // particle handling
    template<typename... particleArgs>
    CUDAHOSTDEV Particle<particleArgs...> loadParticle(size_t id) const; //!< get a particle object with the requested members
    template<typename... particleArgs>
    CUDAHOSTDEV void storeParticle(size_t id, const Particle<particleArgs...>& p); //!< set the attributes of particle id according to the particle object

    void uploadData(); //!< if this object contains HOST_ and DEV_ base classes data will be uploaded from the host to the device
    void downloadData(); //!< if this object contains HOST_ and DEV_ base classes data will be downloaded from the device to the host

    // status checks
    CUDAHOSTDEV size_t size() const {return m_numParticles;} //!< return the number of particles in this buffer
    CUDAHOSTDEV bool isDeviceCopy() const {return m_isDeviceCopy;} //!< check if the particle object is a shallow copy only containing device pointers

private:
    CUDAHOSTDEV Particles(const Particles& other, bool devCopy) //!< copy constructor that constructs a Particles object by making shallow copies of the bases
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
    Particles<Ts...> deviceCopyHelper(std::tuple<Ts...>&&) const; //!< returns a particles object hat was constructed using Particles<Ts...>(*this,true)

    template <typename...Ts, typename...Us>
    void updownhelper(std::tuple<Ts...>&&, std::tuple<Us...>&&ust); //!< calls operator = to assign all base classed of Ts to base classes of type Us

    template <typename T, typename...Us>
    void updownhelper(std::tuple<Us...>&&); //!< calls operator = to assign all base classes Us to T

    bool m_isDeviceCopy; //!< if this is a device copy no memory will be freed on destruction
    size_t m_numParticles; //!< the number of particles stored in this buffer
};

// forward declarations
//--------------------
template <typename T, typename lsFunctor>
class DEVICE_BASE;
//--------------------

//-------------------------------------------------------------------
/**
 * class template DEVICE_BASE
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

    // status checks
    size_t size() { return m_size;} //!< returns the number of particles

    // friends and types
    template <typename Type, typename Functor> friend class DEVICE_BASE; // be friends with the corresponding device base
    using bind_ref_to_t = DEVICE_BASE<T,lsFunctor>; //!< show to which device_base this can be assigned

protected:
    HOST_BASE & operator=(const size_t & f) {return *this;} //!< ignore assignments of size_t from the base class

private:
    size_t m_size; //!< the number of particles stored in this buffer
    T* m_data;  //!< the actual data
};

//-------------------------------------------------------------------
/**
 * class template DEVICE_BASE
 *
 * @brief Class Template that holds particle attributes in device memory
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
    CUDAHOSTDEV DEVICE_BASE() : m_size(0), m_data(nullptr), m_isDeviceCopy(false) {}
    explicit DEVICE_BASE(size_t n);
    ~DEVICE_BASE() {if(!m_isDeviceCopy) assert_cuda(cudaFree(m_data));}

    // copy swap idom where copy construction is only allowed on th host
    // to copy on the device use device copy
    DEVICE_BASE(const DEVICE_BASE & other);
    CUDAHOSTDEV DEVICE_BASE( DEVICE_BASE&& other) noexcept : DEVICE_BASE() {swap(*this,other);}
    CUDAHOSTDEV DEVICE_BASE& operator=(DEVICE_BASE other) {swap(*this,other); return *this;}
    CUDAHOSTDEV friend void swap(DEVICE_BASE & first, DEVICE_BASE & second)
    {
        using thrust::swap;
        swap(first.m_data,second.m_data);
        swap(first.m_size,second.m_size);
        swap(first.m_isDeviceCopy,second.m_isDeviceCopy);
    }

    // converting from and to a compatible host base
    using host_type = HOST_BASE<T,lsFunctor>; //!< the type of host base this device base can be converted to
    DEVICE_BASE(const host_type & other); //!< construct from a compatible host base
    operator host_type() const; //!< convert to a compatible host base

    // shallow copying
    CUDAHOSTDEV DEVICE_BASE createDeviceCopy() const; //!< create a shallow copy for usage on the device

    // particle handling
    template<typename ... Args>
    CUDAHOSTDEV void loadParticle(size_t id, Particle<Args ...> & p) const; //!< get a particle object with the requested members  Note: while this is a host device function, calling it on the host will have no effect
    template<typename ... Args>
    CUDAHOSTDEV void storeParticle(size_t id, const Particle<Args ...> & p); //!< set the attributes of particle id according to the particle object  Note: while this is a host device function, calling it on the device will have no effect

    // status checks
    CUDAHOSTDEV size_t size() { return m_size;} //!< returns the number of particles

    // types
    using bind_ref_to_t = host_type; //!< the type of host base this device base can be converted to

protected:
    DEVICE_BASE & operator=(const size_t & f) {return *this;}

private:
    bool m_isDeviceCopy; //!< if this is a shallow copy no memory is freed on destruction
    size_t m_size; //!< the umber of particles
    T* m_data; //!< the actual data
};

//-------------------------------------------------------------------
// function definitions for Particles class

template<typename... Args>
template<typename... TArgs>
Particles<Args...>::Particles(const Particles<TArgs...> &other)
                                                : m_numParticles(other.size()),
                                                  m_isDeviceCopy(other.isDeviceCopy()),
                                                  Args(ext_base_cast<Args>(other))... {}

template<typename... Args>
template<typename... TArgs>
Particles<Args...> & Particles<Args...>::operator=(const Particles<TArgs...> &b)
{
    int t[] = {0, ((void)Args::operator=(ext_base_cast<Args>(b)),1)...};
    (void)t[0]; // silence compiler warning abut t being unused
    return *this;
}

template<typename... Args>
template<typename... particleArgs>
Particle<particleArgs...> Particles<Args...>::loadParticle(size_t id) const
{
    Particle<particleArgs...> p{};
    int t[] = {0, ((void)Args::loadParticle(id,p),1)...}; // call load particle functions of all the base classes
    (void)t[0]; // silence compiler warning abut t being unused
    return p;
}

template<typename... Args>
template<typename... particleArgs>
void Particles<Args...>::storeParticle(size_t id, const Particle<particleArgs...> &p)
{
    int t[] = {0, ((void)Args::storeParticle(id,p),1)...};
    (void)t[0]; // silence compiler warning abut t being unused
}

template<typename... Args>
template <typename...Ts>
Particles<Ts...> Particles<Args...>::deviceCopyHelper(std::tuple<Ts...>&&) const
{
    static_assert(sizeof...(Ts)>0,"You can not create a device copy if there is no device base.");
    return Particles<Ts...>(*this,true);
};

template<typename... Args>
auto Particles<Args...>::createDeviceCopy() const
{
#if defined(__CUDA_ARCH__)
    return Particles<Args...>(*this,true);
#else
    return deviceCopyHelper(mpu::remove_t<copy_condition,Args...>());
#endif
}

template<typename... Args>
template<typename T, typename... Us>
void Particles<Args...>::updownhelper(std::tuple<Us...> &&)
{
    int t[] = {0, ((void)T::operator=(static_cast<Us>(*this)),1)...};
    (void)t[0]; // silence compiler warning abut t being unused
}

template<typename... Args>
template<typename... Ts, typename... Us>
void Particles<Args...>::updownhelper(std::tuple<Ts...> &&, std::tuple<Us...> && ust)
{
    int t[] = {0, ((void)updownhelper<Ts>(ust),1)...};
    (void)t[0]; // silence compiler warning abut t being unused
}

template<typename... Args>
void Particles<Args...>::uploadData()
{
    updownhelper(mpu::remove_t<copy_condition,Args...>(),mpu::remove_t<inv_copy_condition,Args...>());
}

template<typename... Args>
void Particles<Args...>::downloadData()
{
    updownhelper(mpu::remove_t<inv_copy_condition,Args...>(),mpu::remove_t<copy_condition,Args...>());
}

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

//-------------------------------------------------------------------
// function definitions for DEVICE_BASE class
template<typename T, typename lsFunctor>
DEVICE_BASE<T, lsFunctor>::DEVICE_BASE(size_t n) :  m_size(n), m_data(nullptr), m_isDeviceCopy(false)
{
    assert_cuda(cudaMalloc(&m_data, m_size*sizeof(T)));
}

template<typename T, typename lsFunctor>
DEVICE_BASE<T, lsFunctor>::DEVICE_BASE(const DEVICE_BASE &other) : DEVICE_BASE(other.m_size)
{
    assert_cuda( cudaMemcpy(m_data, other.m_data, m_size, cudaMemcpyDeviceToDevice));
}

template<typename T, typename lsFunctor>
DEVICE_BASE<T, lsFunctor>::DEVICE_BASE(const DEVICE_BASE::host_type &other) : DEVICE_BASE(other.m_size)
{
    assert_cuda( cudaMemcpy(m_data, other.m_data, m_size, cudaMemcpyHostToDevice));
}

template<typename T, typename lsFunctor>
DEVICE_BASE<T, lsFunctor>::operator host_type() const
{
    host_type host(m_size);
    logDEBUG("PARTICLES") << m_size;
    logDEBUG("PARTICLES") << host.m_size;
    logDEBUG("PARTICLES") << host.m_data;
    logDEBUG("PARTICLES") << m_data;
    assert_cuda( cudaMemcpy(host.m_data,m_data,m_size,cudaMemcpyDeviceToHost));
    return host;
}

template<typename T, typename lsFunctor>
DEVICE_BASE<T, lsFunctor> DEVICE_BASE<T, lsFunctor>::createDeviceCopy() const
{
    DEVICE_BASE b;
    b.m_data=m_data;
    b.m_size = m_size;
    b.m_isDeviceCopy=true;
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


#endif //MPUTILS_GLOBALPARTICLES_H
