/*
 * mpUtils
 * Particles.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the Particles class
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */

#ifndef MPUTILS_PARTICLES_H
#define MPUTILS_PARTICLES_H

// includes
//--------------------
#include <thrust/swap.h>
#include <mpUtils.h>
#include "ext_base_cast.h"
#include "Particles_helper.h"
//--------------------

// forward declarations
//--------------------

//-------------------------------------------------------------------
// define the data types used for the simulation
#define SINGLE_PRECISION

#if defined(DOUBLE_PRECISION)
    using f1_t=double;
    using f2_t=double2;
    using f3_t=double3;
    using f4_t=double4;
#else
    using f1_t=float;
    using f2_t=float2;
    using f3_t=float3;
    using f4_t=float4;
#endif

//-------------------------------------------------------------------
/**
 * class Particle
 *
 * enables handling and manipulation of the attributes of a single Particle
 *
 * usage:
 * Specify which particle attributes you want to manipulate by passing type names of the trivial classes below
 * as template parameters. You can use:
 *  POS  <- only use when register memory is low, use POSM instead
 *  M
 *  POSM <- combines position and mass, much faster!
 *  VEL
 *  ACC
 *
 */
template <typename... Args>
class Particle : public Args...
{
public:
    Particle()= default; //!< default construct particle values are undefined

    template <typename... T, std::enable_if_t< mpu::conjunction<mpu::is_list_initable<Args, T&&>...>::value, int> = 0>
    CUDAHOSTDEV
    explicit Particle(T && ... args) : Args(std::forward<T>(args))... {} //!< construct a particle from its attributes

    template <typename... T>
    CUDAHOSTDEV
    Particle(const Particle<T...> &b) : Args(ext_base_cast<Args>(b))... {} //!< construct a particle from another particle with different attributes

    template <typename... T>
    CUDAHOSTDEV
    Particle<Args...>& operator=(const Particle<T...> &b)
    {
        int t[] = {0, ((void)Args::operator=(ext_base_cast<Args>(b)),1)...};
        return *this;
    }
};

//-------------------------------------------------------------------
/**
 * class SharedParticles
 *
 * handles particles in shared memory on the device
 *
 * usage:
 * Specify how many particles should be stored in shared memory and what attributes should be defined for them.
 * Beware that you have to take care of synchronization yourself.
 * Supported particle attributes are:
 * SHARED_POSM
 * SHARED_VEL
 * SHARED_ACC
 *
 */
template <size_t n, template <size_t> class... TArgs>
class SharedParticles : public TArgs<n>...
{
public:
    __device__ SharedParticles() : TArgs<n>()... {}
    SharedParticles(const SharedParticles&)=delete;
    SharedParticles& operator=(const SharedParticles&)=delete;

//    __device__ void copyFromGlobal(size_t shared_id, size_t global_id, const Particles& global); //!< load particle from global to shared memory
//    __device__ void copyToGlobal(size_t shared_id, size_t global_id, const Particles& global); //!< store a particle in global memory

    template<typename... particleArgs>
    __device__
    Particle<particleArgs...> loadParticle(size_t id) //!< get a particle object with the requested members
    {
        Particle<particleArgs...> p{};
        int t[] = {0, ((void)TArgs<n>::loadParticle(id,p),1)...}; // call load particle functions of all the base classes
        return p;
    }

    template<typename... particleArgs>
    __device__
    void storeParticle(const Particle<particleArgs...>& p,size_t id) //!< set the attributes of particle id according to the particle object
    {
        int t[] = {0, ((void)TArgs<n>::storeParticle(id,p),1)...};
    }

    __device__ size_t size() {return n;}
};

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
// creating all the base classes for the different Particles and particle buffers
struct VOID {using bind_ref_to_t=void;};
struct VOID1 {using bind_ref_to_t=void;};
struct VOID2 {using bind_ref_to_t=void;};
struct VOID3 {using bind_ref_to_t=void;};

MAKE_PARTICLE_BASE(POS,pos,f3_t);
MAKE_PARTICLE_BASE(MASS,mass,f1_t);
MAKE_PARTICLE_BASE(VEL,vel,f3_t);
MAKE_PARTICLE_BASE(ACC,acc,f3_t);

MAKE_SHARED_PARTICLE_BASE(SHARED_POSM,f4_t,POS,(f3_t{v.x,v.y,v.z}),m_sm[id].x=b.pos.x;m_sm[id].y=b.pos.y;m_sm[id].z=b.pos.z,
                                          MASS, v.z, m_sm[id].w=b.mass,
                                         VOID1, VOID1(), ,
                                         VOID2, VOID2(), );


// MAKE_SHARED_PARTICLE_BASE(SHARED_POSM,f4_t, _ppm_t, (f3_t{v.x,v.y,v.z},v.z), ({p.pos.x,p.pos.y,p.pos.z,p.mass}));
// MAKE_SHARED_PARTICLE_BASE(SHARED_VEL,f4_t,Particle<VEL>,(f3_t{v.x,v.y,v.z}), ({p.vel.x,p.vel.y,p.vel.z,0.0}));
// MAKE_SHARED_PARTICLE_BASE(SHARED_ACC,f4_t,Particle<ACC>,(f3_t{v.x,v.y,v.z}), ({p.acc.x,p.acc.y,p.acc.z,0.0}));

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


#endif //MPUTILS_PARTICLES_H
