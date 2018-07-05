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
#include <Cuda/cudaUtils.h>
#include <vector_functions.hpp>
//--------------------

// forward declare particles class
//--------------------
class Particles;
template <size_t n, template <size_t> class... TArgs> class SharedParticles;

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
 * enables handling and manipulation of the attributes uf a single Particle
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
    __host__ __device__ Particle() {}

    template <typename... T>
    __host__ __device__
    explicit Particle(T &&... args) : Args(std::forward<T>(args))... {}

private:
    // friends
    friend class Particles;
    template <size_t n, template <size_t> class... TArgs> friend class SharedParticles;

    /**
     * construct a particle from a particle buffer eg what is listed as friends
     */
    template <typename T>
    __host__ __device__
    explicit Particle(size_t id,T& buffer) : Args(id,buffer)... {}

    /**
     * store a particle inside a particle buffer eg what is listed as friends
     */
    template <typename T>
    __host__ __device__
    void store(size_t id, T& buffer) const
    {
        int t[] = {0, ((void)Args::store(id,buffer),1)...}; // call store functions of all the base classes
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
    __host__ __device__ SharedParticles() : TArgs<n>()... {}

    __device__ void copyFromGlobal(size_t shared_id, size_t global_id, const Particles& global); //!< load particle from global to shared memory
    __device__ void copyToGlobal(size_t shared_id, size_t global_id, const Particles& global); //!< store a particle in global memory

    template<typename... particleArgs>
    __device__
    Particle<particleArgs...> loadParticle(size_t id); //!< get a particle object with the requested members

    template<typename... particleArgs>
    __device__
    void storeParticle(const Particle<particleArgs...>& p,size_t id); //!< set the attributes of particle id according to the particle object

    __device__ size_t size() {return n;}

private:
    // particles have lots of friends ^^
    friend class POS;
    friend class M;
    friend class POSM;
    friend class VEL;
    friend class ACC;
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
class Particles : public mpu::Managed
{
public:
    Particles();
    Particles(size_t n);
    ~Particles();

    void free(); //!< free all memory
    void reallocate(size_t n); //!< reallocate all particle data and initialize to zero
    void copyToDevice();    //!< initiate copy of data from host to device
    void copyFromDevice();  //!< initiate copying of data from host to device

    template<typename... Args>
    __host__ __device__
    Particle<Args...> loadParticle(size_t id); //!< get a particle object with the requested members

    template<typename... Args>
    __host__ __device__
    void storeParticle(const Particle<Args...>& p,size_t id); //!< set the attributes of particle id according to the particle object

    __host__ __device__ size_t size() {return m_numParticles;} //!< return the number of particles

    // make particles non copyable
    Particles(const Particles& that) = delete;
    Particles& operator=(const Particles& that) = delete;

private:
    void allocate(size_t n);

    size_t m_numParticles;

    // particle storage on host
    f4_t* m_hpos;
    f4_t* m_hvel;
    f4_t* m_hacc;

    // particle storage on device
    f4_t* m_dpos;
    f4_t* m_dvel;
    f4_t* m_dacc;

    // particles have lots of friends ^^
    friend class POS;
    friend class M;
    friend class POSM;
    friend class VEL;
    friend class ACC;

    template <size_t n> friend class SHARED_POSM;
    template <size_t n> friend class SHARED_VEL;
    template <size_t n> friend class SHARED_ACC;
};

//-------------------------------------------------------------------
// define template functions of all the classes
template <size_t n, template <size_t> class... TArgs>
__device__
void SharedParticles<n, TArgs...>::copyFromGlobal(size_t shared_id, size_t global_id, const Particles &global)
{
    int t[] = {0, ((void)TArgs<n>::copyFromGlobal(shared_id,global_id,global),1)...}; // call copy functions of all the base classes
}

template <size_t n, template <size_t> class... TArgs>
__device__
void SharedParticles<n, TArgs...>::copyToGlobal(size_t shared_id, size_t global_id, const Particles &global)
{
    int t[] = {0, ((void)TArgs<n>::copyToGlobal(shared_id,global_id,global),1)...}; // call copy functions of all the base classes
}

template <size_t n, template <size_t> class... TArgs>
template<typename... particleArgs>
__device__
Particle<particleArgs...> SharedParticles<n, TArgs...>::loadParticle(size_t id)
{
    return Particle<particleArgs...>(id,*this);
}

template <size_t n, template <size_t> class... TArgs>
template<typename... particleArgs>
__device__
void SharedParticles<n, TArgs...>::storeParticle(const Particle<particleArgs...> &p, size_t id)
{
    p.store(id,*this);
}

template<typename... Args>
Particle<Args...> Particles::loadParticle(size_t id)
{
    return Particle<Args...>(id,*this);
}

template<typename... Args>
void Particles::storeParticle(const Particle<Args...> &p, size_t id)
{
    p.store(id,*this);
}


//-------------------------------------------------------------------
// define some classes that hold the members of the particle class

//--------------------
// hold the position
class POS
{
public:
    f3_t pos;

protected:
    __host__ __device__ POS() : pos({0,0,0}) {}
    __host__ __device__ POS(f3_t val) : pos(val) {}

    template <typename T>
    __host__ __device__
    explicit POS(size_t id,const T& buffer)
    {
#if defined(__CUDA_ARCH__)
        f4_t b = buffer.m_dpos[id];
#else
        f4_t b = buffer.m_hpos[id];
#endif
        pos = mpu::toDim3<f3_t>(b);
    }

    template <typename T>
    __host__ __device__
    void store(size_t id, T& buffer) const
    {
#if defined(__CUDA_ARCH__)
        buffer.m_dpos[id].x = pos.x;
        buffer.m_dpos[id].y = pos.y;
        buffer.m_dpos[id].z = pos.z;
#else
        buffer.m_hpos[id].x = pos.x;
        buffer.m_hpos[id].y = pos.y;
        buffer.m_hpos[id].z = pos.z;
#endif
    }
};

//--------------------
// hold the mass
class M
{
public:
    f1_t mass;

protected:
    __host__ __device__ M() : mass(0) {}
    __host__ __device__ M(f1_t val) : mass(val) {}

    template <typename T>
    __host__ __device__
    explicit M(size_t id,const T& buffer)
#if defined(__CUDA_ARCH__)
    : mass(buffer.m_dpos[id].w) {}
#else
    : mass(buffer.m_hpos[id].w) {}
#endif

    template <typename T>
    __host__ __device__
    void store(size_t id, T& buffer) const
    {
#if defined(__CUDA_ARCH__)
        buffer.m_dpos[id].w = mass;
#else
        buffer.m_hpos[id].w = mass;
#endif
    }
};

//--------------------
// hold the position and mass
class POSM
{
public:
    f3_t pos;
    f1_t mass;

protected:
    __host__ __device__ POSM() : pos({0,0,0}), mass(0) {}
    __host__ __device__ POSM(f4_t val) : pos(mpu::toDim3<f3_t>(val)), mass(val.w)  {}

    template <typename T>
    __host__ __device__
    explicit POSM(size_t id,const T& buffer)
    {
#if defined(__CUDA_ARCH__)
        f4_t b = buffer.m_dpos[id];
#else
        f4_t b = buffer.m_hpos[id];
#endif
        pos = mpu::toDim3<f3_t>(b);
        mass = b.w;
    }

    template <typename T>
    __host__ __device__
    void store(size_t id, T& buffer) const
    {
#if defined(__CUDA_ARCH__)
        buffer.m_dpos[id] = {pos.x,pos.y,pos.z,mass};
#else
        buffer.m_hpos[id] = {pos.x,pos.y,pos.z,mass};
#endif
    }
};

//--------------------
// hold velocity
class VEL
{
public:
    f3_t vel;

protected:
    __host__ __device__ VEL() : vel({0,0,0}) {}
    __host__ __device__ VEL(f3_t val) : vel(val) {}

    template <typename T>
    __host__ __device__
    explicit VEL(size_t id,const T& buffer)
    {
#if defined(__CUDA_ARCH__)
        f4_t b = buffer.m_dvel[id];
#else
        f4_t b = buffer.m_hvel[id];
#endif
        vel = mpu::toDim3<f3_t>(b);
    }

    template <typename T>
    __host__ __device__
    void store(size_t id, T& buffer) const
    {
#if defined(__CUDA_ARCH__)
        buffer.m_hvel[id] = {vel.x, vel.y, vel.z, 0};
#else
        buffer.m_hvel[id] = {vel.x, vel.y, vel.z, 0};
#endif
    }
};

//--------------------
// hold acceleration
class ACC
{
public:
    f3_t acc;

protected:
    __host__ __device__ ACC() : acc({0,0,0}) {}
    __host__ __device__ ACC(f3_t val) : acc(val) {}

    template <typename T>
    __host__ __device__
    explicit ACC(size_t id,const T& buffer)
    {
#if defined(__CUDA_ARCH__)
        f4_t b = buffer.m_dacc[id];
#else
        f4_t b = buffer.m_hacc[id];
#endif
        acc = mpu::toDim3<f3_t>(b);
    }

    template <typename T>
    __host__ __device__
    void store(size_t id, T& buffer) const
    {
#if defined(__CUDA_ARCH__)
        buffer.m_hacc[id] = {acc.x, acc.y, acc.z, 0};
#else
        buffer.m_hacc[id] = {acc.x, acc.y, acc.z, 0};
#endif
    }
};

//-------------------------------------------------------------------
// define some classes that hold the members of the sharedParticles class

//--------------------
// hold  position and mass
template <size_t n>
class SHARED_POSM
{
public:
    __device__
    SHARED_POSM()
    {
        __shared__ f4_t mem[n];
        m_dpos = mem;
    }

protected:
    f4_t* m_dpos;

    __device__
    void copyFromGlobal(size_t shared_id, size_t global_id, const Particles& global)
    {
        m_dpos[shared_id] = global.m_dpos[global_id];
    }

    __device__
    void copyToGlobal(size_t shared_id, size_t global_id, const Particles& global)
    {
        global.m_dpos[global_id] = m_dpos[shared_id];
    }
};

//--------------------
// hold the velocity
template <size_t n>
class SHARED_VEL
{
public:
    __device__
    SHARED_VEL()
    {
        __shared__ f4_t mem[n];
        m_dvel = mem;
    }
protected:
    f4_t* m_dvel;

    __device__
    void copyFromGlobal(size_t shared_id, size_t global_id, const Particles& global)
    {
        m_dvel[shared_id] = global.m_dvel[global_id];
    }

    __device__
    void copyToGlobal(size_t shared_id, size_t global_id, const Particles& global)
    {
        global.m_dvel[global_id] = m_dvel[shared_id];
    }
};

//--------------------
// hold the acceleration
template <size_t n>
class SHARED_ACC
{
public:
    __device__
    SHARED_ACC()
    {
        __shared__ f4_t mem[n];
        m_dacc = mem;
    }
protected:
    f4_t* m_dacc;

    __device__
    void copyFromGlobal(size_t shared_id, size_t global_id, const Particles& global)
    {
        m_dacc[shared_id] = global.m_dacc[global_id];
    }

    __device__
    void copyToGlobal(size_t shared_id, size_t global_id, const Particles& global)
    {
        global.m_dacc[global_id] = m_dacc[shared_id];
    }
};

#endif //MPUTILS_PARTICLES_H
