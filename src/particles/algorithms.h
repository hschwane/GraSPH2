/*
 * mpUtils
 * algorithms.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */
#ifndef MPUTILS_ALGORITHMS_H
#define MPUTILS_ALGORITHMS_H

// includes
//--------------------
#include <mpUtils/mpUtils.h>
#include <mpUtils/mpCuda.h>
#include "partice_attributes.h"
#include "particle_buffer_impl.h"
#include "Particle.h"
#include "HostParticleBuffer.h"
#include "DeviceParticleBuffer.h"
#include "DeviceParticleReference.h"
#include "SharedParticles.h"
//--------------------

//-------------------------------------------------------------------
// template functions to execute some common algorithms on particles

/**
 * @brief Allows to execute the function specified by job on each particle in the HostParticleBuffer pb on the Host. (Pass a DeviceParticleBuffer to execute on the device)
 * @tparam job the type of job struct that specifies what function to execute
 * @tparam blockSize the cuda block size when executing on the device
 * @tparam particleBuffer the type of HostParticleBuffer on which the function is executed
 * @param pb the HostParticleBuffer on which the function is executed
 */
template<typename job, size_t blockSize=0, typename particleBuffer, std::enable_if_t< mpu::is_instantiation_of< HostParticleBuffer,particleBuffer>::value, int> =0>
void do_for_each(particleBuffer& pb);

/**
 * @brief Allows to execute the function specified by job on each particle in the DeviceParticleBuffer pb on the device. (Pass a HostParticleBuffer to execute on the host)
 * @tparam job the type of job struct that specifies what function to execute
 * @tparam blockSize the cuda block size when executing on the device
 * @tparam particleBuffer the type of DeviceParticleBuffer on which the function is executed
 * @param pb the DeviceParticleBuffer on which the function is executed
 */
template<typename job, size_t blockSize=256, typename particleBuffer, std::enable_if_t< mpu::is_instantiation_of< DeviceParticleBuffer,particleBuffer>::value, int> =0>
void do_for_each(particleBuffer& pb);


//-------------------------------------------------------------------
// template functions definitions

namespace detail {

    template<typename ParticleType, typename buffer>
    struct load_helper;

    template<typename ...Args, typename buffer>
    struct load_helper<Particle<Args...>, buffer>
    {
        CUDAHOSTDEV static auto load(const buffer &pb, size_t i)
        {
            return pb.template loadParticle<Args...>(i);
        }
    };

    template<typename job, typename deviceReference>
    __global__ void do_for_each_impl(deviceReference pb)
    {
        for(const auto &i : mpu::gridStrideRange(pb.size()))
        {
            auto pi = load_helper<typename job::load_type, deviceReference>::load(pb, i);
            auto result = job::for_each(pi, i);
            pb.storeParticle(i, result);
        }
    }

}

template<typename job, size_t blockSize, typename particleBuffer, std::enable_if_t< mpu::is_instantiation_of< DeviceParticleBuffer,particleBuffer>::value, int> >
void do_for_each(particleBuffer& pb)
{
    detail::do_for_each_impl<job><<< mpu::numBlocks(pb.size(),blockSize), blockSize>>>( pb.getDeviceReference());
}

template<typename job, size_t blockSize, typename particleBuffer, std::enable_if_t< mpu::is_instantiation_of< HostParticleBuffer,particleBuffer>::value, int> >
void do_for_each(particleBuffer& pb)
{
    #pragma omp parallel for
    for(int i = 0; i < pb.size(); i++)
    {
        auto pi = detail::load_helper< typename job::load_type, particleBuffer>::load(pb,i);
        auto result = job::for_each( pi, i);
        pb.storeParticle( i, result);
    }
}


/**
 * @brief execute a function for every pair of particles
 * @param PARTICLES the Particles object that manages the particles
 * @param BASES_OF_I choose bases according to the attributes of particle i needed for the interaction
 * @param BASES_OF_I_TO_LOAD choose bases according to the attributes of particle i that should be loaded from memory
 * @param BASES_OF_I_TO_STORE choose bases according to the attributes of particle i that should be stored in memory
 * @param BASES_OF_J choose bases according to the attributes of particle j needed for the interaction
 * @param PAIR_CODE code to be executed for each pair of particles, the particles are named pi and pj
 * @param SINGLE_CODE code to be executed after all interactions before pi is saved to global memory
 */
#define DO_FOR_EACH_PAIR( PARTICLES, BASES_OF_I, BASES_OF_I_TO_LOAD, BASES_OF_I_TO_STORE, BASES_OF_J, SINGLE_CODEA, PAIR_CODE, SINGLE_CODEB)\
{\
    for(const auto &i : mpu::gridStrideRange( PARTICLES .size())) \
    { \
        Particle< BASES_OF_I > pi = PARTICLES .loadParticle< BASES_OF_I_TO_LOAD >(i); \
        SINGLE_CODEA \
        for(const auto &j : mpu::Range<int>( PARTICLES .size())) \
        { \
            const auto pj = PARTICLES .loadParticle< BASES_OF_J >(j); \
            PAIR_CODE \
        } \
        SINGLE_CODEB \
        PARTICLES .storeParticle(i, static_cast<Particle< BASES_OF_I_TO_STORE >>(pi)); \
    } \
}

/**
 * @brief execute a function for every pair of particles using shared memory to save memory accesses
 * @param TILE_SIZE a compile time constant, the number of particles that are hold in shared memory USE THE BLOCK SIZE
 * @param PARTICLES the Particles object that manages the particles
 * @param SHARED_BASES choose SHARED_ bases from Particles.h according to the particle properties the should be saved in shared memory
 * @param BASES_OF_I choose bases according to the attributes of particle i needed for the interaction
 * @param BASES_OF_I_TO_LOAD choose bases according to the attributes of particle i that should be loaded from memory
 * @param BASES_OF_I_TO_STORE choose bases according to the attributes of particle i that should be stored in memory
 * @param BASES_OF_J choose bases according to the attributes of particle j needed for the interaction
 * @param SINGLE_CODEA code to be executed before all interactions just after pi is loaded from global memory
 * @param PAIR_CODE code to be executed for each pair of particles, the particles are named pi and pj
 * @param SINGLE_CODEB code to be executed after all interactions before pi is saved to global memory
 */
#define DO_FOR_EACH_PAIR_SM( TILE_SIZE, PARTICLES, SHARED_BASES, BASES_OF_I, BASES_OF_I_TO_LOAD, BASES_OF_I_TO_STORE, BASES_OF_J, SINGLE_CODEA, PAIR_CODE, SINGLE_CODEB) \
{\
    SharedParticles<TILE_SIZE, SHARED_BASES> shared;\
    const int remain = PARTICLES .size() % TILE_SIZE; \
    for(const auto &i : mpu::gridStrideRange( PARTICLES .size())) \
    { \
        Particle< BASES_OF_I > pi = PARTICLES .loadParticle< BASES_OF_I_TO_LOAD >(i); \
        SINGLE_CODEA \
        \
        const int thisTileSize = (i < PARTICLES .size() - remain) ? TILE_SIZE : remain; \
        const int numTiles = (PARTICLES .size() + thisTileSize - 1) / thisTileSize; \
        for (int tile = 0; tile < numTiles; tile++) \
        { \
            int loadIndex = tile*TILE_SIZE+threadIdx.x; \
            if( loadIndex < PARTICLES .size()) \
            {\
                const auto p = PARTICLES .loadParticle< BASES_OF_J >(tile*TILE_SIZE+threadIdx.x); \
                shared.storeParticle(threadIdx.x,p); \
            }\
            __syncthreads(); \
            \
            for(int j = 0; j < thisTileSize && (j+remain < thisTileSize || tile < numTiles); j++) \
            { \
                const auto pj = shared.template loadParticle< BASES_OF_J >(j);\
                PAIR_CODE \
            } \
            __syncthreads();\
        } \
        SINGLE_CODEB \
        PARTICLES .storeParticle(i, static_cast<Particle<BASES_OF_I_TO_STORE>>(pi)); \
    } \
}

#endif //MPUTILS_ALGORITHMS_H
