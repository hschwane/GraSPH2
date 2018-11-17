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
#include <nvfunctional>
#include "Particles.h"
#include <mpUtils.h>
//--------------------

//-------------------------------------------------------------------
// template functions to execute some common algorithms on particles

/**
 * @brief initialize all particles
 * @param PARTICLES the Particles object that manages the particles
 * @param BASES_OF_I choose bases according to the attributes of particle i that need to be initialized
 * @param CODE code to be executed for each particle, the particle is named pi you can use i to access the particles id
 */
#define INIT_EACH(PARTICLES, BASES_OF_I, CODE)  \
{ \
    for(const auto &i : mpu::gridStrideRange(PARTICLES .size())) \
    { \
        Particle< BASES_OF_I > pi{}; \
        CODE \
        PARTICLES .storeParticle(i, pi); \
    } \
}

/**
 * @brief execute a function for every particle
 * @param PARTICLES the Particles object that manages the particles
 * @param BASES_OF_I choose bases according to the attributes of particle i needed for the computation
 * @param BASES_OF_I_TO_LOAD choose bases according to the attributes of particle i that should be loaded from memory
 * @param BASES_OF_I_TO_STORE choose bases according to the attributes of particle i that should be stored in memory
 * @param CODE code to be executed for each particle, the particle is named pi
 */
#define DO_FOR_EACH( PARTICLES, BASES_OF_I, BASES_OF_I_TO_LOAD, BASES_OF_I_TO_STORE, CODE) \
{ \
    for(const auto &i : mpu::gridStrideRange( PARTICLES .size())) \
    { \
        Particle< BASES_OF_I > pi = PARTICLES .loadParticle< BASES_OF_I_TO_LOAD >(i); \
        CODE \
        PARTICLES .storeParticle(i, pi); \
    } \
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
#define DO_FOR_EACH_PAIR( PARTICLES, BASES_OF_I, BASES_OF_I_TO_LOAD, BASES_OF_I_TO_STORE, BASES_OF_J, PAIR_CODE, SINGLE_CODE)\
{\
    for(const auto &i : mpu::gridStrideRange( PARTICLES .size())) \
    { \
        Particle< BASES_OF_I > pi = PARTICLES .loadParticle< BASES_OF_I_TO_LOAD >(i); \
        for(const auto &j : mpu::Range<int>( PARTICLES .size())) \
        { \
            const auto pj = PARTICLES .loadParticle< BASES_OF_J >(j); \
            PAIR_CODE \
        } \
        SINGLE_CODE \
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
            const auto p = PARTICLES .loadParticle< BASES_OF_J >(tile*TILE_SIZE+threadIdx.x); \
            shared.storeParticle(threadIdx.x,p); \
            __syncthreads(); \
            \
            for(int j = 0; j < thisTileSize; j++) \
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
