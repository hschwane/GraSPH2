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
#include "Particle.h"
#include "SharedParticles.h"
#include "GlobalParticles.h"
#include <mpUtils.h>
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
// create bases for particles and particle buffers

MAKE_PARTICLE_BASE(POS,pos,f3_t);
MAKE_PARTICLE_BASE(MASS,mass,f1_t);
MAKE_PARTICLE_BASE(VEL,vel,f3_t);
MAKE_PARTICLE_BASE(ACC,acc,f3_t);

struct mass_impl
{
    CUDAHOSTDEV static Particle<MASS> load(const float & v) { return Particle<MASS>(v); }

    template <typename T> CUDAHOSTDEV static void store(float & v, const T& p) {}
    CUDAHOSTDEV static void store(float & v, const MASS& p) {v=p.mass;}
};

template <size_t n>
using SHARED_MASS = SHARED_BASE<n,float, mass_impl>;
using HOST_MASS = HOST_BASE<float, mass_impl>;
using DEV_MASS = DEVICE_BASE<float, mass_impl>;

#endif //MPUTILS_PARTICLES_H
