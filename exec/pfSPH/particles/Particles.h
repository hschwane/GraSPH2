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

//-------------------------------------------------------------------
// 3D position as f4_t
struct pos_impl
{
    CUDAHOSTDEV static Particle<POS> load(const f4_t & v) { return Particle<POS>(f3_t{v.x,v.y,v.z}); }

    template <typename T> CUDAHOSTDEV static void store(f4_t & v, const T& p) {}
    CUDAHOSTDEV static void store(f4_t & v, const POS& p) {v=f4_t{p.pos.x,p.pos.y,p.pos.z,0.0f};}
};

template <size_t n>
using SHARED_POS = SHARED_BASE<n,f4_t, pos_impl>;
using HOST_POS = HOST_BASE<f4_t, pos_impl>;
using DEV_POS = DEVICE_BASE<f4_t, pos_impl>;

//-------------------------------------------------------------------
// mass as f1_t
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

//-------------------------------------------------------------------
// 3D position and mass as f4_t
struct posm_impl
{
    CUDAHOSTDEV static Particle<POS> load(const f4_t & v) { return Particle<POS,MASS>(f3_t{v.x,v.y,v.z},v.w); }

    template <typename T> CUDAHOSTDEV static void store(f4_t & v, const T& p) {}
    CUDAHOSTDEV static void store(f4_t & v, const POS& p) {v.x=p.pos.x; v.y=p.pos.y; v.z=p.pos.z;}
    CUDAHOSTDEV static void store(f4_t & v, const MASS& p) {v.w=p.mass;}
};

template <size_t n>
using SHARED_POSM = SHARED_BASE<n,f4_t, posm_impl>;
using HOST_POSM = HOST_BASE<f4_t, posm_impl>;
using DEV_POSM = DEVICE_BASE<f4_t, posm_impl>;

//-------------------------------------------------------------------
// 3D velocity as f4_t
struct vel_impl
{
    CUDAHOSTDEV static Particle<VEL> load(const f4_t & v) { return Particle<VEL>(f3_t{v.x,v.y,v.z}); }

    template <typename T> CUDAHOSTDEV static void store(f4_t & v, const T& p) {}
    CUDAHOSTDEV static void store(f4_t & v, const VEL& p) {v=f4_t{p.vel.x,p.vel.y,p.vel.z,0.0f};}
};

template <size_t n>
using SHARED_VEL = SHARED_BASE<n,f4_t, vel_impl>;
using HOST_VEL = HOST_BASE<f4_t, vel_impl>;
using DEV_VEL = DEVICE_BASE<f4_t, vel_impl>;

//-------------------------------------------------------------------
// 3D acceleration as f4_t
struct acc_impl
{
    CUDAHOSTDEV static Particle<ACC> load(const f4_t & v) { return Particle<ACC>(f3_t{v.x,v.y,v.z}); }

    template <typename T> CUDAHOSTDEV static void store(f4_t & v, const T& p) {}
    CUDAHOSTDEV static void store(f4_t & v, const ACC& p) {v=f4_t{p.acc.x,p.acc.y,p.acc.z,0.0f};}
};

template <size_t n>
using SHARED_ACC = SHARED_BASE<n,f4_t, acc_impl>;
using HOST_ACC = HOST_BASE<f4_t, acc_impl>;
using DEV_ACC = DEVICE_BASE<f4_t, acc_impl>;

#endif //MPUTILS_PARTICLES_H
