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
MAKE_PARTICLE_BASE(RHO,rho,f1_t);

//-------------------------------------------------------------------
// 3D position as f4_t
struct pos_impl
{
    CUDAHOSTDEV static auto load(const f4_t & v) { return Particle<POS>(f3_t{v.x,v.y,v.z}); }
    template <typename T> CUDAHOSTDEV static void store(f4_t & v, const T& p);
    static constexpr f4_t defaultValue = {0,0,0,0};
};

template<typename T> CUDAHOSTDEV void pos_impl::store(f4_t &v, const T &p) {}
template<> CUDAHOSTDEV void pos_impl::store<POS>(f4_t & v, const POS& p) {v=f4_t{p.pos.x,p.pos.y,p.pos.z,0.0f};}

template <size_t n>
using SHARED_POS = SHARED_BASE<n,f4_t, pos_impl>;
using HOST_POS = HOST_BASE<f4_t, pos_impl>;
using DEV_POS = DEVICE_BASE<f4_t, pos_impl>;

//-------------------------------------------------------------------
// mass as f1_t
struct mass_impl
{
    CUDAHOSTDEV static auto load(const float & v) { return Particle<MASS>(v); }
    template <typename T> CUDAHOSTDEV static void store(float & v, const T& p);
    static constexpr f1_t defaultValue = {0};
};

template<typename T> CUDAHOSTDEV void mass_impl::store(float &v, const T &p) {}
template<> CUDAHOSTDEV void mass_impl::store<MASS>(float & v, const MASS& p) {v=p.mass;}

template <size_t n>
using SHARED_MASS = SHARED_BASE<n,float, mass_impl>;
using HOST_MASS = HOST_BASE<float, mass_impl>;
using DEV_MASS = DEVICE_BASE<float, mass_impl>;

//-------------------------------------------------------------------
// 3D position and mass as f4_t
struct posm_impl
{
    CUDAHOSTDEV static auto load(const f4_t & v) { return Particle<POS,MASS>(f3_t{v.x,v.y,v.z},v.w); }
    template <typename T> CUDAHOSTDEV static void store(f4_t & v, const T& p);
    static constexpr f4_t defaultValue = {0,0,0,0};
};

template<typename T> CUDAHOSTDEV void posm_impl::store(f4_t &v, const T &p) {}
template <> CUDAHOSTDEV void posm_impl::store<POS>(f4_t & v, const POS& p) {v.x=p.pos.x; v.y=p.pos.y; v.z=p.pos.z;}
template <> CUDAHOSTDEV void posm_impl::store<MASS>(f4_t & v, const MASS& p) {v.w=p.mass;}

template <size_t n>
using SHARED_POSM = SHARED_BASE<n,f4_t, posm_impl>;
using HOST_POSM = HOST_BASE<f4_t, posm_impl>;
using DEV_POSM = DEVICE_BASE<f4_t, posm_impl>;

//-------------------------------------------------------------------
// 3D velocity as f4_t
struct vel_impl
{
    CUDAHOSTDEV static auto load(const f4_t & v) { return Particle<VEL>(f3_t{v.x,v.y,v.z}); }
    template <typename T> CUDAHOSTDEV static void store(f4_t & v, const T& p);
    static constexpr f4_t defaultValue = {0,0,0,0};
};

template<typename T> CUDAHOSTDEV void vel_impl::store(f4_t &v, const T &p) {}
template<> CUDAHOSTDEV void vel_impl::store<VEL>(f4_t & v, const VEL& p) {v=f4_t{p.vel.x,p.vel.y,p.vel.z,0.0f};}

template <size_t n>
using SHARED_VEL = SHARED_BASE<n,f4_t, vel_impl>;
using HOST_VEL = HOST_BASE<f4_t, vel_impl>;
using DEV_VEL = DEVICE_BASE<f4_t, vel_impl>;

//-------------------------------------------------------------------
// 3D acceleration as f4_t
struct acc_impl
{
    CUDAHOSTDEV static auto load(const f4_t & v) { return Particle<ACC>(f3_t{v.x,v.y,v.z}); }
    template <typename T> CUDAHOSTDEV static void store(f4_t & v, const T& p);
    static constexpr f4_t defaultValue = {0,0,0,0};
};

template<typename T> CUDAHOSTDEV void acc_impl::store(f4_t &v, const T &p) {}
template<> CUDAHOSTDEV void acc_impl::store<ACC>(f4_t & v, const ACC& p) {v=f4_t{p.acc.x,p.acc.y,p.acc.z,0.0f};}

template <size_t n>
using SHARED_ACC = SHARED_BASE<n,f4_t, acc_impl>;
using HOST_ACC = HOST_BASE<f4_t, acc_impl>;
using DEV_ACC = DEVICE_BASE<f4_t, acc_impl>;

//-------------------------------------------------------------------
// hydrodynamic density rho
struct density_impl
{
    CUDAHOSTDEV static auto load(const f1_t & v) { return Particle<RHO>(v); }
    template <typename T> CUDAHOSTDEV static void store(f1_t & v, const T& p);
    static constexpr f1_t defaultValue = 0;
};

template<typename T> CUDAHOSTDEV void density_impl::store(f1_t &v, const T &p) {}
template<> CUDAHOSTDEV void density_impl::store<RHO>(f1_t & v, const RHO& p) {v=p.rho;}

template <size_t n>
using SHARED_RHO = SHARED_BASE<n,f1_t, density_impl>;
using HOST_RHO = HOST_BASE<f1_t, density_impl>;
using DEV_RHO = DEVICE_BASE<f1_t, density_impl>;

#endif //MPUTILS_PARTICLES_H
