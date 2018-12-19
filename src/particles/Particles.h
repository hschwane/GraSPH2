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
#include <mpUtils/mpUtils.h>
#include <mpUtils/mpCuda.h>
#include "../types.h"
//--------------------

//-------------------------------------------------------------------
// create bases for particles and particle buffers

MAKE_PARTICLE_BASE(POS,pos,f3_t, deriv_of_nothing);
MAKE_PARTICLE_BASE(MASS,mass,f1_t, deriv_of_nothing);
MAKE_PARTICLE_BASE(VEL,vel,f3_t, POS);
MAKE_PARTICLE_BASE(ACC,acc,f3_t, VEL);
MAKE_PARTICLE_BASE(XVEL,xvel,f3_t, POS);
MAKE_PARTICLE_BASE(DENSITY,density,f1_t, deriv_of_nothing);
MAKE_PARTICLE_BASE(DENSITY_DT,density_dt,f1_t, DENSITY);
MAKE_PARTICLE_BASE(DSTRESS,dstress,m3_t, deriv_of_nothing);
MAKE_PARTICLE_BASE(DSTRESS_DT,dstress_dt,m3_t, DSTRESS);

//-------------------------------------------------------------------
// 3D position as f4_t
struct pos_impl
{
    CUDAHOSTDEV static auto load(const f4_t & v) { return Particle<POS>(f3_t{v.x,v.y,v.z}); }
    template <typename T> CUDAHOSTDEV static void store(f4_t & v, const T& p);
    static constexpr f4_t defaultValue = {0,0,0,0};
};

template<typename T> CUDAHOSTDEV void pos_impl::store(f4_t &v, const T &p) {}
template<> CUDAHOSTDEV inline void pos_impl::store<POS>(f4_t & v, const POS& p) {v=f4_t{p.pos.x,p.pos.y,p.pos.z,0.0f};}

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
template<> CUDAHOSTDEV inline void mass_impl::store<MASS>(float & v, const MASS& p) {v=p.mass;}

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
template <> CUDAHOSTDEV inline void posm_impl::store<POS>(f4_t & v, const POS& p) {v.x=p.pos.x; v.y=p.pos.y; v.z=p.pos.z;}
template <> CUDAHOSTDEV inline void posm_impl::store<MASS>(f4_t & v, const MASS& p) {v.w=p.mass;}

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
template<> CUDAHOSTDEV void inline vel_impl::store<VEL>(f4_t & v, const VEL& p) {v=f4_t{p.vel.x,p.vel.y,p.vel.z,0.0f};}

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
template<> CUDAHOSTDEV void inline acc_impl::store<ACC>(f4_t & v, const ACC& p) {v=f4_t{p.acc.x,p.acc.y,p.acc.z,0.0f};}

template <size_t n>
using SHARED_ACC = SHARED_BASE<n,f4_t, acc_impl>;
using HOST_ACC = HOST_BASE<f4_t, acc_impl>;
using DEV_ACC = DEVICE_BASE<f4_t, acc_impl>;

//-------------------------------------------------------------------
// 3D smoothed velocity for xsph as f4_t
struct xvel_impl
{
    CUDAHOSTDEV static auto load(const f4_t & v) { return Particle<XVEL>(f3_t{v.x,v.y,v.z}); }
    template <typename T> CUDAHOSTDEV static void store(f4_t & v, const T& p);
    static constexpr f4_t defaultValue = {0,0,0,0};
};

template<typename T> CUDAHOSTDEV void xvel_impl::store(f4_t &v, const T &p) {}
template<> CUDAHOSTDEV void inline xvel_impl::store<XVEL>(f4_t & v, const XVEL& p) {v=f4_t{p.xvel.x,p.xvel.y,p.xvel.z,0.0f};}

template <size_t n>
using SHARED_XVEL = SHARED_BASE<n,f4_t, xvel_impl>;
using HOST_XVEL = HOST_BASE<f4_t, xvel_impl>;
using DEV_XVEL = DEVICE_BASE<f4_t, xvel_impl>;


//-------------------------------------------------------------------
// hydrodynamic density rho
struct density_impl
{
    CUDAHOSTDEV static auto load(const f1_t & v) { return Particle<DENSITY>(v); }
    template <typename T> CUDAHOSTDEV static void store(f1_t & v, const T& p);
    static constexpr f1_t defaultValue = 0;
};

template<typename T> CUDAHOSTDEV void density_impl::store(f1_t &v, const T &p) {}
template<> CUDAHOSTDEV inline void density_impl::store<DENSITY>(f1_t & v, const DENSITY& p) {v=p.density;}

template <size_t n>
using SHARED_DENSITY = SHARED_BASE<n,f1_t, density_impl>;
using HOST_DENSITY = HOST_BASE<f1_t, density_impl>;
using DEV_DENSITY = DEVICE_BASE<f1_t, density_impl>;

//-------------------------------------------------------------------
// time derivative of hydrodynamic density rho
struct density_dt_impl
{
    CUDAHOSTDEV static auto load(const f1_t & v) { return Particle<DENSITY_DT>(v); }
    template <typename T> CUDAHOSTDEV static void store(f1_t & v, const T& p);
    static constexpr f1_t defaultValue = 0;
};

template<typename T> CUDAHOSTDEV void density_dt_impl::store(f1_t &v, const T &p) {}
template<> CUDAHOSTDEV void inline density_dt_impl::store<DENSITY_DT>(f1_t & v, const DENSITY_DT& p) {v=p.density_dt;}

template <size_t n>
using SHARED_DENSITY_DT = SHARED_BASE<n,f1_t, density_dt_impl>;
using HOST_DENSITY_DT = HOST_BASE<f1_t, density_dt_impl>;
using DEV_DENSITY_DT = DEVICE_BASE<f1_t, density_dt_impl>;

//-------------------------------------------------------------------
// deviatoric stress tensor S
struct deviatoric_stress_impl
{
    CUDAHOSTDEV static auto load(const m3_t & v) { return Particle<DSTRESS>(v); }
    template <typename T> CUDAHOSTDEV static void store(m3_t & v, const T& p);
    static constexpr f1_t defaultValue = 0;
};

template<typename T> CUDAHOSTDEV void deviatoric_stress_impl::store(m3_t &v, const T &p) {}
template<> CUDAHOSTDEV void inline deviatoric_stress_impl::store<DSTRESS>(m3_t & v, const DSTRESS& p) {v=p.dstress;}

template <size_t n>
using SHARED_DSTRESS = SHARED_BASE<n,m3_t, deviatoric_stress_impl>;
using HOST_DSTRESS = HOST_BASE<m3_t, deviatoric_stress_impl>;
using DEV_DSTRESS = DEVICE_BASE<m3_t, deviatoric_stress_impl>;


//-------------------------------------------------------------------
// time derivative of deviatoric stress tensor S
struct deviatoric_stress_dt_impl
{
    CUDAHOSTDEV static auto load(const m3_t & v) { return Particle<DSTRESS_DT>(v); }
    template <typename T> CUDAHOSTDEV static void store(m3_t & v, const T& p);
    static constexpr f1_t defaultValue = 0;
};

template<typename T> CUDAHOSTDEV void deviatoric_stress_dt_impl::store(m3_t &v, const T &p) {}
template<> CUDAHOSTDEV inline void deviatoric_stress_dt_impl::store<DSTRESS_DT>(m3_t & v, const DSTRESS_DT& p) {v=p.dstress_dt;}

template <size_t n>
using SHARED_DSTRESS_DT = SHARED_BASE<n,m3_t, deviatoric_stress_dt_impl>;
using HOST_DSTRESS_DT = HOST_BASE<m3_t, deviatoric_stress_dt_impl>;
using DEV_DSTRESS_DT = DEVICE_BASE<m3_t, deviatoric_stress_dt_impl>;

#endif //MPUTILS_PARTICLES_H
