/*
 * GraSPH2
 * particle_buffer_impl.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */
#ifndef GRASPH2_PARTICLE_BUFFER_IMPL_H
#define GRASPH2_PARTICLE_BUFFER_IMPL_H

// includes
//--------------------
#include <mpUtils/mpUtils.h>
#include <mpUtils/mpCuda.h>
#include "types.h"
#include "partice_attributes.h"
#include "Particle.h"
#include "Host_base.h"
#include "Device_base.h"
#include "Shared_base.h"
#include "Device_reference.h"
//--------------------

//!< class to identify particle buffer implementations
class pb_impl {};

/**
 * @brief macro to generate implementations for particle buffers from an impl struct
 */
#define MAKE_PARTICLE_BUFFER_IMPLEMENTATION(NAME,IMPL) \
template <size_t n> \
using SHARED_ ## NAME = SHARED_BASE<n,IMPL>; \
using HOST_ ## NAME = HOST_BASE<IMPL>; \
using DEV_ ## NAME = DEVICE_BASE<IMPL>; \
using DREF_ ## NAME = DEVICE_REFERENCE<IMPL>

//-------------------------------------------------------------------
// definitions of all _impl classes that manage data storage

//-------------------------------------------------------------------
// 3D position as f4_t
struct pos_impl : pb_impl
{
    using type = f4_t;
    static constexpr type defaultValue = {0,0,0,0};
    using particleType = Particle<POS>;

    CUDAHOSTDEV static auto load(const type & v);
    template <typename T> CUDAHOSTDEV static void store(type & v, const T& p);
};

CUDAHOSTDEV inline auto pos_impl::load(const type & v) { return particleType(f3_t{v.x,v.y,v.z}); }
template<typename T> CUDAHOSTDEV inline void pos_impl::store(type &v, const T &p) {}
template<> CUDAHOSTDEV inline void pos_impl::store<POS>(type & v, const POS& p) {v=type{p.pos.x,p.pos.y,p.pos.z,0.0f};}


//-------------------------------------------------------------------
// mass as f1_t
struct mass_impl : pb_impl
{
    using type = f1_t;
    static constexpr type defaultValue = {0};
    using particleType = Particle<MASS>;

    CUDAHOSTDEV static auto load(const type & v);
    template <typename T> CUDAHOSTDEV static void store(type & v, const T& p);
};

CUDAHOSTDEV inline auto mass_impl::load(const type & v) { return particleType(v); }
template<typename T> CUDAHOSTDEV inline void mass_impl::store(type &v, const T &p) {}
template<> CUDAHOSTDEV inline void mass_impl::store<MASS>(type & v, const MASS& p) {v=p.mass;}


//-------------------------------------------------------------------
// 3D position and mass as f4_t
struct posm_impl : pb_impl
{
    using type = f4_t;
    static constexpr type defaultValue = {0,0,0,0};
    using particleType = Particle<POS,MASS>;

    CUDAHOSTDEV static auto load(const type & v);
    template <typename T> CUDAHOSTDEV static void store(type & v, const T& p);
};

CUDAHOSTDEV inline auto posm_impl::load(const type & v) { return particleType( f3_t{v.x,v.y,v.z}, v.w); }
template<typename T> CUDAHOSTDEV inline void posm_impl::store(type &v, const T &p) {}
template<> CUDAHOSTDEV inline void posm_impl::store<POS>(type & v, const POS& p) {v.x=p.pos.x; v.y=p.pos.y; v.z=p.pos.z;}
template<> CUDAHOSTDEV inline void posm_impl::store<MASS>(type & v, const MASS& p) {v.w=p.mass;}


//-------------------------------------------------------------------
// 3D velocity as f4_t
struct vel_impl : pb_impl
{
    using type = f4_t;
    static constexpr type defaultValue = {0,0,0,0};
    using particleType = Particle<VEL>;

    CUDAHOSTDEV static auto load(const type & v);
    template <typename T> CUDAHOSTDEV static void store(type & v, const T& p);
};

CUDAHOSTDEV inline auto vel_impl::load(const type & v) { return particleType(f3_t{v.x,v.y,v.z}); }
template<typename T> CUDAHOSTDEV inline void vel_impl::store(type &v, const T &p) {}
template<> CUDAHOSTDEV inline void vel_impl::store<VEL>(type & v, const VEL& p) {v=type{p.vel.x,p.vel.y,p.vel.z,0.0f};}


//-------------------------------------------------------------------
// 3D acceleration as f4_t
struct acc_impl : pb_impl
{
    using type = f4_t;
    static constexpr type defaultValue = {0,0,0,0};
    using particleType = Particle<ACC>;

    CUDAHOSTDEV static auto load(const type & v);
    template <typename T> CUDAHOSTDEV static void store(type & v, const T& p);
};

CUDAHOSTDEV inline auto acc_impl::load(const type & v) { return particleType(f3_t{v.x,v.y,v.z}); }
template<typename T> CUDAHOSTDEV inline void acc_impl::store(type &v, const T &p) {}
template<> CUDAHOSTDEV inline void acc_impl::store<ACC>(type & v, const ACC& p) {v=type{p.acc.x,p.acc.y,p.acc.z,0.0f};}

//-------------------------------------------------------------------
// balsara switch value
struct balsara_impl : pb_impl
{
    using type = f1_t;
    static constexpr type defaultValue = {0};
    using particleType = Particle<BALSARA>;

    CUDAHOSTDEV static auto load(const type & v);
    template <typename T> CUDAHOSTDEV static void store(type & v, const T& p);
};

CUDAHOSTDEV inline auto balsara_impl::load(const type & v) { return particleType(v); }
template<typename T> CUDAHOSTDEV inline void balsara_impl::store(type &v, const T &p) {}
template<> CUDAHOSTDEV inline void balsara_impl::store<BALSARA>(type & v, const BALSARA& p) {v=p.balsara;}


//-------------------------------------------------------------------
// 3D smoothed velocity for xsph as f4_t
struct xvel_impl : pb_impl
{
    using type = f4_t;
    static constexpr type defaultValue = {0,0,0,0};
    using particleType = Particle<XVEL>;

    CUDAHOSTDEV static auto load(const type & v);
    template <typename T> CUDAHOSTDEV static void store(type & v, const T& p);
};

CUDAHOSTDEV inline auto xvel_impl::load(const type & v) { return particleType(f3_t{v.x,v.y,v.z}); }
template<typename T> CUDAHOSTDEV inline void xvel_impl::store(type &v, const T &p) {}
template<> CUDAHOSTDEV inline void xvel_impl::store<XVEL>(type & v, const XVEL& p) {v=type{p.xvel.x,p.xvel.y,p.xvel.z,0.0f};}


//-------------------------------------------------------------------
// hydrodynamic density rho
struct density_impl : pb_impl
{
    using type = f1_t;
    static constexpr type defaultValue = {0};
    using particleType = Particle<DENSITY>;

    CUDAHOSTDEV static auto load(const type & v);
    template <typename T> CUDAHOSTDEV static void store(type & v, const T& p);
};

CUDAHOSTDEV inline auto density_impl::load(const type & v) { return particleType(v); }
template<typename T> CUDAHOSTDEV inline void density_impl::store(type &v, const T &p) {}
template<> CUDAHOSTDEV inline void density_impl::store<DENSITY>(type & v, const DENSITY& p) {v=p.density;}


//-------------------------------------------------------------------
// time derivative of hydrodynamic density rho
struct density_dt_impl : pb_impl
{
    using type = f1_t;
    static constexpr type defaultValue = {0};
    using particleType = Particle<DENSITY_DT>;

    CUDAHOSTDEV static auto load(const type & v);
    template <typename T> CUDAHOSTDEV static void store(type & v, const T& p);
};

CUDAHOSTDEV inline auto density_dt_impl::load(const type & v) { return particleType(v); }
template<typename T> CUDAHOSTDEV inline void density_dt_impl::store(type &v, const T &p) {}
template<> CUDAHOSTDEV inline void density_dt_impl::store<DENSITY_DT>(type & v, const DENSITY_DT& p) {v=p.density_dt;}


//-------------------------------------------------------------------
// deviatoric stress tensor S
struct deviatoric_stress_impl : pb_impl
{
    using type = m3_t;
    static constexpr f1_t defaultValue = {0};
    using particleType = Particle<DSTRESS>;

    CUDAHOSTDEV static auto load(const type & v);
    template <typename T> CUDAHOSTDEV static void store(type & v, const T& p);
};

CUDAHOSTDEV inline auto deviatoric_stress_impl::load(const type & v) { return particleType(v); }
template<typename T> CUDAHOSTDEV inline void deviatoric_stress_impl::store(type &v, const T &p) {}
template<> CUDAHOSTDEV inline void deviatoric_stress_impl::store<DSTRESS>(type & v, const DSTRESS& p) {v=p.dstress;}


//-------------------------------------------------------------------
// time derivative of deviatoric stress tensor S
struct deviatoric_stress_dt_impl : pb_impl
{
    using type = m3_t;
    static constexpr f1_t defaultValue = {0};
    using particleType = Particle<DSTRESS_DT>;

    CUDAHOSTDEV static auto load(const type & v);
    template <typename T> CUDAHOSTDEV static void store(type & v, const T& p);
};

CUDAHOSTDEV inline auto deviatoric_stress_dt_impl::load(const type & v) { return particleType(v); }
template<typename T> CUDAHOSTDEV inline void deviatoric_stress_dt_impl::store(type &v, const T &p) {}
template<> CUDAHOSTDEV inline void deviatoric_stress_dt_impl::store<DSTRESS_DT>(type & v, const DSTRESS_DT& p) {v=p.dstress_dt;}

//-------------------------------------------------------------------
// balsara switch value
struct maxvsig_impl : pb_impl
{
    using type = f1_t;
    static constexpr type defaultValue = {0};
    using particleType = Particle<MAXVSIG>;

    CUDAHOSTDEV static auto load(const type & v);
    template <typename T> CUDAHOSTDEV static void store(type & v, const T& p);
};

CUDAHOSTDEV inline auto maxvsig_impl::load(const type & v) { return particleType(v); }
template<typename T> CUDAHOSTDEV inline void maxvsig_impl::store(type &v, const T &p) {}
template<> CUDAHOSTDEV inline void maxvsig_impl::store<MAXVSIG>(type & v, const MAXVSIG& p) {v=p.max_vsig;}


//-------------------------------------------------------------------
// generate the aliases for different types of buffers and define the order of attributes

MAKE_PARTICLE_BUFFER_IMPLEMENTATION(POS,pos_impl);
MAKE_PARTICLE_BUFFER_IMPLEMENTATION(MASS,mass_impl);
MAKE_PARTICLE_BUFFER_IMPLEMENTATION(POSM,posm_impl);
MAKE_PARTICLE_BUFFER_IMPLEMENTATION(VEL,vel_impl);
MAKE_PARTICLE_BUFFER_IMPLEMENTATION(ACC,acc_impl);
MAKE_PARTICLE_BUFFER_IMPLEMENTATION(BALSARA,balsara_impl);
MAKE_PARTICLE_BUFFER_IMPLEMENTATION(XVEL,xvel_impl);
MAKE_PARTICLE_BUFFER_IMPLEMENTATION(DENSITY,density_impl);
MAKE_PARTICLE_BUFFER_IMPLEMENTATION(DENSITY_DT,density_dt_impl);
MAKE_PARTICLE_BUFFER_IMPLEMENTATION(DSTRESS,deviatoric_stress_impl);
MAKE_PARTICLE_BUFFER_IMPLEMENTATION(DSTRESS_DT,deviatoric_stress_dt_impl);
MAKE_PARTICLE_BUFFER_IMPLEMENTATION(MAXVSIG,maxvsig_impl);

using host_base_order = std::tuple<HOST_POS,HOST_MASS,HOST_POSM,HOST_VEL,HOST_ACC,HOST_BALSARA,HOST_XVEL,HOST_DENSITY,HOST_DENSITY_DT,HOST_DSTRESS,HOST_DSTRESS_DT,HOST_MAXVSIG >;
using device_base_order = std::tuple<DEV_POS,DEV_MASS,DEV_POSM,DEV_VEL,DEV_ACC,DEV_BALSARA,DEV_XVEL,DEV_DENSITY,DEV_DENSITY_DT,DEV_DSTRESS,DEV_DSTRESS_DT,DEV_MAXVSIG>;
using dref_base_order = std::tuple<DREF_POS,DREF_MASS,DREF_POSM,DREF_VEL,DREF_ACC,DREF_BALSARA,DREF_XVEL,DREF_DENSITY,DREF_DENSITY_DT,DREF_DSTRESS,DREF_DSTRESS_DT,DREF_MAXVSIG>;
using shared_base_order = std::tuple<SHARED_POS<0>,SHARED_MASS<0>,SHARED_POSM<0>,SHARED_VEL<0>,SHARED_ACC<0>,SHARED_BALSARA<0>,SHARED_XVEL<0>,SHARED_DENSITY<0>,SHARED_DENSITY_DT<0>,SHARED_DSTRESS<0>,SHARED_DSTRESS_DT<0>,SHARED_MAXVSIG<0>>;

#endif //GRASPH2_PARTICLE_BUFFER_IMPL_H
