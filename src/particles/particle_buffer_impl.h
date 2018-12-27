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
using DEV_ ## NAME = DEVICE_BASE<IMPL>

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

CUDAHOSTDEV auto pos_impl::load(const type & v) { return particleType(f3_t{v.x,v.y,v.z}); }
template<typename T> CUDAHOSTDEV void pos_impl::store(type &v, const T &p) {}
template<> CUDAHOSTDEV inline void pos_impl::store<POS>(type & v, const POS& p) {v=type{p.pos.x,p.pos.y,p.pos.z,0.0f};}

MAKE_PARTICLE_BUFFER_IMPLEMENTATION(POS,pos_impl);


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

CUDAHOSTDEV auto mass_impl::load(const type & v) { return particleType(v); }
template<typename T> CUDAHOSTDEV void mass_impl::store(type &v, const T &p) {}
template<> CUDAHOSTDEV inline void mass_impl::store<MASS>(type & v, const MASS& p) {v=p.mass;}

MAKE_PARTICLE_BUFFER_IMPLEMENTATION(MASS,mass_impl);


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

CUDAHOSTDEV auto posm_impl::load(const type & v) { return particleType( f3_t{v.x,v.y,v.z}, v.w); }
template<typename T> CUDAHOSTDEV void posm_impl::store(type &v, const T &p) {}
template<> CUDAHOSTDEV inline void posm_impl::store<POS>(type & v, const POS& p) {v.x=p.pos.x; v.y=p.pos.y; v.z=p.pos.z;}
template<> CUDAHOSTDEV inline void posm_impl::store<MASS>(type & v, const MASS& p) {v.w=p.mass;}

MAKE_PARTICLE_BUFFER_IMPLEMENTATION(POSM,posm_impl);


//    //-------------------------------------------------------------------
//    // 3D velocity as f4_t
//    struct vel_impl
//    {
//        CUDAHOSTDEV static auto load(const f4_t & v) { return Particle<VEL>(f3_t{v.x,v.y,v.z}); }
//        template <typename T> CUDAHOSTDEV static void store(f4_t & v, const T& p);
//        static constexpr f4_t defaultValue = {0,0,0,0};
//    };
//
//    template<typename T> CUDAHOSTDEV void vel_impl::store(f4_t &v, const T &p) {}
//    template<> CUDAHOSTDEV void inline vel_impl::store<VEL>(f4_t & v, const VEL& p) {v=f4_t{p.vel.x,p.vel.y,p.vel.z,0.0f};}
//
//    template <size_t n>
//    using SHARED_VEL = SHARED_BASE<n,f4_t, vel_impl>;
//    using HOST_VEL = HOST_BASE<f4_t, vel_impl>;
//    using DEV_VEL = DEVICE_BASE<f4_t, vel_impl>;
//
//    //-------------------------------------------------------------------
//    // 3D acceleration as f4_t
//    struct acc_impl
//    {
//        CUDAHOSTDEV static auto load(const f4_t & v) { return Particle<ACC>(f3_t{v.x,v.y,v.z}); }
//        template <typename T> CUDAHOSTDEV static void store(f4_t & v, const T& p);
//        static constexpr f4_t defaultValue = {0,0,0,0};
//    };
//
//    template<typename T> CUDAHOSTDEV void acc_impl::store(f4_t &v, const T &p) {}
//    template<> CUDAHOSTDEV void inline acc_impl::store<ACC>(f4_t & v, const ACC& p) {v=f4_t{p.acc.x,p.acc.y,p.acc.z,0.0f};}
//
//    template <size_t n>
//    using SHARED_ACC = SHARED_BASE<n,f4_t, acc_impl>;
//    using HOST_ACC = HOST_BASE<f4_t, acc_impl>;
//    using DEV_ACC = DEVICE_BASE<f4_t, acc_impl>;
//
//    //-------------------------------------------------------------------
//    // 3D smoothed velocity for xsph as f4_t
//    struct xvel_impl
//    {
//        CUDAHOSTDEV static auto load(const f4_t & v) { return Particle<XVEL>(f3_t{v.x,v.y,v.z}); }
//        template <typename T> CUDAHOSTDEV static void store(f4_t & v, const T& p);
//        static constexpr f4_t defaultValue = {0,0,0,0};
//    };
//
//    template<typename T> CUDAHOSTDEV void xvel_impl::store(f4_t &v, const T &p) {}
//    template<> CUDAHOSTDEV void inline xvel_impl::store<XVEL>(f4_t & v, const XVEL& p) {v=f4_t{p.xvel.x,p.xvel.y,p.xvel.z,0.0f};}
//
//    template <size_t n>
//    using SHARED_XVEL = SHARED_BASE<n,f4_t, xvel_impl>;
//    using HOST_XVEL = HOST_BASE<f4_t, xvel_impl>;
//    using DEV_XVEL = DEVICE_BASE<f4_t, xvel_impl>;
//
//
//    //-------------------------------------------------------------------
//    // hydrodynamic density rho
//    struct density_impl
//    {
//        CUDAHOSTDEV static auto load(const f1_t & v) { return Particle<DENSITY>(v); }
//        template <typename T> CUDAHOSTDEV static void store(f1_t & v, const T& p);
//        static constexpr f1_t defaultValue = 0;
//    };
//
//    template<typename T> CUDAHOSTDEV void density_impl::store(f1_t &v, const T &p) {}
//    template<> CUDAHOSTDEV inline void density_impl::store<DENSITY>(f1_t & v, const DENSITY& p) {v=p.density;}
//
//    template <size_t n>
//    using SHARED_DENSITY = SHARED_BASE<n,f1_t, density_impl>;
//    using HOST_DENSITY = HOST_BASE<f1_t, density_impl>;
//    using DEV_DENSITY = DEVICE_BASE<f1_t, density_impl>;
//
//    //-------------------------------------------------------------------
//    // time derivative of hydrodynamic density rho
//    struct density_dt_impl
//    {
//        CUDAHOSTDEV static auto load(const f1_t & v) { return Particle<DENSITY_DT>(v); }
//        template <typename T> CUDAHOSTDEV static void store(f1_t & v, const T& p);
//        static constexpr f1_t defaultValue = 0;
//    };
//
//    template<typename T> CUDAHOSTDEV void density_dt_impl::store(f1_t &v, const T &p) {}
//    template<> CUDAHOSTDEV void inline density_dt_impl::store<DENSITY_DT>(f1_t & v, const DENSITY_DT& p) {v=p.density_dt;}
//
//    template <size_t n>
//    using SHARED_DENSITY_DT = SHARED_BASE<n,f1_t, density_dt_impl>;
//    using HOST_DENSITY_DT = HOST_BASE<f1_t, density_dt_impl>;
//    using DEV_DENSITY_DT = DEVICE_BASE<f1_t, density_dt_impl>;
//
//    //-------------------------------------------------------------------
//    // deviatoric stress tensor S
//    struct deviatoric_stress_impl
//    {
//        CUDAHOSTDEV static auto load(const m3_t & v) { return Particle<DSTRESS>(v); }
//        template <typename T> CUDAHOSTDEV static void store(m3_t & v, const T& p);
//        static constexpr f1_t defaultValue = 0;
//    };
//
//    template<typename T> CUDAHOSTDEV void deviatoric_stress_impl::store(m3_t &v, const T &p) {}
//    template<> CUDAHOSTDEV void inline deviatoric_stress_impl::store<DSTRESS>(m3_t & v, const DSTRESS& p) {v=p.dstress;}
//
//    template <size_t n>
//    using SHARED_DSTRESS = SHARED_BASE<n,m3_t, deviatoric_stress_impl>;
//    using HOST_DSTRESS = HOST_BASE<m3_t, deviatoric_stress_impl>;
//    using DEV_DSTRESS = DEVICE_BASE<m3_t, deviatoric_stress_impl>;
//
//
//    //-------------------------------------------------------------------
//    // time derivative of deviatoric stress tensor S
//    struct deviatoric_stress_dt_impl
//    {
//        CUDAHOSTDEV static auto load(const m3_t & v) { return Particle<DSTRESS_DT>(v); }
//        template <typename T> CUDAHOSTDEV static void store(m3_t & v, const T& p);
//        static constexpr f1_t defaultValue = 0;
//    };
//
//    template<typename T> CUDAHOSTDEV void deviatoric_stress_dt_impl::store(m3_t &v, const T &p) {}
//    template<> CUDAHOSTDEV inline void deviatoric_stress_dt_impl::store<DSTRESS_DT>(m3_t & v, const DSTRESS_DT& p) {v=p.dstress_dt;}
//
//    template <size_t n>
//    using SHARED_DSTRESS_DT = SHARED_BASE<n,m3_t, deviatoric_stress_dt_impl>;
//    using HOST_DSTRESS_DT = HOST_BASE<m3_t, deviatoric_stress_dt_impl>;
//    using DEV_DSTRESS_DT = DEVICE_BASE<m3_t, deviatoric_stress_dt_impl>;

#endif //GRASPH2_PARTICLE_BUFFER_IMPL_H
