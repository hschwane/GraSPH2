/*
 * mpUtils
 * Particle_impl.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */
#ifndef MPUTILS_PARTICLE_IMPL_H
#define MPUTILS_PARTICLE_IMPL_H

// includes
//--------------------
#include <mpUtils.h>
#include "ext_base_cast.h"
//--------------------

//-------------------------------------------------------------------
// macro to generate classes that hold the members of the particle class
#define MAKE_PARTICLE_BASE(class_name, member_name, member_type) \
class r ## class_name; \
class class_name \
{ \
public: \
    member_type member_name; \
    class_name()=default;\
    CUDAHOSTDEV explicit class_name(member_type v) : member_name(std::move(v)) {}\
    CUDAHOSTDEV explicit class_name(no_baseclass_flag v) : member_name() {}\
    CUDAHOSTDEV class_name & operator=(const no_baseclass_flag & v) {return *this;}\
    using bind_ref_to_t = r ## class_name; \
}; \
class r ## class_name \
{ \
public: \
    member_type & member_name; \
    CUDAHOSTDEV explicit r ## class_name(member_type & v) : member_name(v) {} \
    CUDAHOSTDEV explicit r ## class_name(class_name & v) : member_name(v. member_name) {} \
    CUDAHOSTDEV operator class_name() const {return class_name(member_name);} \
    CUDAHOSTDEV r ## class_name& operator=(const class_name & v) {member_name = v. member_name; return *this;} \
    CUDAHOSTDEV r ## class_name& operator=(const no_baseclass_flag & v) {return *this;}\
    using bind_ref_to_t = class_name ; \
}

//-------------------------------------------------------------------
// macro to generate classes that hold the members of the SharedParticles class
#define MAKE_SHARED_PARTICLE_BASE(class_name, member_type, particle_base1, load_base1, store_base1, particle_base2, load_base2, store_base2, particle_base3, load_base3, store_base3, particle_base4, load_base4, store_base4) \
template <size_t n> \
class class_name \
{  \
public: \
    using particle_t=Particle<particle_base1,particle_base2,particle_base3,particle_base4>; \
    __device__ class_name() { __shared__ member_type mem[n]; m_sm = mem;} \
    template<typename... Args> \
    __device__ void loadParticle(size_t id, Particle<Args ...> & p) {member_type v = m_sm[id]; p = particle_t (load_base1,load_base2,load_base3,load_base4);} \
    __device__ void storeParticle(size_t id, const particle_t & p) {storeParticleHelper(id,p);}\
private: \
    member_type * m_sm; \
    \
    template<typename... Args> \
    __device__ void storeParticleHelper(size_t id, const Particle<Args...>& p) {int t[] = {0, ((void)storeBase(id, ext_base_cast<Args>(p)),1)...};} \
    __device__ void storeBase(size_t id, const particle_base1 & b) {store_base1;}\
    __device__ void storeBase(size_t id, const particle_base2 & b) {store_base2;}\
    __device__ void storeBase(size_t id, const particle_base3 & b) {store_base3;}\
    __device__ void storeBase(size_t id, const particle_base4 & b) {store_base4;}\
    __device__ void storeBase(size_t id, const no_baseclass_flag & b) {} \
}

#endif //MPUTILS_PARTICLE_IMPL_H
