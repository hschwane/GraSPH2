/*
 * mpUtils
 * Particle.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */
#ifndef MPUTILS_PARTICLE_H
#define MPUTILS_PARTICLE_H

// includes
//--------------------
#include <mpUtils/mpUtils.h>
#include <mpUtils/mpCuda.h>
#include "ext_base_cast.h"
//--------------------

//-------------------------------------------------------------------
/**
 * class Particle
 *
 * enables handling and manipulation of the attributes of a single Particle
 *
 * usage:
 * Use the Macro MAKE_PARTICLE_BASE to create different particle attributes. Or use the predefined ones in Particles.h
 * Then pass the name of all the attributes you want to manipulate as template arguments.
 * Example: Particle<POS,MASS> p; p will have a 3D position and a mass.
 * You can use Particle<rPOS,rMASS> ref = p; to create a particle that holds references to the attributes of p.
 *
 */
template <typename... Args>
class Particle : public Args...
{
public:
    Particle()= default; //!< default construct particle values are undefined

    template <typename... T, std::enable_if_t< mpu::conjunction<mpu::is_list_initable<Args, T&&>...>::value, int> = 0>
    CUDAHOSTDEV
    explicit Particle(T && ... args) : Args(std::forward<T>(args))... {} //!< construct a particle from its attributes

    template <typename... T>
    CUDAHOSTDEV
    Particle(const Particle<T...> &b) : Args(ext_base_cast<Args>(b))... {} //!< construct a particle from another particle with different attributes

    template <typename... T>
    CUDAHOSTDEV Particle<Args...>& operator=(const Particle<T...> &b)
    {
        int t[] = {0, ((void)Args::operator=(ext_base_cast<Args>(b)),1)...};
        (void)t[0]; // silence compiler warning about t being unused
        return *this;
    }
};

//-------------------------------------------------------------------
//!< macro to generate classes that hold the members of the particle class
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

#endif //MPUTILS_PARTICLE_H
