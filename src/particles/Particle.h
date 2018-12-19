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

//!< class to identify particle bases
class particle_base {};

//!< flag to mark particle bases that is not the derivative of anything
class deriv_of_nothing {};

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
    static_assert( mpu::conjunction_v< std::is_base_of<particle_base,Args>...>,
            "Only use the Particle class with template arguments generated with the macro \"MAKE_PARTICLE_BASE\"! See file Particles.h."); //!< check if only valid bases are used for the particle

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
// concatinate particles

template <typename PA, typename PB>
struct particle_concat;

template <typename ...PA_bases, typename ...PB_bases>
struct particle_concat <Particle<PA_bases...>,Particle<PB_bases...>>
{
    using type = Particle<PA_bases...,PB_bases...>;
};

/**
 * @brief concatinate particles. Returns a particle that has all bases of both particles.
 */
template <typename PA, typename PB>
using particle_concat_t=particle_concat<PA,PB>;


//-------------------------------------------------------------------
/**
 * @brief Macro to generate a base class to be used with the particle class template.
 *          It will also generate a second base whose name is prepended by an "r" and contains a reference which binds to non reference base.
 * @param class_name the name you want the class to have
 * @param member_name the name the data member should have
 * @param mamber type the type the data member should have
 * @param is_deriv_of another particle base of which this on is the derivative.
 *          Or in other words: When integrating the value of this classes value will be used to integrate the value of is_deriv_of.
 */
#define MAKE_PARTICLE_BASE(class_name, member_name, member_type, is_deriv_of) \
class r ## class_name; \
class class_name : particle_base \
{ \
public: \
    using type = member_type; \
    using is_derivative_of = is_deriv_of; \
    using bind_ref_to_t = r ## class_name; \
    \
    class_name()=default;\
    CUDAHOSTDEV explicit class_name(type v) : member_name(std::move(v)) {}\
    CUDAHOSTDEV explicit class_name(no_baseclass_flag v) : member_name() {}\
    CUDAHOSTDEV class_name & operator=(const no_baseclass_flag & v) {return *this;}\
    \
    CUDAHOSTDEV type getMember() {return member_name;} \
    CUDAHOSTDEV void setMember(const type& v) {member_name = v;} \
    \
    type member_name; \
}; \
class r ## class_name : particle_base \
{ \
public: \
    using bind_ref_to_t = class_name ; \
    using is_derivative_of = is_deriv_of; \
    using type = member_type; \
    \
    CUDAHOSTDEV explicit r ## class_name(member_type & v) : member_name(v) {} \
    CUDAHOSTDEV explicit r ## class_name(class_name & v) : member_name(v. member_name) {} \
    CUDAHOSTDEV operator class_name() const {return class_name(member_name);} \
    CUDAHOSTDEV r ## class_name& operator=(const class_name & v) {member_name = v. member_name; return *this;} \
    CUDAHOSTDEV r ## class_name& operator=(const no_baseclass_flag & v) {return *this;}\
    \
    CUDAHOSTDEV type getMember() {return member_name;} \
    CUDAHOSTDEV void setMember(const type& v) {member_name = v;} \
    \
    member_type & member_name; \
}

#endif //MPUTILS_PARTICLE_H
