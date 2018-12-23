/*
 * GraSPH2
 * Particle_base.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements a macro to to generate particle attributes
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */
#ifndef GRASPH2_PARTICLE_BASE_H
#define GRASPH2_PARTICLE_BASE_H

// includes
//--------------------
#include "mpUtils/mpCuda.h"
#include "particle_tmp_utils.h"
//--------------------

//!< class to identify particle bases
class particle_base {};

//!< flag to mark particle bases that is not the derivative of anything
class deriv_of_nothing {};

//-------------------------------------------------------------------
/**
 * @brief Macro to generate a base class to be used as a particle attribute with the particle class template.
 *          It will also generate a second base whose name is prepended by an "r" and contains a reference which binds to non reference base.
 * @param class_name the name you want the class to have
 * @param member_name the name the data member should have
 * @param mamber type the type the data member should have
 * @param is_deriv_of another particle base of which this on is the derivative.
 *          Or in other words: When integrating the value of this classes value will be used to integrate the value of is_deriv_of.
 */
#define MAKE_PARTICLE_ATTRIB(class_name, member_name, member_type, is_deriv_of) \
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

#endif //GRASPH2_PARTICLE_BASE_H
