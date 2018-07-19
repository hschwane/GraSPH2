/*
 * mpUtils
 * ext_base_cast.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */
#ifndef MPUTILS_EXT_BASE_CAST_H
#define MPUTILS_EXT_BASE_CAST_H

// includes
//--------------------
#include <mpUtils.h>
//--------------------

// template function ext_base_cast which is used to cast a Particle to type that can be assigned to Base
// that can be assigned to Base if possible. if not the no_baseclass_flag will be returned
//-------------------------------------------------------------------
// A flag structure that will be returned if no suitable base class was found.
struct no_baseclass_flag {};

// This overload is selected only if B is a base of T.
template <typename B, typename T, std::enable_if_t<mpu::is_base_of_v<B,T>, int> = 0>
CUDAHOSTDEV inline const B &ext_base_cast(const T &x)
{
    return static_cast<const B &>(x);
}

// this overload is selected B has a type bind_ref_to_t which is a base of B and x is const
template <typename B, typename T, std::enable_if_t<mpu::is_base_of_v<typename B::bind_ref_to_t,T>, int> = 0>
CUDAHOSTDEV inline auto &ext_base_cast(const T &x)
{
    return static_cast<const typename B::bind_ref_to_t &>(x);
}

// this overload is selected B has a type bind_ref_to_t which is a base of B and x is not const
template <typename B, typename T, std::enable_if_t<mpu::is_base_of_v<typename B::bind_ref_to_t,T>, int> = 0>
CUDAHOSTDEV inline auto &base_cast(T &x)
{
    return static_cast<typename B::bind_ref_to_t &>(x);
}

// Overload taken if B is not a base of T. In this case we return
// an object that can be converted to anything.
template <typename B, typename T, std::enable_if_t<!std::is_base_of<B,T>::value && !std::is_base_of<typename B::bind_ref_to_t,T>::value, int> = 0>
CUDAHOSTDEV inline no_baseclass_flag ext_base_cast(const T &)
{
    return no_baseclass_flag{};
}


#endif //MPUTILS_EXT_BASE_CAST_H
