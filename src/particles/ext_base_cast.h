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
#include <mpUtils/mpUtils.h>
#include <mpUtils/mpCuda.h>
//--------------------

// template function ext_base_cast which is used to cast a Particle to type that can be assigned to Base
// that can be assigned to Base if possible. if not the no_baseclass_flag will be returned
//-------------------------------------------------------------------
// A flag structure that will be returned if no suitable base class was found.
struct no_baseclass_flag {};

template <class T>
using sizeMember_t = decltype(std::declval<T>().size());

// This overload is selected only if B is a base of T.
template <typename B, typename T, std::enable_if_t<mpu::is_base_of_v<B,T>, int> = 0>
CUDAHOSTDEV inline const B &ext_base_cast(const T &x)
{
    return static_cast<const B &>(x);
}

// this overload is selected B has a type bind_ref_to_t which is a base of B and x is const
template <typename B, typename T, std::enable_if_t<mpu::is_base_of_v<typename B::bind_ref_to_t,T> && !std::is_base_of<B,T>::value , int> = 0>
CUDAHOSTDEV inline auto &ext_base_cast(const T &x)
{
    return static_cast<const typename B::bind_ref_to_t &>(x);
}

// this overload is selected B has a type bind_ref_to_t which is a base of B and x is not const
template <typename B, typename T, std::enable_if_t<mpu::is_base_of_v<typename B::bind_ref_to_t,T>, int> = 0>
CUDAHOSTDEV inline auto &ext_base_cast(T &x)
{
    return static_cast<typename B::bind_ref_to_t &>(x);
}

// Overload taken if B is not a base of T. In this case we return the no baseclass flag
template <typename B, typename T, std::enable_if_t< !std::is_base_of<B,T>::value
                                                    && !std::is_base_of<typename B::bind_ref_to_t,T>::value
                                                    && !mpu::is_detected<sizeMember_t,B>(), int> = 0>
CUDAHOSTDEV inline no_baseclass_flag ext_base_cast(const T &x)
{
    return no_baseclass_flag{};
}

// Overload taken if B is not a base of T and B has a member called size.
// In this case we return the size so the new object can
// be constructed with an appropriate size
template <typename B, typename T, std::enable_if_t< !std::is_base_of<B,T>::value
                                                    && !std::is_base_of<typename B::bind_ref_to_t,T>::value
                                                    && mpu::is_detected<sizeMember_t,B>(), int> = 0>
CUDAHOSTDEV inline size_t ext_base_cast(const T &x)
{
    return x.size();
}

#endif //MPUTILS_EXT_BASE_CAST_H
