/*
 * mpUtils
 * templateUtils.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */
#ifndef MPUTILS_TEMPLATEUTILS_H
#define MPUTILS_TEMPLATEUTILS_H

// includes
//--------------------
#include <type_traits>
#include "type_traitUtils.h"
//--------------------

// this file contains device/host functions that also need to compile when using gcc
//--------------------
#ifndef CUDAHOSTDEV
    #ifdef __CUDACC__
        #define CUDAHOSTDEV __host__ __device__
    #else
        #define CUDAHOSTDEV
    #endif
#endif
//--------------------

// namespace
//--------------------
namespace mpu {
//--------------------

//-------------------------------------------------------------------
// base_cast
// Thanks to Francesco Biscani!
namespace detail {
    // A structure that can be implicitly converted to any type.
    struct no_baseclass_flag {};

    // Machinery to cast the input reference x to one of its bases B.
    // This overload is selected only if B is a base of T.
    template<typename B, typename T, std::enable_if_t<std::is_base_of<B, T>::value, int> = 0>
    CUDAHOSTDEV const B &base_cast(const T &x)
    {
        return static_cast<const B &>(x);
    }

    // Overload taken if B is not a base of T. In this case we return
    // an object that can be converted to anything.
    template<typename B, typename T, std::enable_if_t<!std::is_base_of<B, T>::value, int> = 0>
    CUDAHOSTDEV no_baseclass_flag base_cast(const T &)
    {
        return no_baseclass_flag{};
    }
}

/**
 * @brief If B is a base class of x, x will be casted to B and returned as const reference.
 *      Otherwise a no_baseclass_flag will be returned.
 * @tparam B The type of the base class in question.
 * @tparam T The type of x.
 * @param x  A const reference to the object to be casted to type B.
 * @return   A const reference to x casted the type of B or a const reference to a default constructed B
 */
template<typename B, typename T>
CUDAHOSTDEV auto base_cast(const T &x)
{
    return detail::base_cast<B>(x);
};

/**
 * the type of a tuple that results from the concatination of tuples of the types in Ts
 */
template<typename...Ts>
using tuple_cat_t = decltype(std::tuple_cat(std::declval<Ts>()...));

/**
 * @brief tuple with all the types in Ts for which cond<T>::value is true
 */
template<template<class T> class cond, typename...Ts>
using remove_t = tuple_cat_t<
        typename std::conditional<
                cond<Ts>::value,
                std::tuple<Ts>,
                std::tuple<>
        >::type...
>;

/**
 * @brief shorthand for integral constant of type bool
 */
template <bool B>
using bool_constant = std::integral_constant<bool, B>;

/**
 * @brief negate the type trait B
 */
template<class B>
struct negation : bool_constant<!bool(B::value)> { };

}

#endif //MPUTILS_TEMPLATEUTILS_H
