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

// this overload is selected if B has a type bind_ref_to_t which is a base of B and x is const
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

// Overload taken if B is not a base of T and T has a member called size.
// In this case we return the size so the new object can
// be constructed with an appropriate size
template <typename B, typename T, std::enable_if_t< !std::is_base_of<B,T>::value
                                                    && !std::is_base_of<typename B::bind_ref_to_t,T>::value
                                                    && mpu::is_detected<sizeMember_t,T>(), int> = 0>
CUDAHOSTDEV inline size_t ext_base_cast(const T &x)
{
    return x.size();
}


//-------------------------------------------------------------------
/**
 * @brief build a index list for the types in Tuple based of the position of the same type in Reference
 */
template <typename Tuple, typename Reference>
struct build_comp_index_list;

template <typename ... Ts, typename Reference>
struct build_comp_index_list <std::tuple<Ts...>, Reference>
{
    using type = std::index_sequence< mpu::index_of_v<Ts, Reference>... >;
};

template <typename Tuple, typename Reference>
using build_comp_index_list_t = typename build_comp_index_list<Tuple,Reference>::type;


//-------------------------------------------------------------------
/**
 * @brief check if the order of types in Tuple is the same as in Reference and there are no duplicates
 */
template <typename Tuple, typename Reference>
static constexpr bool checkOrder_v = mpu::is_strict_rising< build_comp_index_list_t<Tuple,Reference> >::value;


//-------------------------------------------------------------------
/**
 * @brief build tuple using indices and a reference tuple, for each number from the list is used as an index, then the type at
 *          that index in tuple Reference is used to build the new tuple
 * @tparam Reference the reference tuple from which the types are chosen
 * @tparam IndexList a std::integer_list of indexes to specify which types should be chosen from Reference
 */
template <typename Reference, typename IndexList>
struct make_tpl;

template <typename Reference, typename T, T first, T ... Ints>
struct make_tpl<Reference, std::integer_sequence<T,first,Ints...>>
{
    using head_tuple = typename std::tuple< typename std::tuple_element<first,Reference>::type>; // do work
    using tail_tuple = typename make_tpl<Reference,std::integer_sequence<T,Ints...>>::type; // recursive call
    using type = mpu::tuple_cat_t< head_tuple , tail_tuple >; // put together
};

template <typename Reference, typename T>
struct make_tpl<Reference, std::integer_sequence<T>>
{
    using type = std::tuple<>;
};

template <typename IndexList, typename Reference>
using make_tpl_t = typename make_tpl<Reference,IndexList>::type;


//-------------------------------------------------------------------
/**
 * @brief reorder Tuple following the order of types in Reference and remove duplicates
 */
template <typename Tuple, typename Reference>
using reorderd_t = make_tpl_t< mpu::is_rm_duplicates_t< mpu::is_sort_asc_t< build_comp_index_list_t<Tuple,Reference>>>, Reference>;


//-------------------------------------------------------------------
/**
 * @brief Concatenate particles or particle buffers. Returns a particle or buffer that has all bases of both particles/buffers.
 */
template <typename PA, typename PB>
struct particle_concat_impl;

template <template<typename ...T> class Class, typename ...PA_bases, typename ...PB_bases>
struct particle_concat_impl < Class<PA_bases...>, Class<PB_bases...>>
{
    using type = Class<PA_bases...,PB_bases...>;
};

template <typename ...Particles>
struct particle_concat;

template <typename first ,typename ...rest>
struct particle_concat<first,rest...>
{
    using type = particle_concat_impl<first, particle_concat<rest...>>;
};

template <typename first>
struct particle_concat<first>
{
    using type = first;
};

/**
 * @brief Concatenate particles or particle buffers. Returns a particle or buffer that has all bases of both particles/buffers.
 */
template <typename ...Particles>
using particle_concat_t= typename particle_concat<Particles...>::type;



//-------------------------------------------------------------------
/**
 * @brief Create a tuple of types from the attributes of a particle or buffer
 */
template <typename PA>
struct particle_to_tuple;

template <template<typename ...T> class Class, typename ...PA_bases>
struct particle_to_tuple < Class<PA_bases...>>
{
    using type = std::tuple<PA_bases...>;
};

/**
 * @brief Create a tuple of types from the attributes of a particle or buffer
 */
template <typename PA>
using particle_to_tuple_t =typename particle_to_tuple<PA>::type;

#endif //MPUTILS_EXT_BASE_CAST_H
