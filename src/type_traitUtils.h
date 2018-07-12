/*
 * mpUtils
 * templateUtils.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * some of this is availible in c++ 17 and above but not in c++14
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */
#ifndef PROJECT_DETECTOR_H
#define PROJECT_DETECTOR_H

// includes
//--------------------
#include <type_traits>
//--------------------

// namespace
//--------------------
namespace mpu {
//--------------------

//-------------------------------------------------------------------
/**
 * @brief Helper for using expression SFINAE by partial instantiation of a parameter which defaults to void. See c++17 void_t.
 *      This is availible in std since c++17.
 */
template <class... Ts>
using void_t = void;

//-------------------------------------------------------------------
// implement is_detected (availible in c++17)
// https://blog.tartanllama.xyz/detection-idiom/#fn:2
namespace detail {
    template <template <class...> class Trait, class Enabler, class... Args>
    struct is_detected : std::false_type{};

    template <template <class...> class Trait, class... Args>
    struct is_detected<Trait, void_t<Trait<Args...>>, Args...> : std::true_type{};
}

/**
 * @brief Evaluates to std::true_type if Trait can be instantiated by Args. Evaluates to std::false_type otherwise.
 *      This is availible in std since c++17.
 */
template <template <class...> class Trait, class... Args>
using is_detected = typename detail::is_detected<Trait, void, Args...>::type;

//-------------------------------------------------------------------
/**
 * @brief Shorthand for std::is_base_of<A,B>::value
 *      This is availible in std since c++17.
 */
template< class Base, class Derived >
inline constexpr bool is_base_of_v = std::is_base_of<Base, Derived>::value;

//-------------------------------------------------------------------
// is_list_initable
// Thanks to Francesco Biscani!
namespace detail {
    template<typename T, typename U>
    using list_init_t = decltype(::new(static_cast<void *>(nullptr)) T{std::declval<U>()});
}

/**
 * @brief Evaluates to std::true_type if it is possible to construct a T from a U using a initializer list.
 */
template <typename T, typename U>
using is_list_initable = is_detected<detail::list_init_t,T,U>;


}

#endif //PROJECT_DETECTOR_H
