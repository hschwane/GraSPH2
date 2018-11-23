/*
 * mpUtils
 * kernel.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */
#ifndef MPUTILS_KERNEL_H
#define MPUTILS_KERNEL_H

// includes
//--------------------
#include "../types.h"
#include <math.h>
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

namespace kernel {

namespace detail {

/**
 * @brief calculates the dimension dependent prefactor of the spline functions
 * @tparam dimension Dim::one, Dim::two, or Dim::three
 * @param h the smoothing length
 * @return the dimension dependent prefactor of the spline functions
 */
template<Dim dimension>
CUDAHOSTDEV f1_t splinePrefactor(f1_t h) {throw std::logic_error("illegal dimension for this kernel");}

template<>
CUDAHOSTDEV f1_t splinePrefactor<Dim::one>(f1_t h)
{
    return 4.0 / (3.0 * h);
}

template<>
CUDAHOSTDEV f1_t splinePrefactor<Dim::two>(f1_t h)
{
    return 40.0 / (7.0 * M_PI * h * h);
}

template<>
CUDAHOSTDEV f1_t splinePrefactor<Dim::three>(f1_t h)
{
    return 8.0 / (M_PI * h * h * h);
}

/**
* @brief calculates the dimension dependent prefactor of the first derivatives spline functions
* @tparam dimension Dim::one, Dim::two, or Dim::three
* @param h the smoothing length
* @return the dimension dependent prefactor of the spline functions
*/
template<Dim dimension>
CUDAHOSTDEV f1_t dsplinePrefactor(f1_t h) {} // {throw std::logic_error("illegal dimension for this kernel");}

template<>
CUDAHOSTDEV f1_t dsplinePrefactor<Dim::one>(f1_t h)
{
    return 24.0 / (3.0 * h * h);
}

template<>
CUDAHOSTDEV f1_t dsplinePrefactor<Dim::two>(f1_t h)
{
    return 240.0 / (7.0 * M_PI * h * h * h);
}

template<>
CUDAHOSTDEV f1_t dsplinePrefactor<Dim::three>(f1_t h)
{
    return 48.0 / (M_PI * h * h * h * h);
}

/**
* @brief calculates the dimension dependent prefactor of the first derivatives spiky kernel
* @tparam dimension Dim::one, Dim::two, or Dim::three
* @param h the smoothing length
* @return the dimension dependent prefactor of the spiky kernel
*/
template<Dim dimension>
CUDAHOSTDEV f1_t dspikyPrefactor(f1_t h) {}// {throw std::logic_error("illegal dimension for this kernel");}

template<>
CUDAHOSTDEV f1_t dspikyPrefactor<Dim::two>(f1_t h)
{
    return -30.0 / (M_PI * h*h*h*h*h);
}

template<>
CUDAHOSTDEV f1_t dspikyPrefactor<Dim::three>(f1_t h)
{
    return -45.0 / (M_PI * h*h*h*h*h*h);
}

}

// -----------------------------------------------------------------------------------------------------
// B-Spline

/**
 * @brief calculates the dimension dependent prefactor of the spline functions
 * @tparam dimension Dim::one, Dim::two, or Dim::three
 * @param h the smoothing length
 * @return the dimension dependent prefactor of the spline functions
 */
using detail::splinePrefactor;

/**
 * @brief calculates the b-spline kernel function Monaghan & Lattanzio (1985)
 * @param factor dimension dependend precomputed prefactor (use splinePrefactor)
 * @param r the distance from the sample point r
 * @param h the smoothing length
 * @return value of the spline kernel function
 */
CUDAHOSTDEV inline f1_t Wspline(f1_t r, f1_t h, f1_t factor)
{
    f1_t q = r / h;
    return factor *( (q <= 1.0) ? ((q < 0.5) ? (6 * q * q * q - 6 * q * q + 1)
                                     : 2 * (1 - q) * (1 - q) * (1 - q))
                        : f1_t(0.0));
}

/**
 * @brief calculates the b-spline kernel function Monaghan & Lattanzio (1985)
 * @tparam dimension Dim::one, Dim::two, or Dim::three
 * @param r the distance from the sample point r
 * @param h the smoothing length
 * @return value of the spline kernel function
 */
template<Dim dimension>
CUDAHOSTDEV f1_t Wspline(f1_t r, f1_t h)
{
    return Wspline(r,h,splinePrefactor<dimension>(h));
}

// -----------------------------------------------------------------------------------------------------
// B-Spline derivative

/**
* @brief calculates the dimension dependent prefactor of the first derivatives spline functions
* @tparam dimension Dim::one, Dim::two, or Dim::three
* @param h the smoothing length
* @return the dimension dependent prefactor of the spline functions
*/
using detail::dsplinePrefactor;

/**
 * @brief calculates the first derivative of the b-spline kernel function Monaghan & Lattanzio (1985)
 * @param factor dimension dependend precomputed prefactor (use dsplinePrefactor)
 * @param r the distance from the sample point r
 * @param h the smoothing length
 * @return value of the spline function derivative
 */
CUDAHOSTDEV inline f1_t dWspline(f1_t r, f1_t h, f1_t factor)
{
    f1_t q = r / h;
    return factor *( (q <= 1.0) ? ((q < 0.5) ? (3 * q * q - 2 * q)
                                     : -1 * (1 - q) * (1 - q))
                        : f1_t(0.0));
}

/**
 * @brief calculates the first derivative of the b-spline kernel function Monaghan & Lattanzio (1985)
 * @tparam dimension Dim::one, Dim::two, or Dim::three
 * @param r the distance from the sample point r
 * @param h the smoothing length
 * @return value of the spline function derivative
 */
template<Dim dimension>
CUDAHOSTDEV f1_t dWspline(f1_t r, f1_t h)
{
    return dWspline(r,h,detail::dsplinePrefactor<dimension>(h));
}

// -----------------------------------------------------------------------------------------------------
// spiky derivative

/**
* @brief calculates the dimension dependent prefactor of the first derivatives spiky kernel
* @tparam dimension Dim::one, Dim::two, or Dim::three
* @param h the smoothing length
* @return the dimension dependent prefactor of the spiky kernel
*/
using detail::dspikyPrefactor;

/**
 * @brief calculate the first derivative of the spiky kernel function Desbrun 1996
 * @param r2 the square of the distance to the sample point
 * @param h the smoothing length
 * @param factor the precomputed prefactor (use dspikyPrefactor)
 * @return the value of the poly6 derivative
 */
CUDAHOSTDEV inline f1_t dWspiky(f1_t r, f1_t h, f1_t factor)
{
    return (r < h) ? factor * (h-r) * (h-r) : 0.0;
}

/**
 * @brief calculate the first derivative of the spiky kernel function Desbrun 1996
 * @param r2 the square of the distance to the sample point
 * @param h the smoothing length
 * @return the value of the poly6 derivative
 */
template<Dim dimension>
CUDAHOSTDEV f1_t dWspiky(f1_t r, f1_t h)
{
    return dWspiky(r,h,dspikyPrefactor<dimension>(h));
}

// -----------------------------------------------------------------------------------------------------
// poly 6

/**
 * @brief calculate the prefactor of the poly6 kernel which can be saved for performance reasons
 *          this is only availible for 3d right now
 * @param h the smoothing length
 * @return the prefactor of the poly6 kernel
 */
CUDAHOSTDEV inline f1_t Wpoly6Factor(f1_t h)
{
    return (315.0 / (64.0 * M_PI * pow(h, 9.0)));
}

/**
 * @brief calculate the poly6 kernel function Müller et al 2003
 * @param r2 the square of the distance to the sample point
 * @param h the smoothing length
 * @return the value of the poly6 kernel
 */
CUDAHOSTDEV inline f1_t Wpoly6(f1_t r2, f1_t h)
{
    const f1_t h2 = h * h;
    const f1_t hmr = h2 - r2;
    return (r2 < h2) ? Wpoly6Factor(h) * hmr * hmr * hmr : 0.0;
}

/**
 * @brief calculate the poly6 kernel function Müller et al 2003 using a precomputed prefactor
 * @param r2 the square of the distance to the sample point
 * @param h2 the square of the smoothing length
 * @param factor the precomputed prefactor
 * @return the value of the poly6 kernel
 */
CUDAHOSTDEV inline f1_t Wpoly6(f1_t r2, f1_t h2, f1_t factor)
{
    const f1_t hmr = h2 - r2;
    return (r2 < h2) ? factor * hmr * hmr * hmr : 0.0;
}

// -----------------------------------------------------------------------------------------------------
// poly 6 derivative

/**
 * @brief calculate the prefactor of the derivative of the poly6 kernel which can be saved for performance reasons
 * only availible for 3d right now
 * @param h the smoothing length
 * @return the prefactor of the poly6 kernel
 */
CUDAHOSTDEV inline f1_t dWpoly6Factor(f1_t h)
{
    return (-945.0 / (32.0 * M_PI * pow(h, 9.0)));
}

/**
 * @brief calculate the first derivative of the poly6 kernel function Müller et al 2003
 * @param r2 the square of the distance to the sample point
 * @param h the smoothing length
 * @return the value of the poly6 derivative
 */
CUDAHOSTDEV inline f1_t dWpoly6(f1_t r2, f1_t h)
{
    const f1_t h2 = h * h;
    const f1_t hmr = h2 - r2;
    return (r2 < h2) ? dWpoly6Factor(h) * hmr * hmr : 0.0;
}

/**
 * @brief calculate the poly6 kernel function Müller et al 2003 using a precomputed prefactor
 * @param r2 the square of the distance to the sample point
 * @param h the smoothing length
 * @param factor the precomputed prefactor
 * @return the value of the poly6 derivative
 */
CUDAHOSTDEV inline f1_t dWpoly6(f1_t r2, f1_t h2, f1_t factor)
{
    const f1_t hmr = h2 - r2;
    return (r2 < h2) ? factor * hmr * hmr : 0.0;
}


}

#endif //MPUTILS_KERNEL_H
