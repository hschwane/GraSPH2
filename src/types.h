/*
 * mpUtils
 * typeSettings.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */
#ifndef MPUTILS_TYPESETTINGS_H
#define MPUTILS_TYPESETTINGS_H

// includes
//--------------------
#include <mpUtils/mpCuda.h>
#include "precisionSettings.h"
//--------------------

/**
 * @brief enum class to specialize templates on different dimensions
 */
enum class Dim
{
    one,
    two,
    three
};

//-------------------------------------------------------------------
// define the data types used for the simulation
#if defined(DOUBLE_PRECISION)
    using f1_t=double;
    using f2_t=double2;
    using f3_t=double3;
    using f4_t=double4;
    using m2_t=mpu::Mat<double,2,2>;
    using m3_t=mpu::Mat<double,3,3>;
    using m4_t=mpu::Mat<double,4,4>;
    constexpr double operator ""_ft(long double d) noexcept
    {
        return double(d);
    }
#else
    using f1_t=float;
    using f2_t=float2;
    using f3_t=float3;
    using f4_t=float4;
    using m2_t=mpu::Mat<float,2,2>;
    using m3_t=mpu::Mat<float,3,3>;
    using m4_t=mpu::Mat<float,4,4>;
    constexpr float operator ""_ft(long double d) noexcept
    {
        return float(d);
    }
#endif

//!< helper functions to check how many floats or doubles are in a datatype
template<typename A> int getDim();

//-------------------------------------------------------------------
// template function definitions

template<typename A> inline int getDim()
{return 0;}

template<> inline int getDim<f1_t>()
{return 1;}

template<> inline int getDim<f2_t>()
{return 2;}

template<> inline int getDim<f3_t>()
{return 3;}

template<> inline int getDim<f4_t>()
{return 4;}

template<> inline int getDim<m2_t>()
{return 4;}

template<> inline int getDim<m3_t>()
{return 9;}

template<> inline int getDim<m4_t>()
{return 16;}

#endif //MPUTILS_TYPESETTINGS_H
