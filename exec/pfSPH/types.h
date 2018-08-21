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
#include <mpUtils.h>
//--------------------

#define SINGLE_PRECISION
//#define DOUBLE_PRECISION

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
#else
using f1_t=float;
using f2_t=float2;
using f3_t=float3;
using f4_t=float4;
#endif


#endif //MPUTILS_TYPESETTINGS_H
