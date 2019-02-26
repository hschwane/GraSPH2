/*
 * mpUtils
 * eos.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 * declaring functions to calculate various equation of states
 */
#ifndef MPUTILS_EOS_H
#define MPUTILS_EOS_H

// includes
//--------------------
#include "types.h"
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
namespace eos {
//--------------------

/**
 * @brief polytropic equation of state for gas. eg Springel 2002
 * @param rho current density
 * @param a constant of proportionality (in Springel 2002 this is a function of specific entropy)
 * @param gamma polytropic exponent gamma, for isothermal flow just omit gamma it will default to 1
 * @return current pressure
 */
CUDAHOSTDEV inline f1_t polytropic(f1_t rho, f1_t a, f1_t gamma);

/**
 * @brief polytropic equation of state for gas. eg Springel 2002
 * @param rho current density
 * @param a constant of proportionality (in Springel 2002 this is a function of specific entropy)
 * @param gamma polytropic exponent gamma, for isothermal flow just omit gamma it will default to 1
 * @return current pressure
 */
CUDAHOSTDEV inline f1_t polytropic(f1_t rho, f1_t a);

/**
 * @brief liquid equation of state eg Müller 2003
 * @param c2 speed of sound squared
 * @param rho current density
 * @param rho0 rest density
 * @return current pressure
 */
CUDAHOSTDEV inline f1_t liquid(f1_t rho, f1_t rho0, f1_t c2);

/**
 * @brief Murnaghan equation of state, simple eos that can be used to model solid materials. Values for K0 and dK0 can
 *          be found on the web and in different papers.
 * @param rho current density
 * @param rho0 rest density
 * @param K0 bulk modulus of the material
 * @param dK0 derivative of K0 with respect to pressure
 * @return current pressure
 */
CUDAHOSTDEV inline f1_t murnaghan(f1_t rho, f1_t rho0, f1_t K0, f1_t dK0);


// function definitions
//-------------------------------------------------------------------

CUDAHOSTDEV f1_t polytropic(f1_t rho, f1_t a, f1_t gamma)
{
    return a * pow(rho,gamma);
}

CUDAHOSTDEV f1_t polytropic(f1_t rho, f1_t a)
{
    return a * rho;
}

CUDAHOSTDEV f1_t liquid(f1_t rho, f1_t rho0, f1_t c2)
{
    return c2*(rho-rho0);
}

CUDAHOSTDEV f1_t murnaghan(f1_t rho, f1_t rho0, f1_t K0, f1_t dK0)
{
    return K0/dK0 * (pow(rho/rho0,dK0)-1.0_ft);
}

}

#endif //MPUTILS_EOS_H
