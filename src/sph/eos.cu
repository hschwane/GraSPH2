/*
 * GraSPH2
 * eos.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 * declaring functions to calculate various equation of states
 *
 */

// includes
//--------------------
#include "eos.h"
#include <math.h>
//--------------------

// namespace
//--------------------
namespace eos {
//--------------------

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
    return K0/dK0 * (pow(rho/rho0,dK0)-1);
}

}