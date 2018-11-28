/*
 * GraSPH2
 * models.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 * file defines helper functions for different physical sph models like viscosity and plasticity
 *
 */

// includes
//--------------------
#include "models.h"
//--------------------

f1_t artificialViscosity(const f1_t alpha, const f1_t density_i, const f1_t density_j, const f3_t &vij, const f3_t &rij,
        const f1_t r, const f1_t ci, const f1_t cj, const f1_t balsara_i, const f1_t balsara_j)
{
    const f1_t wij = dot(rij, vij) /r;
    f1_t II = 0;
    if(wij < 0)
    {
        const f1_t vsig = f1_t(ci+cj - 3.0f*wij);
        const f1_t rhoij = (density_i + density_j)*f1_t(0.5f);
        II = -0.25f * (balsara_i+balsara_j) * alpha * wij * vsig / rhoij;
    }
    return II;
}

void plasticity(m3_t &destress, const f1_t Y)
{
    // second invariant of deviatoric stress
    f1_t J2 = 0;
    for(uint e = 0; e < 9; ++e)
        J2 += destress(e) * destress(e);
    J2 *= 0.5f;

    const f1_t miese_f = min(  Y*Y /(3.0f*J2),1.0f);

    destress *= miese_f;
}

f1_t mohrCoulombYieldStress(const f1_t tanFrictionAngle, const f1_t pressure, const f1_t cohesion)
{
    return tanFrictionAngle * pressure + cohesion;
}
