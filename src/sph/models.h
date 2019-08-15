/*
 * GraSPH2
 * models.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 * file declares helper functions for different physical sph models like viscosity and plasticity
 *
 */
#ifndef GRASPH2_MODELS_H
#define GRASPH2_MODELS_H

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

/**
 * @brief calculates II, the viscosity factor for artificial viscosity after Monaghan 1997.
 *          Viscosity vanishes if particles move away from each other. To remove shear viscosity
 *          use an additional balsara switch.
 * @param alpha the strength of the viscosity should be between 0 and 1
 * @param density_i the density of particle i
 * @param density_j the density of particle j
 * @param vij the relative velocity of the particles
 * @param rij the relative position of the particles
 * @param r the length of rij
 * @param ci the speed of sound of particle i
 * @param cj the speed of sound of particle j
 * @param balsara_i balsara-switch value of particle i
 * @param balsara_j balsara-switch value of particle j
 * @return returns the viscosity factor II
 */
CUDAHOSTDEV inline f1_t artificialViscosity(f1_t alpha,
                                        f1_t density_i, f1_t density_j,
                                        const f3_t& vij,  const f3_t& rij,
                                        f1_t r,
                                        f1_t ci, f1_t cj,
                                        f1_t  balsara_i=1.0f, f1_t balsara_j=1.0f);

/**
 * @brief computes the balsara switch value for a particle. (See Balsara 1995)
 * @param divv divergence of the velocity field
 * @param curlv curl of the velocity field
 * @param c speed of sound
 * @param h smoothing length
 * @return the value of the balsara switch for this particle
 */
CUDAHOSTDEV inline f1_t balsaraSwitch(f1_t divv, const f3_t& curlv, f1_t c, f1_t h);

/**
 * @brief limits the amount of deviatoric stress using the von miese yield criterion.
 *          See Schäfer et al 2016. To get more complex models like mohr-coulomb calculate
 *          Y for each particle individually instead of passing a material constant.
 * @param destress the deviatoric stress tensor of particle i to be limited (this will be changed in place)
 * @param pressure the pressure of particle i
 * @param Y the material dependend yield stress. Lower Y will result in an material that dow not return to it's original shape earlier.
 *          High Y will result in fully elastic body.
 */
CUDAHOSTDEV inline void plasticity(m3_t &destress, f1_t Y);

/**
 * @brief calculates the yield stress using the mohr-coulomb model
 * @param tanFrictionAngle tangent of the inner friction angle, maximal angle of attack for a force without material sliding away / failing
 * @param pressure the current pressure of particle i
 * @param cohesion cohesion of the material, higher values increases "stickyness"
 * @return the yield stress Y to be used for the plasticyty funciton above
 */
CUDAHOSTDEV inline f1_t mohrCoulombYieldStress(f1_t tanFrictionAngle, f1_t pressure, f1_t cohesion);

/**
 * @brief Calculates the artificial stress for one partical. Artificial Stress can be used to counteract
 *         the tensile instability. The total artificial stress that needs to be added to the stress when calculating
 *         the acceleration is the sum of the artificial stress for both particles multiplied by pow( W(r)/W(mpd)  ,n)
 *         Where n is a material property and mpd the mean particle seperation.
 * @param mateps a material property, typically between 0.2 and 0.4
 * @param sigOverRho the deviatoric stress sigma of a particle divided by the square of its density
 * @return the influence of one particles stress to the artificial pressure during an interaction
 */
CUDAHOSTDEV inline m3_t artificialStress(f1_t mateps, const m3_t& sigOverRho);

/**
 * @brief Adds the influence of the interaction between particle i and j to the strain rate and rotation rate tensors
 *         of particle i. Strain rate and rotation rate are needed to calculate the time derivative of the deviatoric stress tensor.
 * @param edot the strain rate tensor of i
 * @param rdot the rotation rate tensor of i
 * @param mass_j the mass of j
 * @param density_i the density of i
 * @param vij the relative velocity of the particles
 * @param gradw the kernel gradient
 */
CUDAHOSTDEV inline void addStrainRateAndRotationRate(m3_t& edot, m3_t& rdot,
                                               f1_t mass_j, f1_t density_j,
                                               const f3_t& vij, const f3_t& gradw);

/**
 * @brief calculates the time derivative of the deviatoric stress tensor from the strain rate tensor and the rotation rate tensor.
 * @param edot the strain rate tensor
 * @param rdot the rotation rate tensor
 * @param dstress the current deviatoric stress tensor
 * @param shear the shear modulus of the material
 * @return the time derivative of the deviatoric stress tensor
 */
CUDAHOSTDEV inline m3_t dstress_dt(const m3_t& edot, const m3_t& rdot, const m3_t& dstress, f1_t shear);


// function definitions
//-------------------------------------------------------------------

f1_t artificialViscosity(const f1_t alpha, const f1_t density_i, const f1_t density_j, const f3_t &vij, const f3_t &rij,
                         const f1_t r, const f1_t ci, const f1_t cj, const f1_t balsara_i, const f1_t balsara_j)
{
    const f1_t wij = dot(rij, vij) / r;
    f1_t II = 0.0_ft;
    if(wij < 0.0_ft)
    {
        const f1_t vsig = f1_t(ci + cj - 3.0_ft * wij);
        const f1_t rhoij = (density_i + density_j) * f1_t(0.5_ft);
        II = -0.25_ft * (balsara_i + balsara_j) * alpha * wij * vsig / rhoij;
    }
    return II;
}

f1_t balsaraSwitch(const f1_t divv, const f3_t& curlv, const f1_t c, const f1_t h)
{
    const f1_t absdiv = fabs(divv);
    const f1_t abscurl = length(curlv);
    return absdiv / (absdiv + abscurl + 0.0001_ft * c/h);
}

void plasticity(m3_t &destress, const f1_t Y)
{
    // second invariant of deviatoric stress
    f1_t J2 = 0.0_ft;
    for(uint e = 0; e < 9; ++e)
        J2 += destress(e) * destress(e);
    J2 *= 0.5_ft;

    const f1_t miese_f = Y*Y / (3.0_ft * J2);
    if(miese_f < 1)
        destress *= miese_f;
}

f1_t mohrCoulombYieldStress(const f1_t tanFrictionAngle, const f1_t pressure, const f1_t cohesion)
{
    return tanFrictionAngle * pressure + cohesion;
}

m3_t artificialStress(const f1_t mateps, const m3_t &sigOverRho)
{
    m3_t arts;
    for(size_t e = 0; e < 9; e++)
        arts(e) = ((sigOverRho(e) > 0) ? (-mateps * sigOverRho(e)) : 0.0_ft);
    return arts;
}

void addStrainRateAndRotationRate(m3_t &edot, m3_t &rdot, const f1_t mass_j, const f1_t density_j, const f3_t &vij,
                                  const f3_t &gradw)
{
    f1_t tmp = -0.5_ft * mass_j / density_j;
    edot[0][0] += tmp * (vij.x * gradw.x + vij.x * gradw.x);
    edot[0][1] += tmp * (vij.x * gradw.y + vij.y * gradw.x);
    edot[0][2] += tmp * (vij.x * gradw.z + vij.z * gradw.x);
    edot[1][0] += tmp * (vij.y * gradw.x + vij.x * gradw.y);
    edot[1][1] += tmp * (vij.y * gradw.y + vij.y * gradw.y);
    edot[1][2] += tmp * (vij.y * gradw.z + vij.z * gradw.y);
    edot[2][0] += tmp * (vij.z * gradw.x + vij.x * gradw.z);
    edot[2][1] += tmp * (vij.z * gradw.y + vij.y * gradw.z);
    edot[2][2] += tmp * (vij.z * gradw.z + vij.z * gradw.z);

    rdot[0][0] += tmp * (vij.x * gradw.x - vij.x * gradw.x);
    rdot[0][1] += tmp * (vij.x * gradw.y - vij.y * gradw.x);
    rdot[0][2] += tmp * (vij.x * gradw.z - vij.z * gradw.x);
    rdot[1][0] += tmp * (vij.y * gradw.x - vij.x * gradw.y);
    rdot[1][1] += tmp * (vij.y * gradw.y - vij.y * gradw.y);
    rdot[1][2] += tmp * (vij.y * gradw.z - vij.z * gradw.y);
    rdot[2][0] += tmp * (vij.z * gradw.x - vij.x * gradw.z);
    rdot[2][1] += tmp * (vij.z * gradw.y - vij.y * gradw.z);
    rdot[2][2] += tmp * (vij.z * gradw.z - vij.z * gradw.z);
}

m3_t dstress_dt(const m3_t &edot, const m3_t &rdot, const m3_t &dstress, const f1_t shear)
{
    m3_t dstress_dt;
    for(int d = 0; d < 3; ++d)
        for(int e = 0; e < 3; ++e)
        {
            dstress_dt[d][e] = 2.0_ft * shear * edot[d][e];
            for(int f = 0; f < 3; f++)
            {
                if(d == e)
                    dstress_dt[d][e] -= 2.0_ft * shear * edot[f][f] / 3.0_ft;

                dstress_dt[d][e] += dstress[d][f] * rdot[f][e];
                dstress_dt[d][e] -= dstress[f][e] * rdot[d][f];
            }

        }
    return dstress_dt;
}


#endif //GRASPH2_MODELS_H
