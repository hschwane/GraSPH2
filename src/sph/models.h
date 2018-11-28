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
 *          Viscosity vanishes if particles move away from each other. To remove shear velocity
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
CUDAHOSTDEV f1_t artificialViscosity(f1_t alpha,
                                        f1_t density_i, f1_t density_j,
                                        const f3_t& vij,  const f3_t& rij,
                                        f1_t r,
                                        f1_t ci, f1_t cj,
                                        f1_t  balsara_i=1.0f, f1_t balsara_j=1.0f);

/**
 * @brief limits the amount of deviatoric stress using the von miese yield criterion.
 *          See Schäfer et al 2016. To get more complex models like mohr-coulomb calculate
 *          Y for each particle individually instead of passing a material constant.
 * @param destress the deviatoric stress tensor of particle i to be limited (this will be changed in place)
 * @param pressure the pressure of particle i
 * @param Y the material dependend yield stress. Lower Y will result in an material that dow not return to it's original shape earlier.
 *          High Y will result in fully elastic body.
 */
CUDAHOSTDEV void plasticity(m3_t &destress, f1_t Y);

/**
 * @brief calculates the yield stress using the moh-coulomb model
 * @param tanFrictionAngle tangent of the inner friction angle, maximal angle of attack for a force without material sliding away / failing
 * @param pressure the current pressure of particle i
 * @param cohesion cohesion of the material, higher values increases "stickyness"
 * @return the yield stress Y to be used for the plasticyty funciton above
 */
 CUDAHOSTDEV f1_t mohrCoulombYieldStress(f1_t tanFrictionAngle, f1_t pressure, f1_t cohesion);

#endif //GRASPH2_MODELS_H
