/*
 * GraSPH2
 * computeDerivatives.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2019 Hendrik Schwanekamp
 *
 */
#ifndef GRASPH2_COMPUTEDERIVATIVES_H
#define GRASPH2_COMPUTEDERIVATIVES_H

// includes
//--------------------
#include "settings.h"
#include "sph/models.h"
#include "sph/eos.h"
#include "sph/kernel.h"
#include "particles/Particles.h"
#include <mpUtils/mpUtils.h>
#include <mpUtils/mpCuda.h>
#include "integration.h" // jep i know that is bad design, should be changed later
//--------------------

constexpr f1_t H2 = H*H; //!< square of the smoothing length
constexpr f1_t dW_prefactor = kernel::dsplinePrefactor<dimension>(H); //!< spline kernel prefactor
constexpr f1_t W_prefactor = kernel::splinePrefactor<dimension>(H); //!< spline kernel prefactor

/**
 * @brief add acceleration based on environment
 */
CUDAHOSTDEV inline f3_t environmentAcceleration(const f3_t& position, const f1_t& mass, const f3_t& velocity, f3_t& acceleration)
{
    // acceleration due to tidal forces (clohessy wiltshire model)
#if defined(CLOHESSY_WILTSHIRE)
    acceleration += 3*cw_n*cw_n * position.x + 2*cw_n* velocity.y;
    acceleration += -2*cw_n * velocity.x;
    acceleration += -cw_n*cw_n * position.z;
#endif

#if defined(CONSTANT_ACCELERATION)
    acceleration += constant_acceleration;
#endif

    return acceleration;
}

/**
 * @brief calculates density using the sph method
 */
struct calcDensity
{
    // define particle attributes to use
    using load_type = Particle<POS,MASS>; //!< particle attributes to load from main memory
    using store_type = Particle<DENSITY>; //!< particle attributes to store to main memory
    using pi_type = merge_particles_t<load_type,store_type>; //!< the type of particle you want to work with in your job functions
    using pj_type = Particle<POS,MASS>; //!< the particle attributes to load from main memory of all the interaction partners j
    //!< when using do_for_each_pair_fast a SharedParticles object must be specified which can store all the attributes of particle j
    template<size_t n> using shared = SharedParticles<n,SHARED_POSM>;

    //!< This function is executed for each particle before the interactions are computed.
    CUDAHOSTDEV void do_before(pi_type& pi, size_t id)
    {
    }

    //!< This function will be called for each pair of particles.
    CUDAHOSTDEV void do_for_each_pair(pi_type& pi, const pj_type pj)
    {
        const f3_t rij = pi.pos-pj.pos;
        const f1_t r2 = dot(rij,rij);
        if(r2 <= H2 && r2>0)
        {
            // get the kernel function
            const f1_t r = sqrt(r2);
            const f1_t w = kernel::Wspline(r, H, W_prefactor);

            pi.density += pj.mass * w;
        }
    }

    //!< This function will be called for particle i after the interactions with the other particles are computed.
    CUDAHOSTDEV store_type do_after(pi_type& pi)
    {
#if defined(DEAL_WITH_NO_PARTNERS)
        if(pi.density < 0.001)
            pi.density = rho0*0.01;
#endif
        return pi;
    }
};

/**
 * @brief calculates deriviatives of density, deviatoric stress as well as the balsara switch
 */
struct calcBalsaraDensityDTDStressDT
{
    // define particle attributes to use
    using load_type = Particle<POS,MASS,VEL,DENSITY,DSTRESS>; //!< particle attributes to load from main memory
    using store_type = Particle<BALSARA,DENSITY_DT,DSTRESS_DT>; //!< particle attributes to store to main memory
    using pi_type = merge_particles_t<load_type,store_type>; //!< the type of particle you want to work with in your job functions
    using pj_type = Particle<POS,MASS,VEL,DENSITY>; //!< the particle attributes to load from main memory of all the interaction partners j
    //!< when using do_for_each_pair_fast a SharedParticles object must be specified which can store all the attributes of particle j
    template<size_t n> using shared = SharedParticles<n,SHARED_POSM,SHARED_VEL,SHARED_DENSITY>;

    // setup some variables we need before during and after the pair interactions
    m3_t edot{0}; // strain rate tensor (edot)
    m3_t rdot{0}; // rotation rate tensor
    f1_t divv{0}; // velocity divergence
    f3_t curlv{0}; // velocity curl

    //!< This function is executed for each particle before the interactions are computed.
    CUDAHOSTDEV void do_before(pi_type& pi, size_t id)
    {
    }

    //!< This function will be called for each pair of particles.
    CUDAHOSTDEV void do_for_each_pair(pi_type& pi, const pj_type pj)
    {
        const f3_t rij = pi.pos-pj.pos;
        const f1_t r2 = dot(rij,rij);
        if(r2 <= H2 && r2>0)
        {
            // get the kernel gradient
            const f1_t r = sqrt(r2);
            const f1_t dw = kernel::dWspline(r, H, dW_prefactor);
            const f3_t gradw = (dw / r) * rij;

            const f3_t vij = pi.vel - pj.vel;
#if defined(SOLIDS)
            // strain rate tensor (edot) and rotation rate tensor (rdot)
            addStrainRateAndRotationRate(edot,rdot,pj.mass,pj.density,vij,gradw);
#elif defined(BALSARA_SWITCH)
            curlv += pj.mass / pj.density * cross(vij, gradw);
#endif
#if defined(BALSARA_SWITCH) || defined(INTEGRATE_DENSITY)
            divv -= (pj.mass / pj.density) * dot(vij, gradw);
#endif

        }
    }

    //!< This function will be called for particle i after the interactions with the other particles are computed.
    CUDAHOSTDEV store_type do_after(pi_type& pi)
    {
#if defined(SOLIDS)
        // deviatoric stress time derivative
        pi.dstress_dt = dstress_dt(edot,rdot,pi.dstress,shear);
        // get curl from edot and compute the balsara switch value
        curlv = f3_t{-2*rdot[1][2], -2*rdot[2][0], -2*rdot[0][1] };
#endif
#if defined(INTEGRATE_DENSITY)
        // density time derivative
        pi.density_dt = -pi.density * divv;
#endif
#if defined(BALSARA_SWITCH)
        pi.balsara = balsaraSwitch(divv, curlv, SOUNDSPEED, H);
#endif

        return pi;
    }
};

/**
 * @brief second pass of derivative computation
 */
struct calcAcceleration
{
    // define particle attributes to use
    using load_type = Particle<POS,MASS,VEL,BALSARA,DENSITY,DSTRESS>; //!< particle attributes to load from main memory
    using store_type = Particle<ACC,XVEL,MAXVSIG>; //!< particle attributes to store to main memory
    using pi_type = merge_particles_t<load_type,store_type>; //!< the type of particle you want to work with in your job functions
    using pj_type = Particle<POS,MASS,VEL,BALSARA,DENSITY,DSTRESS>; //!< the particle attributes to load from main memory of all the interaction partners j
    //!< when using do_for_each_pair_fast a SharedParticles object must be specified which can store all the attributes of particle j
    template<size_t n> using shared = SharedParticles<n,SHARED_POSM,SHARED_VEL,SHARED_BALSARA,SHARED_DENSITY,SHARED_DSTRESS>;

    // setup some variables we need before during and after the pair interactions
#if defined(ENABLE_SPH)

    #if defined(SOLIDS)
        using stress_t = m3_t;
    #else
        using stress_t = f1_t;
    #endif

    stress_t sigOverRho_i; // stress over density square used for acceleration
    #if defined(ARTIFICIAL_STRESS)
        stress_t arts_i; // artificial stress from i
    #endif
#endif

    //!< This function is executed for each particle before the interactions are computed.
    CUDAHOSTDEV void do_before(pi_type& pi, size_t id)
    {
#if defined(ENABLE_SPH)
        // build stress tensor for particle i using deviatoric stress and pressure
        stress_t sigma_i;
    #if defined(SOLIDS)
        sigma_i = pi.dstress;
        const f1_t pres_i = eos::murnaghan( pi.density, rho0, BULK, dBULKdP);
        sigma_i[0][0] -= pres_i;
        sigma_i[1][1] -= pres_i;
        sigma_i[2][2] -= pres_i;
    #else
        sigma_i =  -eos::liquid( pi.density, rho0, SOUNDSPEED*SOUNDSPEED);
    #endif

        sigOverRho_i = sigma_i / (pi.density*pi.density);

    #if defined(ARTIFICIAL_STRESS)
        // artificial stress from i
        #if defined(SOLIDS)
            arts_i = artificialStress(mateps, sigOverRho_i);
        #else
            arts_i = artificialPressure(mateps, sigOverRho_i);
        #endif
    #endif
#endif
    }

    //!< This function will be called for each pair of particles.
    CUDAHOSTDEV void do_for_each_pair(pi_type& pi, const pj_type pj)
    {
        const f3_t rij = pi.pos-pj.pos;
        const f1_t r2 = dot(rij,rij);
        if(r2>0)
        {

#if defined(ENABLE_SELF_GRAVITY)
            // gravity
            const f1_t distSqr = r2 + H2;
            const f1_t invDist = rsqrt(distSqr);
            const f1_t invDistCube = invDist * invDist * invDist;
            pi.acc -= rij * pj.mass * invDistCube;
#endif

#if defined(ENABLE_SPH)
            if(r2 <= H2)
            {
                // get the kernel gradient
                f1_t r = sqrt(r2);
                const f1_t dw = kernel::dWspline(r, H, dW_prefactor);
                const f3_t gradw = (dw / r) * rij;

                // stress and pressure of j
                stress_t sigma_j;
    #if defined(SOLIDS)
                sigma_j = pj.dstress;
                const f1_t pres_j = eos::murnaghan(pj.density, rho0, BULK, dBULKdP);
                sigma_j[0][0] -= pres_j;
                sigma_j[1][1] -= pres_j;
                sigma_j[2][2] -= pres_j;
    #else
                sigma_j = -eos::liquid( pj.density, rho0, SOUNDSPEED*SOUNDSPEED);
    #endif

                stress_t sigOverRho_j = sigma_j / (pj.density * pj.density);

                // stress from the interaction
                stress_t stress = sigOverRho_i + sigOverRho_j;

                const f3_t vij = pi.vel - pj.vel;
    #if defined(ARTIFICIAL_STRESS)
                // artificial stress
                const f1_t f = pow(kernel::Wspline(r, H, W_prefactor) / kernel::Wspline(normalsep, H, W_prefactor) , matexp);
        #if defined(SOLIDS)
                stress_t arts_j = artificialStress(mateps, sigOverRho_j)
        #else
                stress_t arts_j = artificialPressure(mateps, sigOverRho_j);
        #endif
                stress += f*(arts_i + arts_j);
    #endif

                // acceleration from stress
                pi.acc += pj.mass * (stress * gradw);

    #if defined(ARTIFICIAL_VISCOSITY)
                // acceleration from artificial viscosity
                pi.acc -= pj.mass *
                          artificialViscosity(
        #if defined(BALSARA_SWITCH)
                                  pi.max_vsig,
        #endif
                                  alpha, pi.density, pj.density, vij, rij, r, SOUNDSPEED, SOUNDSPEED
        #if defined(BALSARA_SWITCH)
                                  , pi.balsara, pj.balsara
        #endif
                          ) * gradw;
    #endif

    #if defined(XSPH)
                // xsph
                pi.xvel += 2 * pj.mass / (pi.density + pj.density) * (pj.vel - pi.vel) * kernel::Wspline<dimension>(r, H);
    #endif
            }
#endif // ENABLE_SPH
        }
    }

    //!< This function will be called for particle i after the interactions with the other particles are computed.
    __device__ store_type do_after(pi_type& pi)
    {
        // acceleration based on environmental settings
        environmentAcceleration(pi.pos, pi.mass, pi.vel, pi.acc);

#if defined(VARIABLE_TIMESTEP_LEAPFROG)
        if(threadIdx.x == 0 && blockIdx.x == 0)
            nextTS = max_timestep;
#endif

        return pi;
    }
};

template <typename pbT>
void computeDerivatives(pbT& particleBuffer)
{
#if defined(ENABLE_SPH)
    #if !defined(INTEGRATE_DENSITY)
        do_for_each_pair_fast<calcDensity>(particleBuffer);
    #endif
    #if defined(SOLIDS) || defined(INTEGRATE_DENSITY) || defined(BALSARA_SWITCH)
        do_for_each_pair_fast<calcBalsaraDensityDTDStressDT>(particleBuffer);
    #endif
#endif

    do_for_each_pair_fast<calcAcceleration>(particleBuffer);
}


#endif //GRASPH2_COMPUTEDERIVATIVES_H