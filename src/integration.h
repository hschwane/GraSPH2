/*
 * GraSPH2
 * integration.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2019 Hendrik Schwanekamp
 *
 */
#ifndef GRASPH2_INTEGRATION_H
#define GRASPH2_INTEGRATION_H

// includes
//--------------------
#include "settings.h"
#include "sph/models.h"
#include "sph/eos.h"
#include "particles/Particles.h"
#include <mpUtils/mpUtils.h>
#include <mpUtils/mpCuda.h>
//--------------------

CUDAHOSTDEV inline void doPlasticity(Particle<DENSITY,rDSTRESS> p)
{
#if defined(PLASTICITY_MC)
    plasticity(p.dstress, mohrCoulombYieldStress( tanfr,eos::murnaghan(p.density,rho0, BULK, dBULKdP),cohesion));
#elif defined(PLASTICITY_MIESE)
    plasticity(p.dstress,Y);
#endif
}

/**
 * @brief perform fixed timestep, kick drift kick leapfrog integration on the particles also performs the plasticity calculations
 *          density and deviatoric stress are updated during drift step
 * @param particles the device copy to the particle buffer that stores the particles
 * @param dt the timestep for the integration
 * @param not_first_step set false for the first integration step of the simulation
 */
struct fixedTsLeapfrog
{
    using load_type = Particle<POS,VEL,ACC,XVEL,DENSITY,DENSITY_DT,DSTRESS,DSTRESS_DT>; //!< particle attributes to load from main memory
    using store_type = Particle<POS,VEL,DENSITY,DSTRESS>; //!< particle attributes to store to main memory
    using pi_type = merge_particles_t<load_type,store_type>; //!< the type of particle you want to work with in your job functions

    //!< This function is executed for each particle. In p the current particle and in id its position in the buffer is given.
    //!< All attributes of p that are not in load_type will be initialized to some default (mostly zero)
    CUDAHOSTDEV store_type do_for_each(pi_type p, size_t id, bool not_first_step)
    {
        f1_t dt = timestep;

        //   calculate velocity a_t+1/2
        p.vel = p.vel + p.acc * (dt * ((not_first_step) ? 1.0_ft : 0.5_ft) );

        // calculate position r_t+1
#if defined(XSPH) && defined(ENABLE_SPH)
        p.pos = p.pos + (p.vel + xsph_factor*p.xvel) * dt;
#else
        p.pos = p.pos + p.vel * dt;
#endif

#if defined(ENABLE_SPH)
        // density
        p.density = p.density + p.density_dt * dt;
        if(p.density < 0.0_ft)
            p.density = 0.0_ft;

        // deviatoric stress
        p.dstress += p.dstress_dt * dt;

        // execute selected plasticity model
        doPlasticity(p);
#endif
        return p; //!< return particle p, all attributes it shares with load_type will be stored in memory
    }
};

__device__ int currentTS = initial_timestep;

//struct variableTsLeapfrog_getNewTimestep
//{
//    using load_type = Particle<VEL,ACC>; //!< particle attributes to load from main memory
//    using store_type = Particle<VEL>; //!< particle attributes to store to main memory
//    using pi_type = merge_particles_t<load_type,store_type>; //!< the type of particle you want to work with in your job functions
//
//    //!< This function is executed for each particle. In p the current particle and in id its position in the buffer is given.
//    //!< All attributes of p that are not in load_type will be initialized to some default (mostly zero)
//    CUDAHOSTDEV store_type do_for_each(pi_type p, size_t id, bool not_first_step)
//    {
//        int dt = currentTS;
//
//        //   calculate velocity a_t
//        p.vel = p.vel + p.acc * (dt * 0.5_ft);
//
//        // figure out new timestep
//
//
//
//        return p; //!< return particle p, all attributes it shares with load_type will be stored in memory
//    }
//};

//__global__ variableTsLeapfrog_getNewTimestep();
//{
//    for(const auto i : mpu::gridStrideRange(pb.size()))
//    {
//        typename job::pi_type pi{};
//        pi = load_helper<typename job::load_type, deviceReference>::load(pb, i);
//
//        typename job::store_type result = job_i.do_for_each(pi, i, args...);
//        pb.storeParticle(i, result);
//    }
//};
//
//struct variableTsLeapfrog_getNewTimestep
//{
//    using load_type = Particle<POS,VEL,ACC,XVEL,DENSITY,DENSITY_DT,DSTRESS,DSTRESS_DT>; //!< particle attributes to load from main memory
//    using store_type = Particle<POS,VEL,DENSITY,DSTRESS>; //!< particle attributes to store to main memory
//    using pi_type = merge_particles_t<load_type,store_type>; //!< the type of particle you want to work with in your job functions
//
//    //!< This function is executed for each particle. In p the current particle and in id its position in the buffer is given.
//    //!< All attributes of p that are not in load_type will be initialized to some default (mostly zero)
//    CUDAHOSTDEV store_type do_for_each(pi_type p, size_t id, bool not_first_step)
//    {
//        int dt = currentTS;
//
//        //   calculate velocity a_t
//        p.vel = p.vel + p.acc * (dt * 0.5_ft);
//
//        // figure out new timestep
//
//
//
//        // second launch using the result
//
//        // calculate velocity a_t+1/2
//        p.vel = p.vel + p.acc * (dt * 0.5_ft) * not_first_step;
//
//        // calculate position r_t+1
//#if defined(XSPH) && defined(ENABLE_SPH)
//        p.pos = p.pos + (p.vel + xsph_factor*p.xvel) * dt;
//#else
//        p.pos = p.pos + p.vel * dt;
//#endif
//
//#if defined(ENABLE_SPH)
//        // density
//        p.density = p.density + p.density_dt * dt;
//        if(p.density < 0.0_ft)
//            p.density = 0.0_ft;
//
//        // deviatoric stress
//        p.dstress += p.dstress_dt * dt;
//
//        // execute selected plasticity model
//        doPlasticity(p);
//#endif
//        return p; //!< return particle p, all attributes it shares with load_type will be stored in memory
//    }
//};

template <typename pbT>
void integrate(pbT& particleBuffer, bool notFirstStep)
{
#if defined(FIXED_TIMESTEP_LEAPFROG)
    do_for_each<fixedTsLeapfrog>(particleBuffer,notFirstStep);
#elif defined(VARIABLE_TIMESTEP_LEAPFROG)



#endif
}


#endif //GRASPH2_INTEGRATION_H
