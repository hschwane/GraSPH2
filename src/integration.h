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
        f1_t dt = fixed_timestep;

        //   calculate velocity a_t+1/2
        p.vel = p.vel + p.acc * (dt * ((not_first_step) ? 1.0_ft : 0.5_ft) );

        // calculate position r_t+1
#if defined(XSPH) && defined(ENABLE_SPH)
        p.pos = p.pos + (p.vel + xsph_factor*p.xvel) * dt;
#else
        p.pos = p.pos + p.vel * dt;
#endif

#if defined(ENABLE_SPH)
    #if defined(INTEGRATE_DENSITY)
        // density
        p.density = p.density + p.density_dt * dt;
        if(p.density < 0.0_ft)
            p.density = 0.0_ft;
    #endif
    #if defined(SOLIDS)
        // deviatoric stress
        p.dstress += p.dstress_dt * dt;

        // execute selected plasticity model
        doPlasticity(p);
    #endif
#endif
        return p; //!< return particle p, all attributes it shares with load_type will be stored in memory
    }
};

__device__ f1_t currentTS;
__device__ f1_t nextTS;

template<size_t blocksize, typename  DevParticleRefType>
__global__ void variableTsLeapfrog_getNewTimestep(DevParticleRefType pb)
{
    const f1_t dt = currentTS;
    f1_t mindt = max_timestep; // smallest needed timestep is stored here during reduction

    for(const auto i : mpu::gridStrideRange(pb.size()))
    {
        // read the particle
        Particle<VEL,ACC,MAXVSIG> p{};
        p = pb.template loadParticle<VEL,ACC,MAXVSIG>(i);

        //   calculate velocity a_t and store
        p.vel = p.vel + p.acc * (dt * 0.5_ft);
        pb.storeParticle(i, Particle<VEL>(p));

        // calculate new timestep
#if defined(ACCELERATION_CRITERION)
        mindt = min( mindt, sqrt(accel_accuracy*H / length(p.acc)) );
#endif
#if defined(VELOCITY_CRITERION)
        mindt = min( mindt, velocity_accuracy * H / p.max_vsig);
#endif

    }

    // setup a cub::warpreduce and reduce each warp
    typedef cub::WarpReduce<f1_t> WarpReduce;
    __shared__ typename WarpReduce::TempStorage temp_storage[blocksize/32];
    int warp_id = threadIdx.x / 32;
    mindt = WarpReduce(temp_storage[warp_id]).Reduce(mindt,cub::Min());

    // use atomic min to reduce the rest, positive floats compare like integers
    if ( threadIdx.x % warpSize == 0)
#if defined(SINGLE_PRECISION)
        atomicMin( reinterpret_cast<int*>(&nextTS), *reinterpret_cast<int*>(&mindt));
#elif defined(DOUBLE_PRECISION)
        atomicMin( reinterpret_cast<long long int*>(&nextTS), *reinterpret_cast<long long int*>(&mindt));
#endif
};

template<typename  DevParticleRefType>
__global__ void variableTsLeapfrog_useNewTimestep(DevParticleRefType pb)
{
    // read the new timestep and set it as current
    f1_t dt = (nextTS < min_timestep) ? min_timestep : nextTS;
    if(threadIdx.x == 0 && blockIdx.x == 0)
        currentTS = dt;

    for(const auto i : mpu::gridStrideRange(pb.size()))
    {
        // read the particle
        Particle<POS,VEL,ACC,XVEL,DENSITY,DENSITY_DT,DSTRESS,DSTRESS_DT> p{};
        p = pb.template loadParticle<POS,VEL,ACC,XVEL,DENSITY,DENSITY_DT,DSTRESS,DSTRESS_DT>(i);

        // calculate velocity a_t+1/2
        p.vel = p.vel + p.acc * (dt * 0.5_ft);

        // calculate position r_t+1
#if defined(XSPH) && defined(ENABLE_SPH)
        p.pos = p.pos + (p.vel + xsph_factor*p.xvel) * dt;
#else
        p.pos = p.pos + p.vel * dt;
#endif

#if defined(ENABLE_SPH)
    #if defined(INTEGRATE_DENSITY)
        // density
        p.density = p.density + p.density_dt * dt;
        if(p.density < 0.0_ft)
            p.density = 0.0_ft;
    #endif
    #if defined(SOLIDS)
        // deviatoric stress
        p.dstress += p.dstress_dt * dt;

        // execute selected plasticity model
        doPlasticity(p);
    #endif
#endif

        // store the particle
        pb.storeParticle(i, Particle<POS,VEL,DENSITY,DSTRESS>(p));
    }
};

template <typename pbT>
void integrate(pbT& particleBuffer, bool notFirstStep)
{
    static_assert(mpu::is_instantiation_of< DeviceParticleBuffer,pbT>::value,"Integration is only possible with a device particle buffer");
#if defined(FIXED_TIMESTEP_LEAPFROG)
    do_for_each<fixedTsLeapfrog>(particleBuffer,notFirstStep);
#elif defined(VARIABLE_TIMESTEP_LEAPFROG)
    // call first part of leapfrog, which will perform kick and calculate the new timestep only if this is not the first step
    if(notFirstStep)
    {
        variableTsLeapfrog_getNewTimestep<INTEG_BS> <<< mpu::numBlocks(particleBuffer.size() / INTEG_PPT, INTEG_BS),
                INTEG_BS >>> (particleBuffer.getDeviceReference());
        assert_cuda(cudaGetLastError());
    }
    else
    {
        f1_t its = initial_timestep;
        assert_cuda( cudaMemcpyToSymbol(nextTS, &its, sizeof(f1_t))); // reset dt on first step
    }

    // second part which will use the new timestep to perform kick an drift
    variableTsLeapfrog_useNewTimestep<<< mpu::numBlocks(particleBuffer.size()/INTEG_PPT,INTEG_BS),INTEG_BS >>>( particleBuffer.getDeviceReference());
    assert_cuda(cudaGetLastError());

#endif
}

f1_t getCurrentTimestep()
{
#if defined(FIXED_TIMESTEP_LEAPFROG)
    return fixed_timestep;
#elif defined(VARIABLE_TIMESTEP_LEAPFROG)
    static f1_t dt = initial_timestep;
    assert_cuda( cudaMemcpyFromSymbol(&dt,currentTS,sizeof(f1_t)));
    return dt;
#endif
}

#endif //GRASPH2_INTEGRATION_H
