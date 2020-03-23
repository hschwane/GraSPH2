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

CUDAHOSTDEV inline void handleBoundaryConditions(Particle<rPOS,rVEL> p)
{
#if defined(SIMPLE_BOX_BOUNDARY)

    const f1_t velMultiplyer = 1.0_ft +  simple_box_bound_reflectiveness;
    if(p.pos.x > simple_box_bound_max.x)
    {
        p.pos.x = simple_box_bound_max.x;
        p.vel.x -= velMultiplyer*p.vel.x;
    }
    else if(p.pos.x < simple_box_bound_min.x)
    {
        p.pos.x = simple_box_bound_min.x;
        p.vel.x -= velMultiplyer*p.vel.x;
    }
    if(p.pos.y > simple_box_bound_max.y)
    {
        p.pos.y = simple_box_bound_max.y;
        p.vel.y -= velMultiplyer*p.vel.y;
    }
    else if(p.pos.y < simple_box_bound_min.y)
    {
        p.pos.y = simple_box_bound_min.y;
        p.vel.y -= velMultiplyer*p.vel.y;
    }

    if(dimension == Dim::three)
    {
        if(p.pos.z > simple_box_bound_max.z)
        {
            p.pos.z = simple_box_bound_max.z;
            p.vel.z -= velMultiplyer*p.vel.z;
        }
        else if(p.pos.z < simple_box_bound_min.z)
        {
            p.pos.z = simple_box_bound_min.z;
            p.vel.z -= velMultiplyer*p.vel.z;
        }
    }
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
        f1_t dt = fixed_timestep_lpfrog;

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

        // handle boundary conditions
        handleBoundaryConditions(p);

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

        // handle boundary conditions
        handleBoundaryConditions(p);

        // store the particle
        pb.storeParticle(i, Particle<POS,VEL,DENSITY,DSTRESS>(p));
    }
};

template<size_t blocksize, typename  DevParticleRefType>
__global__ void variableTsRK3_getNewTimestep(pbT pbValues, pbT pbMidpoint, pbT pbRK3, pbT pb_k1)
{
    assert(pbTarget.size() == pbValues.size() && pbMidpoint.size() == pbRK3.size() && pb_k1.size() == pbRK3.size() && "Buffers need to have the same size");
    const f1_t dt = currentTS;
    f1_t mindt = max_timestep; // smallest needed timestep is stored here during reduction
    f1_t err;
    f1_t temp_dt;

    for(const auto i : mpu::gridStrideRange(pbValues.size()))
    {
        // Calculate error
#if defined(ACCELERATION_CRITERION)
        err = (pbMidpoint.acc - pbRK3.acc)/(pbValues + dt * k1); //BETRAG FEHLT
#endif
#if defined(VELOCITY_CRITERION)
        err = (pbMidpoint.vel - pbRK3.vel)/(pbValues + dt * k1); // BETRAG FEHLT
#endif
        //Calculate timestep
        if(err < relative_error)
        {
            temp_dt = dt * (relative_error / err)^0.3; //BETRAG FEHLT
            mindt =  min( mindt, temp_dt);
        }
        else
        {
            temp_dt = 0.9_ft * dt * (relative_error / err)^0.25; //BETRAG FEHLT
            mindt =  min( mindt, temp_dt);
        }
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
    //Not sure if we need this line
    currentTS = mindt;
}

//Function used to integrate a 'normal' runge Kutta timestep; Used for Euler Midpoint Method
template <typename pbT>
__global__ void rkIntegrateOnce(pbT pbTarget, pbT pbValues, pbT pbDerivatives, f1_t dt)
{
    assert(pbTarget.size() == pbValues.size() && pbTarget.size() == pbDerivatives.size() && "Buffers need to have the same size");

    for(const auto i : mpu::gridStrideRange(pbTarget.size()))
    {
        // read the particle
        Particle<POS,VEL,DENSITY,DSTRESS> p{};
        auto pv = pbValues.template loadParticle<POS,VEL,DENSITY,DSTRESS>(i);
        auto pd = pbDerivatives.template loadParticle<VEL,ACC,XVEL,DENSITY_DT,DSTRESS_DT>(i);

        p.vel = pv.vel + pd.acc * dt;

        #if defined(XSPH) && defined(ENABLE_SPH)
                p.pos = pv.pos + (pd.vel + xsph_factor*pd.xvel) * dt;
        #else
                p.pos = pv.pos + pd.vel * dt;
        #endif

        #if defined(ENABLE_SPH)
            #if defined(INTEGRATE_DENSITY)
                p.density = pv.density + pd.density_dt * dt;
                //Check if newly calculated density is below zero
                if(p.density < 0.0_ft)
                    p.density = 0.0_ft;
            #endif
            #if defined(SOLIDS)
                //Stresstensor
                p.dstress = pv.dstress + pd.dstress_dt * dt;
                doPlasticity(p);
            #endif
        #endif

        // store result particle
        pbTarget.storeParticle(i, p);
    }
}

//Since Runge Kutta 3 is special in its calculation of k2 and k3, there are two more functions
template <typename pbT>
__global__ void rk3Integrate_k2(pbT pbTarget, pbT pbValues, pbT pbDerivatives, f1_t dt)
{
    assert(pbTarget.size() == pbValues.size() && pbTarget.size() == pbDerivatives.size() && "Buffers need to have the same size");

    for(const auto i : mpu::gridStrideRange(pbTarget.size()))
    {
        // read the particle
        Particle<POS,VEL,DENSITY,DSTRESS> p{};
        auto pv = pbValues.template loadParticle<POS,VEL,DENSITY,DSTRESS>(i);
        auto k1 = pbDerivatives.template loadParticle<VEL,ACC,XVEL,DENSITY_DT,DSTRESS_DT>(i);

        p.vel = pv.vel + (k1.acc/2.0_ft) * dt;

#if defined(XSPH) && defined(ENABLE_SPH)
        p.pos = pv.pos + ((k1.vel + xsph_factor*pd.xvel)/2.0_ft) * dt;
#else
        p.pos = pv.pos + (k1.vel/2.0_ft) * dt;
#endif

#if defined(ENABLE_SPH)
        #if defined(INTEGRATE_DENSITY)
                p.density = pv.density + k1.density_dt * dt;
                //Check if newly calculated density is below zero
                if(p.density < 0.0_ft)
                    p.density = 0.0_ft;
            #endif
            #if defined(SOLIDS)
                //Stresstensor
                p.dstress = pv.dstress + k1.dstress_dt * dt;
                doPlasticity(p);
            #endif
#endif

        // store result particle
        pbTarget.storeParticle(i, p);
    }
}

template <typename pbT>
__global__ void rk3Integrate_k3(pbT pbTarget, pbT pbValues, pbT pbDerivatives, f1_t dt)
{
    assert(pbTarget.size() == pbValues.size() && pbTarget.size() == pbDerivatives.size() && "Buffers need to have the same size");

    for(const auto i : mpu::gridStrideRange(pbTarget.size()))
    {
        // read the particle
        Particle<POS,VEL,DENSITY,DSTRESS> p{};
        auto k1 = pbValues.template loadParticle<POS,VEL,DENSITY,DSTRESS>(i);
        auto k2 = pbDerivatives.template loadParticle<VEL,ACC,XVEL,DENSITY_DT,DSTRESS_DT>(i);

        p.vel = pv.vel - k1.acc * dt + 2.0_ft * k2.acc * dt;

#if defined(XSPH) && defined(ENABLE_SPH)
        //not sure about k1.vel * dt..
        p.pos = pv.pos - k1.vel * dt + xsph_factor * 2.0_ft * k2.xvel * dt;
#else
        p.pos = pv.pos - k1.vel * dt + 2.0_ft * k2.acc * dt;
#endif

#if defined(ENABLE_SPH)
        #if defined(INTEGRATE_DENSITY)
                p.density = pv.density + k1.density_dt * dt;
                //Check if newly calculated density is below zero
                if(p.density < 0.0_ft)
                    p.density = 0.0_ft;
            #endif
            #if defined(SOLIDS)
                //Stresstensor
                p.dstress = pv.dstress + k1.dstress_dt * dt;
                doPlasticity(p);
            #endif
#endif

        // store result particle
        pbTarget.storeParticle(i, p);
    }
}

template <typename pbT>
__global__ void rk3Compose(pbT pb, const pbT pbDev1, const pbT pbDev2,  f1_t dt)
{
    for(const auto i : mpu::gridStrideRange(pb.size()))
    {
        // read the particle
        auto p = pb.template loadParticle<POS,VEL,ACC,DENSITY,DENSITY_DT,DSTRESS,DSTRESS_DT>(i);
        auto pd1 = pbDev1.template loadParticle<VEL,ACC,XVEL,DENSITY_DT,DSTRESS_DT>(i);
        auto pd2 = pbDev2.template loadParticle<VEL,ACC,XVEL,DENSITY_DT,DSTRESS_DT>(i);

        p.pos = p.pos +  (dt / 6.0_ft) * (p.vel + 4.0_ft*pd1.vel + pd2.vel);
        p.vel = p.vel +  (dt / 6.0_ft) * (p.acc + 4.0_ft*pd1.acc + pd2.acc);

        #if defined(ENABLE_SPH)
            #if defined(INTEGRATE_DENSITY)
                p.density = p.density +  (dt / 6.0_ft) * (p.density_dt + 4.0_ft*pd1.density_dt + pd2.density_dt);
                if(p.density < 0.0_ft)
                    p.density = 0.0_ft;
            #endif
            #if defined(SOLIDS)
                //Stresstensor
                p.dstress = p.dstress +  (dt / 6.0_ft) * (p.dstress_dt + 4.0_ft*pd1.dstress_dt + pd2.dstress_dt);
                doPlasticity(p);
            #endif
        #endif

        // store result particle
        pb.storeParticle(i, Particle<POS,VEL,DENSITY,DSTRESS>(p));
    }
}

template <typename pbT>
__global__ void midpointCompose(pbT pb, const pbT pbDev1, f1_t dt)
{
    for(const auto i : mpu::gridStrideRange(pb.size()))
    {
        // read the particle
        auto p = pb.template loadParticle<POS,VEL,ACC,DENSITY,DENSITY_DT,DSTRESS,DSTRESS_DT>(i);
        auto k = pbDev1.template loadParticle<VEL,ACC,XVEL,DENSITY_DT,DSTRESS_DT>(i);

        p.pos = p.pos +  (dt / 2.0_ft) * k.vel;
        p.vel = p.vel +  (dt / 2.0_ft) * k.acc;

#if defined(ENABLE_SPH)
        #if defined(INTEGRATE_DENSITY)
                p.density = p.density +  (dt / 6.0_ft) * (p.density_dt + 4.0_ft*pd1.density_dt + pd2.density_dt);
                if(p.density < 0.0_ft)
                    p.density = 0.0_ft;
            #endif
            #if defined(SOLIDS)
                //Stresstensor
                p.dstress = p.dstress +  (dt / 6.0_ft) * (p.dstress_dt + 4.0_ft*pd1.dstress_dt + pd2.dstress_dt);
                doPlasticity(p);
            #endif
#endif

        // store result particle
        pb.storeParticle(i, Particle<POS,VEL,DENSITY,DSTRESS>(p));
    }
}



f1_t getCurrentTimestep()
{
#if defined(FIXED_TIMESTEP_LEAPFROG)
    return fixed_timestep_lpfrog;
#elif defined(VARIABLE_TIMESTEP_LEAPFROG)
    static f1_t dt = initial_timestep;
    assert_cuda( cudaMemcpyFromSymbol(&dt,currentTS,sizeof(f1_t)));
    return dt;
#elif defined(RK3fixed)
    return fixed_timestep_rk3;
#elif defined(RK3variable)
    static f1_t dt = initial_timestep;
    assert_cuda( cudaMemcpyFromSymbol(&dt,currentTS,sizeof(f1_t)));
    return dt;
#endif
}

#endif //GRASPH2_INTEGRATION_H
