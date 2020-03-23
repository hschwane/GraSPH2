/*
 * GraSPH2
 * simulation.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2020 Hendrik Schwanekamp
 *
 */
#ifndef GRASPH2_SIMULATION_H
#define GRASPH2_SIMULATION_H

// includes
//--------------------
#include <computeDerivatives.h>
#include <integration.h>
//--------------------
/*
 * Function to call Runge-Kutta integration algorithm for one fixed time step
 */
template <typename pbT>
void doRK3FixedSimulationStep(pbT& pb)
{
    //Create new buffers with same size as input buffer to store derivatives
    static pbT pb1(pb.size()),pb2(pb.size()),pb3(pb.size());

    auto pbref = pb.getDeviceReference();
    auto pb1ref = pb1.getDeviceReference();
    auto pb2ref = pb2.getDeviceReference();

    //First step: compute derivatives
    computeDerivatives(pb);

    //Second Step: Use derivatives from first step to calculate new points pb1 and compute the derivatives...
    rk3Integrate_k2<<< mpu::numBlocks(pb.size() / INTEG_PPT, INTEG_BS), INTEG_BS >>>(pb1ref,pbref,pbref,0.5*fixed_timestep_rk3);
    assert_cuda(cudaGetLastError());
    computeDerivatives(pb1);

    //... to use them as input in third step...
    rk3Integrate_k3<<< mpu::numBlocks(pb.size() / INTEG_PPT, INTEG_BS), INTEG_BS >>>(pb2ref,pbref,pb1ref,fixed_timestep_rk3);
    assert_cuda(cudaGetLastError());
    computeDerivatives(pb2);

    //Finally, we calculate the final values using the derivatives buffer
    rk3Compose<<< mpu::numBlocks(pb.size() / INTEG_PPT, INTEG_BS), INTEG_BS >>>(pbref,pb1ref,pb2ref,fixed_timestep_rk3);
    assert_cuda(cudaGetLastError());
}

/*
 * Function to call Runge-Kutta integration algorithm for one variable time step
 */
template <typename pbT>
void doRK3VariableSimulationStep(pbT& pb)
{
    //Create new buffers with same size as input buffer to store derivatives
    static pbT pb1(pb.size()),pb2(pb.size()),pb3(pb.size());

    auto pbref = pb.getDeviceReference();
    auto pb1ref = pb1.getDeviceReference();
    auto pb2ref = pb2.getDeviceReference();

    computeDerivatives(pb);
    //Use Euler Midpoint Method to calculate error so that we have a value to compare
    rkIntegrateOnce<INTEG_BS> <<< mpu::numBlocks(pb.size() / INTEG_PPT, INTEG_BS),
            INTEG_BS >>> (pb1ref,pbref,pbref,0.5*getCurrentTimestep());
    assert_cuda(cudaGetLastError());
    //Now compose eulers Method...
    midpoint_compose();

    //Then we want to update our timestep as Schäfer et al. 2016, so we calculate RK3 Values
    //Second Step: Use derivatives from first step to calculate new points pb1 and compute the derivatives...
    rk3Integrate_k2<<< mpu::numBlocks(pb.size() / INTEG_PPT, INTEG_BS), INTEG_BS >>>(pb1ref,pbref,pbref,0.5*getCurrentTimestep());
    assert_cuda(cudaGetLastError());
    computeDerivatives(pb1);

    //... to use them as input in third step...
    rk3Integrate_k3<<< mpu::numBlocks(pb.size() / INTEG_PPT, INTEG_BS), INTEG_BS >>>(pb2ref,pbref,pb1ref,getCurrentTimestep());
    assert_cuda(cudaGetLastError());
    computeDerivatives(pb2);

    //Finally, we calculate the final values using the derivatives buffer
    rk3Compose<<< mpu::numBlocks(pb.size() / INTEG_PPT, INTEG_BS), INTEG_BS >>>(pbref,pb1ref,pb2ref,getCurrentTimestep());
    assert_cuda(cudaGetLastError());
    //with the calculated values of both midpoint and rk3 we calculate an error and calculate the new timestep
    variableTsRK3_getNewTimestep<INTEG_BS> <<< mpu::numBlocks(pb.size() / INTEG_PPT, INTEG_BS),
            INTEG_BS >>> (pb.getDeviceReference());
    assert_cuda(cudaGetLastError());

    // second part which will use the new timestep
    computeDerivatives(pb);
    //Then we want to update our timestep as Schäfer et al. 2016, so we calculate RK3 Values
    //Second Step: Use derivatives from first step to calculate new points pb1 and compute the derivatives...
    rk3Integrate_k2<<< mpu::numBlocks(pb.size() / INTEG_PPT, INTEG_BS), INTEG_BS >>>(pb1ref,pbref,pbref,0.5*getCurrentTimestep());
    assert_cuda(cudaGetLastError());
    computeDerivatives(pb1);

    //... to use them as input in third step...
    rk3Integrate_k3<<< mpu::numBlocks(pb.size() / INTEG_PPT, INTEG_BS), INTEG_BS >>>(pb2ref,pbref,pb1ref,getCurrentTimestep());
    assert_cuda(cudaGetLastError());
    computeDerivatives(pb2);
}

/**
 * Function to call Leapfrog integration algorithm for one time step
 */
template <typename pbT>
void doFixedLeapfrogStep(pbT& pb, bool nFirstStep)
{
    computeDerivatives(pb);
    do_for_each<fixedTsLeapfrog>(pb,nFirstStep);
}

/**
 * Function to call variable leapfrog integration algorithm for one time step
 */
template <typename pbT>
void doVariableLeapfrogStep(pbT& pb, bool nFirstStep)
{
    computeDerivatives(pb);
    if(nFirstStep)
    {
        variableTsLeapfrog_getNewTimestep<INTEG_BS> <<< mpu::numBlocks(pb.size() / INTEG_PPT, INTEG_BS),
                INTEG_BS >>> (pb.getDeviceReference());
        assert_cuda(cudaGetLastError());
    }
    else
    {
        f1_t its = initial_timestep;
        assert_cuda( cudaMemcpyToSymbol(nextTS, &its, sizeof(f1_t))); // reset dt on first step
    }

    // second part which will use the new timestep to perform kick an drift
    variableTsLeapfrog_useNewTimestep<<< mpu::numBlocks(pb.size()/INTEG_PPT,INTEG_BS),INTEG_BS >>>( pb.getDeviceReference());
    assert_cuda(cudaGetLastError());
}

template <typename pbT>
void simulate(pbT& particleBuffer, bool notFirstStep)
{
    static_assert(mpu::is_instantiation_of< DeviceParticleBuffer,pbT>::value,"Integration is only possible with a device particle buffer");
#if defined(FIXED_TIMESTEP_LEAPFROG)
    doFixedLeapfrogStep(particleBuffer, notFirstStep);
#elif defined(VARIABLE_TIMESTEP_LEAPFROG)
    doVariableLeapfrogStep(particleBuffer, notFirstStep);
#elif defined(RK3fixed)
    doRK3FixedSimulationStep(particleBuffer);
#endif
}

#endif //GRASPH2_SIMULATION_H
