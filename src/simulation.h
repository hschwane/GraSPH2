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
 * Function to call Runge-Kutta integration algorithm for one time step
 */
template <typename pbT>
void doRK4SimulationStep(pbT& pb,f1_t timestep)
{
    //Create new buffers with same size as input buffer to store derivatives
    static pbT pb1(pb.size()),pb2(pb.size()),pb3(pb.size());

    //First step: compute derivatives
    computeDerivatives(pb)

    //Second Step: Use derivatives from first step to calculate new points pb1 and compute the derivatives...
    rkIntegrateOnce(pb1,pb,pb,0.5*timestep)
    computeDerivatives(pb1)

    //... to use them as input in third step...
    rkIntegrateOnce(pb2,pb,pb1,0.5*timestep)
    computeDerivatives(pb2)

    //... and do the same for the last step
    rkIntegrateOnce(pb3,pb,pb2,timestep)
    computeDerivatives(pb2)

    //Finally, we calculate the final values using the derivatives buffer
    rkCompose(pb,pb1,pb2,pb3,timestep)
}

/*
 * Function to call Leapfrog integration algorithm for one time step
 */
template <typename pbT>
void doFixedLeapfrogStep(pbT& pb, bool nFirstStep)
{
    computeDerivatives(pb)
    do_for_each<fixedTsLeapfrog>(pb,nFirstStep);
}

/*
 * Function to call variable leapfrog integration algorithm for one time step
 */
template <typename pbT>
void doVariableLeapfrogStep(pbT& pb, bool nFirstStep)
{
    computeDerivatives(pb);
    if(doVariableLeapfrogStep(bool nFirstStep))
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
#elif defined(RK4)
    doRK4SimulateionStep(particleBuffer,timestep);
#endif
}

#endif //GRASPH2_SIMULATION_H
