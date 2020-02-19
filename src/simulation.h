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

template <typename pbT>
doRK4SimulateionStep(pbT& pb,timestep)
{
static pbT pb1(pb.sieze()),pb2,pb3;

computeDerivatives(pb)

rkIntegrateOnce(pb1,pb,pb,0.5*timestep)
computeDerivatives(pb1)

rkIntegrateOnce(pb2,pb,pb1,0.5*timestep)
computeDerivatives(pb2)

rkIntegrateOnce(pb3,pb,pb2,timestep)
computeDerivatives(pb2)

rkCompose(pb,pb1,pb2,pb3,timestep)
}

doFixedLeapfrogStep()
{
    computeDerivatives(pb)
    do_for_each<fixedTsLeapfrog>(particleBuffer,notFirstStep);
}

template <typename pbT>
void simulate(pbT& particleBuffer, bool notFirstStep)
{
    static_assert(mpu::is_instantiation_of< DeviceParticleBuffer,pbT>::value,"Integration is only possible with a device particle buffer");
#if defined(FIXED_TIMESTEP_LEAPFROG)
    doFixedLeapfrogStep();
#elif defined(VARIABLE_TIMESTEP_LEAPFROG)
    // call first part of leapfrog, which will perform kick and calculate the new timestep only if this is not the first step
    computeDerivatives(pb)
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

#elif defined(RK4)
    doRK4SimulateionStep(pb,timestep);
#endif
}

#endif //GRASPH2_SIMULATION_H
