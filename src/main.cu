/*
 * mpUtils
 * main.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail: hendrik.schwanekamp@gmx.net
 *
 * mpUtils = my personal Utillities
 * A utility library for my personal c++ projects
 *
 * Copyright 2016 Hendrik Schwanekamp
 *
 */

#include <thrust/random.h>
#include <mpUtils/mpUtils.h>
#include <mpUtils/mpCuda.h>
#include <cuda_gl_interop.h>
#include <cmath>
#include <initialConditions/InitGenerator.h>
#include <initialConditions/particleSources/UniformSphere.h>
#include <initialConditions/particleSources/TextFile.h>

#include "frontends/frontendInterface.h"
#include "particles/Particles.h"
#include "particles/algorithms.h"
#include "sph/kernel.h"
#include "sph/eos.h"
#include "sph/models.h"
#include "ResultStorageManager.h"
#include "settings.h"

/**
 * @brief calculates the number of cuda blocks to be launched
 * @param particles number of particles to be processed
 * @return the number of cuda blocks that should be launched
 */
constexpr size_t NUM_BLOCKS(size_t particles)
{
    return (particles + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

/**
 * @brief computes deriviatives of particle attributes
 * @param particles device copy of the particle Buffer
 * @param speedOfSound your materials sound speed
 */
__global__ void computeDerivatives(DeviceParticlesType particles, f1_t speedOfSound)
{
    DO_FOR_EACH_PAIR_SM( BLOCK_SIZE, particles, MPU_COMMA_LIST(SHARED_POSM,SHARED_VEL,SHARED_DENSITY,SHARED_DSTRESS),
            MPU_COMMA_LIST(POS,MASS,VEL,ACC,XVEL,DENSITY,DENSITY_DT,DSTRESS,DSTRESS_DT),
            MPU_COMMA_LIST(POS,MASS,VEL,DENSITY,DSTRESS), MPU_COMMA_LIST(ACC,XVEL,DENSITY_DT,DSTRESS_DT),
            MPU_COMMA_LIST(POS,MASS,VEL,DENSITY,DSTRESS),

    int numPartners=0;
    m3_t sigOverRho_i; // stress over density square used for acceleration
    m3_t sigma_i;
    m3_t edot(0); // strain rate tensor (edot)
    m3_t rdot(0); // rotation rate tensor
    f1_t vdiv{0}; // velocity divergence
    {
        sigma_i = pi.dstress;
        f1_t pres_i = eos::murnaghan( pi.density, rho0, BULK, dBULKdP);
        sigma_i[0][0] = (sigma_i[0][0] - pres_i);
        sigma_i[1][1] = (sigma_i[1][1] - pres_i);
        sigma_i[2][2] = (sigma_i[2][2] - pres_i);

        for(size_t e=0;e<9;e++)
            sigOverRho_i(e) = sigma_i(e)/ (pi.density*pi.density);
    }
    ,
    {
        const f3_t rij = pi.pos-pj.pos;
        const f1_t r2 = dot(rij,rij);
        f1_t r = sqrt(r2);
        if(r>0)
        {
            // gravity
//            f3_t r = pi.pos - pj.pos;
            f1_t distSqr = r2 + H*H;
            f1_t invDist = rsqrt(distSqr);
            f1_t invDistCube = invDist * invDist * invDist;
            pi.acc -= rij * pj.mass * invDistCube;

            if(r <= H)
            {
                numPartners++;
                // get the kernel gradient
                const f1_t dw = kernel::dWspline<dimension>(r, H);
                const f3_t gradw = (dw / r) * rij;

                // artificial stress

                // artificial viscosity
                const f3_t vij = pi.vel - pj.vel;
                pi.acc -= pj.mass *
                          artificialViscosity(alpha, pi.density, pj.density, vij, rij, r, speedOfSound, speedOfSound) *
                          gradw;

                // pressure
                m3_t sigma_j = pj.dstress;
                f1_t pres_j = eos::murnaghan(pj.density, rho0, BULK, dBULKdP);
                sigma_j[0][0] = (sigma_j[0][0] - pres_j);
                sigma_j[1][1] = (sigma_j[1][1] - pres_j);
                sigma_j[2][2] = (sigma_j[2][2] - pres_j);
                m3_t sigOverRho_j;
                for(size_t e = 0; e < 9; e++)
                    sigOverRho_j(e) = sigma_j(e) / (pj.density * pj.density);

                // artificial stress
                f1_t f = kernel::Wspline<dimension>(r, H) / kernel::Wspline<dimension>(normalsep, H);
                f = pow(f, matexp);
                m3_t arts;

                for(size_t e = 0; e < 9; e++)
                    arts(e) = ((sigOverRho_i(e) > 0) ? (-mateps * sigOverRho_i(e)) : 0.0f) +
                              ((sigOverRho_j(e) > 0) ? (-mateps * sigOverRho_j(e)) : 0.0f);

                m3_t stress;
                for(size_t e = 0; e < 9; e++)
                    stress(e) = sigOverRho_i(e) + sigOverRho_j(e);

                pi.acc.x += pj.mass * f * ((arts[0][0]) * gradw.x + (arts[0][1]) * gradw.y + (arts[0][2]) * gradw.z);
                pi.acc.y += pj.mass * f * ((arts[1][0]) * gradw.x + (arts[1][1]) * gradw.y + (arts[1][2]) * gradw.z);
                pi.acc.z += pj.mass * f * ((arts[2][0]) * gradw.x + (arts[2][1]) * gradw.y + (arts[2][2]) * gradw.z);

                // acceleration
                pi.acc.x += pj.mass * ((stress[0][0]) * gradw.x + (stress[0][1]) * gradw.y + (stress[0][2]) * gradw.z);
                pi.acc.y += pj.mass * ((stress[1][0]) * gradw.x + (stress[1][1]) * gradw.y + (stress[1][2]) * gradw.z);
                pi.acc.z += pj.mass * ((stress[2][0]) * gradw.x + (stress[2][1]) * gradw.y + (stress[2][2]) * gradw.z);

                // strain rate tensor (edot) and rotation rate tensor (rdot)
                f1_t tmp = -0.5f * pj.mass / pi.density;
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

                // density time derivative
                vdiv += (pj.mass / pj.density) * dot(vij, gradw);

                // xsph
                pi.xvel += 2 * pj.mass / (pi.density + pj.density) * (pj.vel - pi.vel) * kernel::Wspline<dimension>(r, H);
            }
        }
    },
    {
//        printf("%i\n",numPartners);
        // density time derivative
        pi.density_dt = pi.density * vdiv;

        // deviatoric stress time derivative
        for(int d = 0; d < 3; ++d)
            for(int e = 0; e < 3; ++e)
            {
                pi.dstress_dt[d][e] += 2*shear*edot[d][e];
                for(int f=0; f<3;f++)
                {
                    if(d==e)
                    {
                        pi.dstress_dt[d][e] += 2*shear*edot[f][f] / 3.0f;
                    }
                    pi.dstress_dt[d][e] += pi.dstress[d][f]*rdot[e][f];
                    pi.dstress_dt[d][e] += pi.dstress[e][f]*rdot[d][f];
                }

            }
    })
}

/**
 * @brief perform leapfrog integration on the particles also performs the plasticity calculations
 * @param particles the device copy to the particle buffer that stores the particles
 * @param dt the timestep for the integration
 * @param not_first_step set false for the first integration step of the simulation
 */
__global__ void integrateLeapfrog(DeviceParticlesType particles, f1_t dt, bool not_first_step)
{
    DO_FOR_EACH(particles, MPU_COMMA_LIST(POS,VEL,ACC,XVEL,DENSITY,DENSITY_DT,DSTRESS,DSTRESS_DT),
                MPU_COMMA_LIST(POS,VEL,ACC,XVEL,DENSITY,DENSITY_DT,DSTRESS,DSTRESS_DT),
                MPU_COMMA_LIST(POS,VEL,DENSITY,DSTRESS),
        {
        //   calculate velocity a_t
        pi.vel = pi.vel + pi.acc * (dt * 0.5f);

        // we could now change delta t here

        // calculate velocity a_t+1/2
        pi.vel = pi.vel + pi.acc * (dt * 0.5f) * not_first_step;

        // calculate position r_t+1
        pi.pos = pi.pos + pi.vel * dt;

        pi.density = pi.density + pi.density_dt * dt;
        if(pi.density < 0.0f)
            pi.density = 0.0f;

        // deviatoric stress
        pi.dstress += pi.dstress_dt * dt;
        f1_t Y = mohrCoulombYieldStress( tan(friction_angle),eos::murnaghan(pi.density,rho0, BULK, dBULKdP),cohesion);
        plasticity(pi.dstress,Y);
    })
}

int main()
{
    mpu::Log myLog( mpu::LogLvl::ALL, mpu::ConsoleSink());

    myLog.printHeader("GraSPH2");
    logINFO("GraSPH2") << "Welcome to GraSPH2!";
    assert_cuda(cudaSetDevice(0));

    // set up frontend
    fnd::initializeFrontend();
    bool simShouldRun = false;
    fnd::setPauseHandler([&simShouldRun](bool pause){simShouldRun = !pause;});

    // generate some particles depending on options in the settings file
    InitGenerator<HostParticlesType> generator;

#if defined(READ_FROM_FILE)
    ps::TextFile tf(FILENAME,SEPERATOR);
    generator.addParticles(tf);
#elif defined(ROTATING_UNIFORM_SPHERE)
    ps::UniformSphere us(particle_count,1.0,tmass,rho0);
    us.addAngularVelocity(angVel);
    generator.addParticles(us);
#endif

    auto hpb = generator.generate();
    if( hpb.size()==0 || (hpb.size() & (hpb.size() - 1)) )
    {
        logFATAL_ERROR("InitialConditions") << "Particle count of " << hpb.size()
                                            << " is not a power of two. Only power of two particle counts are currently supported!";
        myLog.flush();
        throw std::runtime_error("Particle count not supported!");
    }

    // create cuda buffer
    DeviceParticlesType pb(hpb.size());
#if defined(FRONTEND_OPENGL)
    pb.registerGLGraphicsResource<DEV_POSM>(fnd::getPositionBuffer(pb.size()));
    pb.registerGLGraphicsResource<DEV_VEL>(fnd::getVelocityBuffer(pb.size()));
    pb.mapGraphicsResource();
#endif

    // upload particles
    pb = hpb;

    // calculate sound speed
    const f1_t SOUNDSPEED = sqrt(BULK / rho0);

#ifdef STORE_RESULTS
    // set up file saving engine
    ResultStorageManager storage(RESULT_FOLDER,RESULT_PREFIX);
    storage.printToFile(pb,0);
    f1_t timeSinceStore=timestep;
#endif

    // start simulating
    computeDerivatives<<<NUM_BLOCKS(pb.size()),BLOCK_SIZE>>>(pb.createDeviceCopy(),SOUNDSPEED);
    assert_cuda(cudaGetLastError());
    integrateLeapfrog<<<NUM_BLOCKS(pb.size()),BLOCK_SIZE>>>(pb.createDeviceCopy(),timestep,false);
    assert_cuda(cudaGetLastError());
    double simulatedTime=timestep;
#if defined(READ_FROM_FILE)
    simulatedTime += startTime;
#endif

    pb.unmapGraphicsResource(); // used for frontend stuff
    while(fnd::handleFrontend(simulatedTime))
    {
        if(simShouldRun)
        {
            pb.mapGraphicsResource(); // used for frontend stuff

            computeDerivatives<<<NUM_BLOCKS(pb.size()),BLOCK_SIZE>>>(pb.createDeviceCopy(),SOUNDSPEED);
            assert_cuda(cudaGetLastError());
            integrateLeapfrog<<<NUM_BLOCKS(pb.size()),BLOCK_SIZE>>>(pb.createDeviceCopy(),timestep,true);
            assert_cuda(cudaGetLastError());

            simulatedTime += timestep;

#ifdef STORE_RESULTS
            timeSinceStore += timestep;
            if( timeSinceStore > store_intervall)
            {
                storage.printToFile(pb,simulatedTime);
                timeSinceStore=0;
            }
#endif

            pb.unmapGraphicsResource(); // used for frontend stuff
        }
        else
        {
            mpu::sleep_ms(2);
        }
    }

    return 0;
}
