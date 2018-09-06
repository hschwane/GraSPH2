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
#include <mpUtils.h>
#include <mpCuda.h>
#include <cuda_gl_interop.h>

#include "frontends/frontendInterface.h"
#include "particles/Particles.h"
#include "particles/algorithms.h"
#include "sph/kernel.h"
#include "sph/eos.h"

constexpr int BLOCK_SIZE = 256;
constexpr int PARTICLES = 1<<13;
constexpr f1_t H = 0.01;

constexpr f1_t alpha = 1;
constexpr f1_t rho0 = 2;
constexpr f1_t BULK = 8;
constexpr f1_t dBULKdP = 8;
constexpr f1_t shear = 0.22;
const f1_t SOUNDSPEED = sqrt(BULK / rho0);

int NUM_BLOCKS = (PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE;

using DeviceParticlesType = Particles<DEV_POSM,DEV_VEL,DEV_ACC,DEV_DENSITY,DEV_DENSITY_DT,DEV_DSTRESS,DEV_DSTRESS_DT>;

__global__ void generate2DRings(DeviceParticlesType particles)
{
    const float R = 0.5;
    const float r = 0.4;
    const float a = M_PI * (R*R-r*r);
    const float ringMass = rho0 * a;
    const int ringSize = particles.size()/2;
    const float particleMass = ringMass/ringSize;

    const float xOffA = -0.6;
    const float xOffB = 0.6;
    const float speed = 1;


    INIT_EACH(particles, MPU_COMMA_LIST(POS,MASS,VEL,DENSITY),
    {
        thrust::random::default_random_engine rng;
        rng.discard(i);
        thrust::random::uniform_real_distribution<float> dist(-1.0f,1.0f);

        while(length(pi.pos) > R || length(pi.pos) < r)
        {
            pi.pos.x = dist(rng);
            rng.discard(particles.size());
            pi.pos.y = dist(rng);
            rng.discard(particles.size());
        }

        if(i < ringSize)
        {
            pi.pos.x += xOffA;
            pi.vel.x = speed;
        }
        else
        {
            pi.pos.x += xOffB;
            pi.vel.x = -speed;
        }

        pi.mass = particleMass;
        pi.density = rho0;
    });
}

__global__ void computeDerivatives(DeviceParticlesType particles, f1_t speedOfSound)
{
    DO_FOR_EACH_PAIR_SM( BLOCK_SIZE, particles, MPU_COMMA_LIST(SHARED_POSM,SHARED_VEL,SHARED_DENSITY,SHARED_DSTRESS),
            MPU_COMMA_LIST(POS,MASS,VEL,ACC,DENSITY,DENSITY_DT,DSTRESS,DSTRESS_DT),
            MPU_COMMA_LIST(POS,MASS,VEL,DENSITY,DSTRESS), MPU_COMMA_LIST(ACC,DENSITY_DT,DSTRESS_DT),
            MPU_COMMA_LIST(POS,MASS,VEL,DENSITY,DSTRESS),

    int numPartners=0;
    f1_t sigOverRho_i; // stress over density square used for acceleration
    m3_t edot(0); // strain rate tensor (edot)
    m3_t rdot(0); // rotation rate tensor
    f1_t vdiv{0}; // velocity divergence
    {
//        m3_t stress_i = m3_t(-eos::murnaghan( pi.density, rho0, BULK, dBULKdP)) + pi.dstress;
//        sigOverRho_i = stress_i / (pi.density*pi.density);
        sigOverRho_i = -eos::murnaghan( pi.density, rho0, BULK, dBULKdP) / (pi.density*pi.density);
    }
    ,
    {
        const f3_t rij = pi.pos-pj.pos;
        const f1_t r2 = dot(rij,rij);
        f1_t r = sqrt(r2);
        if(r>0 && r <= H)
        {
            numPartners++;
            // get the kernel gradient
            const f1_t dw = kernel::dWspline<Dim::two>(r,H);
            const f3_t gradw = (dw/r) * rij;

            // artificial viscosity
            const f3_t vij = pi.vel-pj.vel;
            const f1_t wij = dot(rij, vij) /r;
            f1_t II = 0;
            if(wij < 0)
            {
                const f1_t vsig = f1_t(2.0*speedOfSound - 3.0*wij);
                const f1_t rhoij = (pi.density + pj.density)*f1_t(0.5);
                II = -0.5f * alpha * wij * vsig / rhoij;
            }
            pi.acc -= pj.mass * II * gradw;

            // pressure and acceleration
//            m3_t stress_j = m3_t(-eos::murnaghan( pj.density, rho0, BULK, dBULKdP)) + pj.dstress;
//            m3_t sigOverRho_j = stress_j / (pj.density*pj.density);
            f1_t sigOverRho_j = -eos::murnaghan( pj.density, rho0, BULK, dBULKdP) / (pj.density*pj.density);
            pi.acc += pj.mass * (sigOverRho_i + sigOverRho_j) * gradw;

            // strain rate tensor (edot) and rotation rate tensor (rdot)
            m3_t alphaOverBeta( vij.x*gradw.x, vij.x*gradw.y, vij.x*gradw.z,
                                vij.y*gradw.x, vij.y*gradw.y, vij.y*gradw.z,
                                vij.z*gradw.x, vij.z*gradw.y, vij.z*gradw.z );
            m3_t betaOverAlpha = mpu::transpose(alphaOverBeta);

            edot += pj.mass * (alphaOverBeta + betaOverAlpha);
            rdot += pj.mass * (alphaOverBeta - betaOverAlpha);

            // density time derivative
            vdiv += (pj.mass/pj.density) * (alphaOverBeta[0][0]+alphaOverBeta[1][1]+alphaOverBeta[2][2]);
        }
    },
    {
        // density time derivative
        pi.density_dt = pi.density * vdiv;

        // deviatoric stress time derivative
        edot *= f1_t(0.5)/pi.density;
        rdot *= f1_t(0.5)/pi.density;

        pi.dstress_dt = 2*shear*edot + pi.dstress*rdot + rdot*pi.dstress;
        f1_t edotOverThree = 2*shear*(edot[0][0]+edot[1][1]+edot[2][2]) / f1_t(3.0);
        pi.dstress_dt[0][0] -= edotOverThree;
        pi.dstress_dt[1][1] -= edotOverThree;
        pi.dstress_dt[2][2] -= edotOverThree;
    })
}

__global__ void integrate(DeviceParticlesType particles, f1_t dt)
{
    DO_FOR_EACH(particles, MPU_COMMA_LIST(POS,VEL,ACC,DENSITY,DENSITY_DT,DSTRESS,DSTRESS_DT),
            MPU_COMMA_LIST(POS,VEL,ACC,DENSITY,DENSITY_DT,DSTRESS,DSTRESS_DT),
            MPU_COMMA_LIST(POS,VEL,DENSITY,DSTRESS),
    {
        // eqn of motion
        pi.vel += pi.acc * dt;
        pi.pos += pi.vel * dt;

        // density
        pi.density += pi.density_dt * dt;

        // deviatoric stress
        pi.dstress += pi.dstress_dt * dt;
    })
}

__global__ void generate2DNBSystem(DeviceParticlesType particles)
{
    INIT_EACH(particles, MPU_COMMA_LIST(POS,MASS,VEL),
              {
                  thrust::random::default_random_engine rng;
                  rng.discard(i);
                  thrust::random::uniform_real_distribution<float> dist(-1.0f,1.0f);

                  pi.pos.x = dist(rng);
                  pi.pos.y = dist(rng);
                  pi.pos.z = 0.0f;
                  pi.mass = 1.0f/particles.size();

                  pi.vel = cross(pi.pos,{0.0f,0.0f, 0.75f});
              });
}

__global__ void nbodyForces(DeviceParticlesType particles, f1_t eps2)
{
    DO_FOR_EACH_PAIR_SM( BLOCK_SIZE, particles, SHARED_POSM, MPU_COMMA_LIST(POS,MASS,VEL,ACC),
                         MPU_COMMA_LIST(POS,MASS,VEL), MPU_COMMA_LIST(ACC), MPU_COMMA_LIST(POS, MASS), {},
    {
        f3_t r = pi.pos - pj.pos;
        f1_t distSqr = dot(r, r) + eps2;

        f1_t invDist = rsqrt(distSqr);
        f1_t invDistCube = invDist * invDist * invDist;
        pi.acc -= r * pj.mass * invDistCube;
    },
    {
        pi.acc -= pi.vel * 0.01;
    })
}

__global__ void integrateLeapfrog(DeviceParticlesType particles, f1_t dt, bool not_first_step)
{
    DO_FOR_EACH(particles, MPU_COMMA_LIST(POS,VEL,ACC), MPU_COMMA_LIST(POS,VEL,ACC), MPU_COMMA_LIST(POS,VEL),
    {
        //   calculate velocity a_t
        pi.vel = pi.vel + pi.acc * (dt * 0.5f);

        // we could now change delta t here

        // calculate velocity a_t+1/2
        pi.vel = pi.vel + pi.acc * (dt * 0.5f) * not_first_step;

        // calculate position r_t+1
        pi.pos = pi.pos + pi.vel * dt;
    })
}

int main()
{
    mpu::Log myLog( mpu::LogLvl::ALL, mpu::ConsoleSink());

    logINFO("pfSPH") << "Welcome to planetformSPH!";
    assert_cuda(cudaSetDevice(0));

    // set up frontend
    fnd::initializeFrontend();
    bool simShouldRun = true;
    fnd::setPauseHandler([&simShouldRun](bool pause){simShouldRun = !pause;});

    // generate some particles
    DeviceParticlesType pb(PARTICLES);
    pb.initialize();

    // register position and velocity buffer with cuda
#if defined(FRONTEND_OPENGL)
    pb.registerGLGraphicsResource<DEV_POSM>(fnd::getPositionBuffer(pb.size()));
    pb.registerGLGraphicsResource<DEV_VEL>(fnd::getVelocityBuffer(pb.size()));
    pb.mapGraphicsResource();
#endif

    generate2DRings<<<NUM_BLOCKS,BLOCK_SIZE>>>(pb.createDeviceCopy());
    assert_cuda(cudaGetLastError());
    assert_cuda(cudaDeviceSynchronize());

    pb.unmapGraphicsResource(); // used for frontend stuff
    while(fnd::handleFrontend())
    {
        if(simShouldRun)
        {
            pb.mapGraphicsResource(); // used for frontend stuff

            computeDerivatives<<<NUM_BLOCKS,BLOCK_SIZE>>>(pb.createDeviceCopy(),SOUNDSPEED);
            assert_cuda(cudaGetLastError());
            integrate<<<NUM_BLOCKS,BLOCK_SIZE>>>(pb.createDeviceCopy(),0.0001f);
            assert_cuda(cudaGetLastError());

            pb.unmapGraphicsResource(); // used for frontend stuff
        }
    }

    return 0;
}
