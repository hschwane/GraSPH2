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
constexpr f1_t H = 0.022;

constexpr f1_t alpha = 1;
constexpr f1_t rho0 = 0.5;
constexpr f1_t BULK = 10;
constexpr f1_t dBULKdP = 1;
constexpr f1_t shear = 6;
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
    const float speed = .25;


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

__global__ void generateSquares(DeviceParticlesType particles)
{
    INIT_EACH(particles, MPU_COMMA_LIST(POS,MASS,VEL,DENSITY),
    {
        float spacing = H/2;
        int squareSize = particles.size()/2;
        int sideres = sqrt(float(squareSize));
        float side = (sideres-1) * spacing;

        const float a = side*side;
        const float squareMass = rho0 * a;
        const float particleMass = squareMass/squareSize;

        const float speed = .25;
        const float seperation = 1;

        if(i < squareSize)
        {
            pi.pos.x = -side / 2 + (i%sideres) *spacing;
            pi.pos.y = -side / 2 + (i/sideres) *spacing;
            pi.pos.x -= seperation/2;
            pi.pos.y -= seperation/4;
            pi.vel.x = speed;
        }
        else
        {
            pi.pos.x = -side / 2 + ((i-squareSize)%sideres) *spacing;
            pi.pos.y = -side / 2 + ((i-squareSize)/sideres) *spacing;
            pi.pos.x += seperation/2;
            pi.pos.y += seperation/4;
            pi.vel.x = -speed;
        }

        pi.mass = particleMass;
        pi.density = rho0;
    })
}

__device__ f1_t artificialViscosity(f1_t alpha, f1_t density_i, f1_t density_j, const f3_t& vij,  const f3_t& rij, f1_t r, f1_t ci, f1_t cj)
{
    const f1_t wij = dot(rij, vij) /r;
    f1_t II = 0;
    if(wij < 0)
    {
        const f1_t vsig = f1_t(ci+cj - 3.0*wij);
        const f1_t rhoij = (density_i + density_j)*f1_t(0.5);
        II = -0.5f * alpha * wij * vsig / rhoij;
    }
    return II;
}

__global__ void computeDerivatives(DeviceParticlesType particles, f1_t speedOfSound)
{
    DO_FOR_EACH_PAIR_SM( BLOCK_SIZE, particles, MPU_COMMA_LIST(SHARED_POSM,SHARED_VEL,SHARED_DENSITY,SHARED_DSTRESS),
            MPU_COMMA_LIST(POS,MASS,VEL,ACC,DENSITY,DENSITY_DT,DSTRESS,DSTRESS_DT),
            MPU_COMMA_LIST(POS,MASS,VEL,DENSITY,DSTRESS), MPU_COMMA_LIST(ACC,DENSITY_DT,DSTRESS_DT),
            MPU_COMMA_LIST(POS,MASS,VEL,DENSITY,DSTRESS),

    int numPartners=0;
    m3_t sigOverRho_i; // stress over density square used for acceleration
    m3_t edot(0); // strain rate tensor (edot)
    m3_t rdot(0); // rotation rate tensor
    f1_t vdiv{0}; // velocity divergence
    {
        sigOverRho_i = pi.dstress;
        f1_t pres_i = eos::murnaghan( pi.density, rho0, BULK, dBULKdP);
        sigOverRho_i[0][0] = (sigOverRho_i[0][0] - pres_i) / (pi.density*pi.density);
        sigOverRho_i[1][1] = (sigOverRho_i[1][1] - pres_i) / (pi.density*pi.density);
        sigOverRho_i[2][2] = (sigOverRho_i[2][2] - pres_i) / (pi.density*pi.density);
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
            pi.acc -= pj.mass * artificialViscosity(alpha,pi.density,pj.density,vij,rij,r,speedOfSound,speedOfSound) * gradw;

            // pressure
            m3_t sigOverRho_j = pj.dstress;
            f1_t pres_j = eos::murnaghan( pj.density, rho0, BULK, dBULKdP);
            sigOverRho_j[0][0] = (sigOverRho_j[0][0] - pres_j) / (pj.density*pj.density);
            sigOverRho_j[1][1] = (sigOverRho_j[1][1] - pres_j) / (pj.density*pj.density);
            sigOverRho_j[2][2] = (sigOverRho_j[2][2] - pres_j) / (pj.density*pj.density);

            // acceleration
            pi.acc.x += pj.mass * ((sigOverRho_i[0][0]+sigOverRho_j[0][0])*gradw.x + (sigOverRho_i[0][1]+sigOverRho_j[0][1])*gradw.y + (sigOverRho_i[0][2]+sigOverRho_j[0][2])*gradw.z);
            pi.acc.y += pj.mass * ((sigOverRho_i[1][0]+sigOverRho_j[1][0])*gradw.x + (sigOverRho_i[1][1]+sigOverRho_j[1][1])*gradw.y + (sigOverRho_i[1][2]+sigOverRho_j[1][2])*gradw.z);
            pi.acc.z += pj.mass * ((sigOverRho_i[2][0]+sigOverRho_j[2][0])*gradw.x + (sigOverRho_i[2][1]+sigOverRho_j[2][1])*gradw.y + (sigOverRho_i[2][2]+sigOverRho_j[2][2])*gradw.z);

            // strain rate tensor (edot) and rotation rate tensor (rdot)
            f1_t tmp= -0.5f * pj.mass/pi.density;
            edot[0][0] += tmp*(vij.x*gradw.x + vij.x*gradw.x);
            edot[0][1] += tmp*(vij.x*gradw.y + vij.y*gradw.x);
            edot[0][2] += tmp*(vij.x*gradw.z + vij.z*gradw.x);
            edot[1][0] += tmp*(vij.y*gradw.x + vij.x*gradw.y);
            edot[1][1] += tmp*(vij.y*gradw.y + vij.y*gradw.y);
            edot[1][2] += tmp*(vij.y*gradw.z + vij.z*gradw.y);
            edot[2][0] += tmp*(vij.z*gradw.x + vij.x*gradw.z);
            edot[2][1] += tmp*(vij.z*gradw.y + vij.y*gradw.z);
            edot[2][2] += tmp*(vij.z*gradw.z + vij.z*gradw.z);

            rdot[0][0] += tmp*(vij.x*gradw.x - vij.x*gradw.x);
            rdot[0][1] += tmp*(vij.x*gradw.y - vij.y*gradw.x);
            rdot[0][2] += tmp*(vij.x*gradw.z - vij.z*gradw.x);
            rdot[1][0] += tmp*(vij.y*gradw.x - vij.x*gradw.y);
            rdot[1][1] += tmp*(vij.y*gradw.y - vij.y*gradw.y);
            rdot[1][2] += tmp*(vij.y*gradw.z - vij.z*gradw.y);
            rdot[2][0] += tmp*(vij.z*gradw.x - vij.x*gradw.z);
            rdot[2][1] += tmp*(vij.z*gradw.y - vij.y*gradw.z);
            rdot[2][2] += tmp*(vij.z*gradw.z - vij.z*gradw.z);


            // density time derivative
            vdiv += (pj.mass/pj.density) * dot(vij,gradw);
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
    bool simShouldRun = false;
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

    generateSquares<<<NUM_BLOCKS,BLOCK_SIZE>>>(pb.createDeviceCopy());
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
            integrate<<<NUM_BLOCKS,BLOCK_SIZE>>>(pb.createDeviceCopy(),0.0003f);
            assert_cuda(cudaGetLastError());

            pb.unmapGraphicsResource(); // used for frontend stuff
        }
    }

    return 0;
}
