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
#include <cmath>

#include "frontends/frontendInterface.h"
#include "particles/Particles.h"
#include "particles/algorithms.h"
#include "sph/kernel.h"
#include "sph/eos.h"

constexpr int BLOCK_SIZE = 256;
constexpr int PARTICLES = 1<<14;

constexpr float pradius = 0.002;
constexpr float a = pradius * pradius * M_PI;
constexpr float mass = 1.0 / PARTICLES;

constexpr f1_t H = pradius*2.5;

constexpr f1_t alpha = 1;
constexpr f1_t rho0 = mass / a;
constexpr f1_t BULK = 64;
constexpr f1_t dBULKdP = 8;
constexpr f1_t shear = 48;
const f1_t SOUNDSPEED = sqrt(BULK / rho0);

constexpr f1_t mateps = 0.0;
constexpr f1_t matexp = 4;
constexpr f1_t normalsep = H*0.3;

constexpr f1_t friction_angle = 55.0f * (M_PI/180.0f);
constexpr f1_t cohesion = 0.01;

int NUM_BLOCKS = (PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE;

using DeviceParticlesType = Particles<DEV_POSM,DEV_VEL,DEV_ACC,DEV_XVEL,DEV_DENSITY,DEV_DENSITY_DT,DEV_DSTRESS,DEV_DSTRESS_DT>;

__global__ void generate2DRings(DeviceParticlesType particles)
{

    const float R = 0.38;
    const float r = 0.3;
    const float seperationX = 1;
    const float seperationY = 0;
    const float speed = 0.5;

    const float ringSize = particles.size()/2;
    const float a = M_PI * (R*R-r*r);
    const float ringMass = rho0 * a;
    const float particleMass = ringMass/ringSize;

    // find the starting index
    int startingIndex = (r*r) * ringSize;
    int lastIteration=0;
    while(abs(startingIndex-lastIteration)>5)
    {
        lastIteration = startingIndex;
        startingIndex = ((r/R)*(r/R)) * (ringSize+startingIndex);
    }

    // calculate the particle distance
    f2_t posA;
    f2_t posB;
    float l = R * sqrt(10/(ringSize+startingIndex));
    float theta = 2 * sqrt(M_PIf32*10);
    posA.x = l * cos(theta);
    posA.y = l * sin(theta);
    l = R * sqrt(11/(ringSize+startingIndex));
    theta = 2 * sqrt(M_PIf32*11);
    posB.x = l * cos(theta);
    posB.y = l * sin(theta);
    f1_t spacing = length(posA-posB);
    printf("particle seperation: %f\n ", spacing);

    INIT_EACH(particles, MPU_COMMA_LIST(POS,MASS,VEL,DENSITY),
    {
        thrust::random::default_random_engine rng;
        rng.discard(i);
        thrust::random::uniform_real_distribution<float> dist(-0.1f*spacing,0.1f*spacing);

        float index;
        if(i<ringSize)
        {
             index = i + startingIndex;
             pi.pos.x = seperationX/2;
             pi.pos.y = seperationY/2;
             pi.vel.x = -speed/2;
        }
        else
        {
            index = i-ringSize + startingIndex;
            pi.pos.x = -seperationX/2;
            pi.pos.y = -seperationY/2;
            pi.vel.x = speed/2;
        }

        l = R * sqrt(index/(ringSize+startingIndex));
        theta = 2 * sqrt(M_PIf32*index);
        pi.pos.x += l * cos(theta);
        pi.pos.y += l * sin(theta);

        pi.mass = particleMass;
        pi.density = rho0;
    });
}

__global__ void generateSquares(DeviceParticlesType particles)
{
    INIT_EACH(particles, MPU_COMMA_LIST(POS,MASS,VEL,DENSITY),
    {
        float ratio = 0.99f;
        float spacing = H/3;
        printf("spacing: %f\n",spacing);
        int squareSize1 = particles.size() * ratio;
        int squareSize2 = particles.size() * (1.0f-ratio);
        int sideres1 = sqrt(float(squareSize1));
        int sideres2 = sqrt(float(squareSize2));
        float side1 = (sideres1-1) * spacing;
        float side2 = (sideres2-1) * spacing;

        const float a1 = side1*side1;
        const float a2 = side2*side2;
        const float squareMass1 = rho0 * a1;
        const float squareMass2 = rho0 * a2;
        const float particleMass1 = squareMass1/squareSize1;
        const float particleMass2 = squareMass2/squareSize2;

        const float speed = 5;
        const float seperationX = 1;
        const float seperationY = 0;

        thrust::random::default_random_engine rng;
        rng.discard(i);
        thrust::random::uniform_real_distribution<float> dist(-0.1f*spacing,0.1f*spacing);

        if(i < squareSize1)
        {
            pi.pos.x = -side1 / 2 + (i%sideres1) *spacing + dist(rng);
            pi.pos.y = -side1 / 2 + (i/sideres1) *spacing + dist(rng);
            pi.pos.x -= seperationX/2;
            pi.pos.y -= seperationY/2;
            pi.vel.x = speed * (1-ratio);
            pi.mass = particleMass1;
        }
        else
        {
            pi.pos.x = -side2 / 2 + ((i-squareSize1)%sideres2) *spacing + dist(rng);
            pi.pos.y = -side2 / 2 + ((i-squareSize1)/sideres2) *spacing + dist(rng);
            pi.pos.x += seperationX/2;
            pi.pos.y += seperationY/2;
            pi.vel.x = -speed * ratio;
            pi.mass = particleMass2;
        }
        pi.density = rho0;
    })
}

__global__ void generateRect(DeviceParticlesType particles)
{
    INIT_EACH(particles, MPU_COMMA_LIST(POS,MASS,VEL,DENSITY),
    {
        int rows = 150;
        float offsetY = -0.45;

        float spacing = H/3;
        printf("spacing: %f\n",spacing);

        int coatCols = 2.0f/spacing;
        int coatSize = coatCols * 3;

        int size = particles.size() - coatSize;
        int cols = size / rows;
        float side1 = (cols-1) * spacing;
        float side2 = (rows-1) * spacing;
        const float a = side1*side2;
        const float mass = rho0 * a;
        const float particleMass = mass/size;

        thrust::random::default_random_engine rng;
        rng.discard(i);
        thrust::random::uniform_real_distribution<float> dist(-0.1f*spacing,0.1f*spacing);

        if(i < coatSize)
        {
            pi.pos.x = -1.0f + ((i) % coatCols) * spacing + dist(rng);
            pi.pos.y = -0.99f + ((i) / coatCols) * spacing + dist(rng);
        }
        else
        {
            pi.pos.x = -side1 / 2 + ((i-coatSize) % cols) * spacing + dist(rng);
            pi.pos.y = -side2 / 2 + ((i-coatSize) / cols) * spacing + dist(rng);
            pi.pos.y += offsetY;
        }
        pi.mass = particleMass;
        pi.density = rho0;
    })
}

__global__ void generate2DHydroNBSystem(DeviceParticlesType particles)
{
    INIT_EACH(particles, MPU_COMMA_LIST(POS,MASS,VEL,DENSITY),
              {
                  thrust::random::default_random_engine rng;
                  rng.discard(i*particles.size());
                  thrust::random::uniform_real_distribution<float> dist(-1.0f,1.0f);

                  do
                  {
                      pi.pos.x = dist(rng);
                      rng.discard(particles.size());
                      pi.pos.y = dist(rng);
                      rng.discard(particles.size());
                  }
                  while(length(pi.pos) > 1);

                  pi.mass = mass;
                  pi.density = rho0;

                  pi.vel.x = pi.pos.y * sqrtf(1);
                  pi.vel.y = - pi.pos.x * sqrtf(1);
              });
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

__device__ void plasticity(m3_t& destress, f1_t pressure)
{
    // second invariant of deviatoric stress
    f1_t J2 = 0;
    for(uint e = 0; e < 9; ++e)
        J2 += destress(e) * destress(e);
    J2 *= 0.5f;

    f1_t Y = tan(friction_angle) * pressure + cohesion;
    f1_t miese_f = min(  Y*Y /(3.0f*J2),1.0f);

    for(uint e = 0; e < 9; ++e)
        destress(e) = miese_f * destress(e);
}

__global__ void computeDensity(DeviceParticlesType particles)
{
    DO_FOR_EACH_PAIR_SM( BLOCK_SIZE, particles, MPU_COMMA_LIST(SHARED_POSM),
                         MPU_COMMA_LIST(POS,MASS,DENSITY),
                         MPU_COMMA_LIST(POS,MASS), MPU_COMMA_LIST(DENSITY),
                         MPU_COMMA_LIST(POS,MASS),
    {},
    {
        const f3_t rij = pi.pos-pj.pos;
        const f1_t r2 = dot(rij,rij);
        f1_t r = sqrt(r2);
        if(r<=H)
        {
            pi.density += pj.mass * kernel::Wspline<Dim::two>(r,H);
        }
    },
    {})
}

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
                const f1_t dw = kernel::dWspline<Dim::two>(r, H);
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
                f1_t f = kernel::Wspline<Dim::two>(r, H) / kernel::Wspline<Dim::two>(normalsep, H);
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
                pi.xvel += 2 * pj.mass / (pi.density + pj.density) * (pj.vel - pi.vel) * kernel::Wspline<Dim::two>(r, H);
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

__global__ void window2DBound(DeviceParticlesType particles)
{
    DO_FOR_EACH(particles, MPU_COMMA_LIST(POS,VEL),
                MPU_COMMA_LIST(POS,VEL),
                MPU_COMMA_LIST(POS,VEL),
    {
        if(pi.pos.x > 1)
        {
            pi.pos.x=1;
            pi.vel.x -= 1.5*pi.vel.x;
        }
        else if(pi.pos.x < -1)
        {
            pi.pos.x=-1;
            pi.vel.x -= 1.5*pi.vel.x;
        }
        if(pi.pos.y > 1)
        {
            pi.pos.y=1;
            pi.vel.y -= 1.5*pi.vel.y;
        }
        else if(pi.pos.y < -1)
        {
            pi.pos.y=-1;
            pi.vel.y = 0;
            pi.vel.x = 0;

        }
    })
}

__global__ void integrate(DeviceParticlesType particles, f1_t dt)
{
    DO_FOR_EACH(particles, MPU_COMMA_LIST(POS,VEL,ACC,XVEL,DENSITY,DENSITY_DT,DSTRESS,DSTRESS_DT),
            MPU_COMMA_LIST(POS,VEL,ACC,XVEL,DENSITY,DENSITY_DT,DSTRESS,DSTRESS_DT),
            MPU_COMMA_LIST(POS,VEL,DENSITY,DSTRESS),
    {
        // eqn of motion
        pi.vel += pi.acc * dt;
        pi.pos += (pi.vel+0.6f*pi.xvel) * dt;

        // density
        pi.density += pi.density_dt * dt;

        if(pi.density < 0)
            pi.density = 0;

        // deviatoric stress
        pi.dstress += pi.dstress_dt * dt;

        plasticity(pi.dstress,eos::murnaghan(pi.density,rho0, BULK, dBULKdP));
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

    logINFO("sim") << "particle mass: " << mass;
    logINFO("sim") << "particle radius: " << pradius;
    logINFO("sim") << "material density: " << rho0;
    logINFO("sim") << "total mass: " << mass * PARTICLES;
    logINFO("sim") << "total radius: " << 1;
    logINFO("sim") << "total surface density: " <<  mass * PARTICLES / (1*1*M_PI);

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

    generate2DHydroNBSystem<<<NUM_BLOCKS,BLOCK_SIZE>>>(pb.createDeviceCopy());
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
            integrate<<<NUM_BLOCKS,BLOCK_SIZE>>>(pb.createDeviceCopy(),0.0002f);
            assert_cuda(cudaGetLastError());

            pb.unmapGraphicsResource(); // used for frontend stuff
        }
    }

    return 0;
}
