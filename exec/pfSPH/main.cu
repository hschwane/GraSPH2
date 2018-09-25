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
constexpr f1_t H = 0.03;

constexpr f1_t alpha =0.6;
constexpr f1_t rho0 = 1;
constexpr f1_t BULK = 8;
constexpr f1_t dBULKdP = 1;
constexpr f1_t shear = 6;
const f1_t SOUNDSPEED = sqrt(BULK / rho0);

constexpr f1_t mateps =0.4;
constexpr f1_t matexp =4;

int NUM_BLOCKS = (PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE;

using DeviceParticlesType = Particles<DEV_POSM,DEV_VEL,DEV_ACC,DEV_DENSITY>;

__global__ void generate2DRings(DeviceParticlesType particles)
{

    const float R = 0.4;
    const float r = 0.25;
    const float seperationX = 1;
    const float seperationY = 0;
    const float speed = 1;


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
    printf("particle seperation: %f\n ",length(posA-posB));

    INIT_EACH(particles, MPU_COMMA_LIST(POS,MASS,VEL,DENSITY),
    {
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
        float spacing = H/2.5f;
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
            pi.pos.y -= seperation/10;
            pi.vel.x = speed;
        }
        else
        {
            pi.pos.x = -side / 2 + ((i-squareSize)%sideres) *spacing;
            pi.pos.y = -side / 2 + ((i-squareSize)/sideres) *spacing;
            pi.pos.x += seperation/2;
            pi.pos.y += seperation/10;
            pi.vel.x = -speed;
        }

        pi.mass = particleMass;
        pi.density = rho0;
    })
}

int generateFromImage(std::string filename, DeviceParticlesType& dpart)
{
    float particleMass = 0.0001125;

    Particles<HOST_POSM> pb(PARTICLES);

    int Xres,Yres,n;
    unsigned char *data = stbi_load(filename.c_str(), &Xres, &Yres, &n, 3);
    if(!data) throw std::runtime_error("unable to open file " + filename);
    int particleCount=0;

    float spacing = 2.0/Xres;
    for(int y = 0; y < Yres; ++y)
        for(int x = 0; x < Xres; ++x)
        {
            float3 c{float(data[3*Xres*y+3*x+0])/255,float(data[3*Xres*y+3*x+1])/255,float(data[3*Xres*y+3*x+2])/255};
            if(length(c) > 0.8 && particleCount < pb.size())
            {
                Particle<POS,MASS> p;
                p.mass = particleMass;
                p.pos.x = -1 + x*spacing;
                p.pos.y = 1 - y*spacing;
                pb.storeParticle(particleCount,p);
                particleCount++;
            }
        }

    int used = particleCount;
    while(particleCount<pb.size())
    {
        Particle<POS,MASS> p;
        p.mass = 0;
        p.pos.x = 3;
        p.pos.y = 3;
        pb.storeParticle(particleCount,p);
        particleCount++;
    }

    stbi_image_free(data);
    dpart=pb;
    return used;
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

__global__ void computeDensity(DeviceParticlesType particles)
{
    DO_FOR_EACH_PAIR_SM( BLOCK_SIZE, particles, MPU_COMMA_LIST(SHARED_POSM),
                     MPU_COMMA_LIST(POS,MASS,DENSITY),
                     MPU_COMMA_LIST(POS,MASS), MPU_COMMA_LIST(DENSITY),
                     MPU_COMMA_LIST(POS,MASS),
    float prefactor;
    {
        prefactor = kernel::detail::splinePrefactor<Dim::two>(H);
    },
    {
        const f3_t rij = pi.pos-pj.pos;
        const f1_t r2 = dot(rij,rij);
        f1_t r = sqrt(r2);
        if(r<=H)
        {
           pi.density += pj.mass * kernel::Wspline(r,H,prefactor);
        }
    },
    {})
}

__global__ void computeDerivatives(DeviceParticlesType particles, f1_t speedOfSound, f2_t extForces)
{
    DO_FOR_EACH_PAIR_SM( BLOCK_SIZE, particles, MPU_COMMA_LIST(SHARED_POSM,SHARED_VEL,SHARED_DENSITY),
            MPU_COMMA_LIST(POS,MASS,VEL,ACC,DENSITY),
            MPU_COMMA_LIST(POS,MASS,VEL,DENSITY), MPU_COMMA_LIST(ACC),
            MPU_COMMA_LIST(POS,MASS,VEL,DENSITY),

    int numPartners=0;
    float prefactor;
    f1_t pOverRho_i; // pressure over density square used for acceleration
//    f1_t apres_i=0;
    {
        f1_t pres_i = eos::liquid( pi.density, rho0, speedOfSound*speedOfSound);
        pOverRho_i =  pres_i / (pi.density * pi.density);
        prefactor = kernel::detail::dspikyPrefactor<Dim::two>(H);

//        apres_i = (pres_i > 0) ? mateps * (pres_i)/(pi.density * pi.density) : 0;
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
            const f1_t dw = kernel::dWspiky(r,H,prefactor);
            const f3_t gradw = (dw/r) * rij;

            // artificial viscosity
            const f3_t vij = pi.vel-pj.vel;
            f1_t II = artificialViscosity(alpha,pi.density,pj.density,vij,rij,r,speedOfSound,speedOfSound);

            // pressure
            f1_t pres_j = eos::liquid( pj.density, rho0, speedOfSound*speedOfSound);
            f1_t pOverRho_j = pres_j / (pj.density * pj.density);

            // artificial pressure
//            f1_t apres_j = 0;
//            apres_j = (pres_j > 0)? mateps * (pres_j)/(pj.density * pj.density) : 0;
//
//            f1_t apres=apres_i+apres_j;
//            f1_t f= pow(kernel::Wspline<Dim::two>(r,H) / kernel::Wspline<Dim::two>(H/2.5f,H),matexp);

            // acc
            pi.acc -= pj.mass * (pOverRho_i + pOverRho_j + II /* + f*apres*/) * gradw;
        }
    },
    {
//        printf("%i\n",numPartners);
        pi.acc.x += extForces.x;
        pi.acc.y += extForces.y;
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
                        pi.vel.y -= 1.5*pi.vel.y;
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

#if defined(FRONTEND_OPENGL)
namespace fnd {
extern glm::vec2 getWindowAcc();
}
#else
namespace fnd {
glm::vec2 fnd::getWindowAcc() {return glm::vec2(0)}
}
#endif

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

//    generateSquares<<<NUM_BLOCKS,BLOCK_SIZE>>>(pb.createDeviceCopy());
//    assert_cuda(cudaGetLastError());
//    assert_cuda(cudaDeviceSynchronize());
    int used=generateFromImage("/home/hendrik/sphInit.png",pb);
    logINFO("LOADER") << "Used " << used << " from " << PARTICLES << " particles to recreate image";

    computeDensity<<<NUM_BLOCKS,BLOCK_SIZE>>>(pb.createDeviceCopy());
    assert_cuda(cudaGetLastError());
    computeDerivatives<<<NUM_BLOCKS,BLOCK_SIZE>>>(pb.createDeviceCopy(),SOUNDSPEED,f2_t{0,-1});
    assert_cuda(cudaGetLastError());
    integrateLeapfrog<<<NUM_BLOCKS,BLOCK_SIZE>>>(pb.createDeviceCopy(),0.0025f, false);
    assert_cuda(cudaGetLastError());
    window2DBound<<<NUM_BLOCKS,BLOCK_SIZE>>>(pb.createDeviceCopy());
    assert_cuda(cudaGetLastError());

    fnd::getWindowAcc();

    pb.unmapGraphicsResource(); // used for frontend stuff
    while(fnd::handleFrontend())
    {
        glm::vec2 windowAcc = fnd::getWindowAcc();

        if(simShouldRun)
        {
            pb.mapGraphicsResource(); // used for frontend stuff

            f2_t extAcc{ -0.4f*windowAcc.x,-0.4f*windowAcc.y-1};

            computeDensity<<<NUM_BLOCKS,BLOCK_SIZE>>>(pb.createDeviceCopy());
            assert_cuda(cudaGetLastError());
            computeDerivatives<<<NUM_BLOCKS,BLOCK_SIZE>>>(pb.createDeviceCopy(),SOUNDSPEED, extAcc);
            assert_cuda(cudaGetLastError());
            integrateLeapfrog<<<NUM_BLOCKS,BLOCK_SIZE>>>(pb.createDeviceCopy(),0.0025f,true);
            assert_cuda(cudaGetLastError());
            window2DBound<<<NUM_BLOCKS,BLOCK_SIZE>>>(pb.createDeviceCopy());
            assert_cuda(cudaGetLastError());

            pb.unmapGraphicsResource(); // used for frontend stuff
        }
    }

    return 0;
}
