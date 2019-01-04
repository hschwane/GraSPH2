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

//#include "initialConditions/InitGenerator.h"
//#include "initialConditions/particleSources/UniformSphere.h"
//#include "initialConditions/particleSources/TextFile.h"
//#include "frontends/frontendInterface.h"
//#include "particles/Particles.h"
//#include "sph/kernel.h"
//#include "sph/eos.h"
//#include "sph/models.h"
//#include "ResultStorageManager.h"
//#include "settings/settings.h"

#include "particles/Particles.h"

struct TestAlgorithmB
{
    using load_type = Particle<POS>;
    using store_type = Particle<MASS>;
    using pi_type = merge_particles_t<load_type,store_type>;

    using pj_type = Particle<POS,MASS>;

    template<size_t n>
    using shared = SharedParticles<n,SHARED_POSM>;

    int m_id{-1};
    int j{0};

    CUDAHOSTDEV void do_before(pi_type& pi, size_t id)
    {
        m_id = id;
    }

    CUDAHOSTDEV void do_for_each_pair(pi_type& pi, pj_type pj)
    {
        printf( "i: %i j: %i \n" , m_id , j++ );
    }

    CUDAHOSTDEV store_type do_after(pi_type& pi)
    {
        return pi;
    }
};

int main()
{
//    HostParticleBuffer<HOST_POS,HOST_MASS> pb(128);
    DeviceParticleBuffer<DEV_POS,DEV_MASS > pb(10);

    pb.storeParticle(1, Particle<POS>(f3_t{2.0f,2.0f,2.0f}));

    do_for_each_pair_fast<TestAlgorithmB>(pb);


    auto p = pb.loadParticle(4);
    std::cout << p.mass << std::endl;



//    Particle<MASS> p;
//    p.mass = 10;
//
//    HOST_MASS hm1(10);
//    DEV_MASS hm2(10);
//
//    hm1.storeParticle(1,p);
//    hm2.storeParticle(1,p);
//
//    DEV_MASS hm3(hm1);
//    HOST_MASS hm4(hm2);
//
//    hm1 = hm3;
//    hm2 = hm4;
//
//    HOST_MASS hm5(hm1);
//    DEV_MASS hm6(hm2);
//
//    hm1 = hm5;
//    hm2 = hm6;
//
//    p.mass = 0;
//    hm1.loadParticle(1,p);
//    std::cout << p.mass <<std::endl;
//
//    p.mass = 0;
//    hm2.loadParticle(1,p);
//    std::cout << p.mass <<std::endl;
//
//    DeviceParticleBuffer<DEV_POS, DEV_MASS> dpb1(600);
//    using test=DeviceParticleBuffer<DEV_POS, DEV_MASS>;
//
//    std::cout << typeid(test).name() << std::endl;
//    std::cout << typeid(typename test::particleType).name() << std::endl;
//    std::cout << typeid(typename test::hostType).name() << std::endl;
//    std::cout << typeid(typename test::referenceType).name() << std::endl;
//    std::cout << typeid(typename test::hostType::deviceType).name() << std::endl;
//
//
//    dpb1.initialize();
//
//    auto hpb1 = dpb1.getHostBuffer();
//
//    auto p2 = hpb1.loadParticle(5);
//
//    std::cout << typeid(decltype(p2)).name() << " -VALUE FOR MASS: " << p2.mass << std::endl;


}




//    constexpr f1_t H2 = H*H; //!< square of the smoothing length
//
//    /**
//     * @brief computes deriviatives of particle attributes
//     * @param particles device copy of the particle Buffer
//     * @param speedOfSound your materials sound speed
//     */
//    __global__ void computeDerivatives(DeviceParticlesType particles, f1_t speedOfSound)
//    {
//        DO_FOR_EACH_PAIR_SM( BLOCK_SIZE, particles, MPU_COMMA_LIST(SHARED_POSM,SHARED_VEL,SHARED_DENSITY,SHARED_DSTRESS),
//                MPU_COMMA_LIST(POS,MASS,VEL,ACC,XVEL,DENSITY,DENSITY_DT,DSTRESS,DSTRESS_DT),
//                MPU_COMMA_LIST(POS,MASS,VEL,DENSITY,DSTRESS), MPU_COMMA_LIST(ACC,XVEL,DENSITY_DT,DSTRESS_DT),
//                MPU_COMMA_LIST(POS,MASS,VEL,DENSITY,DSTRESS),
//
//    //    DO_FOR_EACH_PAIR(particles,
//    //            MPU_COMMA_LIST(POS,MASS,VEL,ACC,XVEL,DENSITY,DENSITY_DT,DSTRESS,DSTRESS_DT),
//    //            MPU_COMMA_LIST(POS,MASS,VEL,DENSITY,DSTRESS), MPU_COMMA_LIST(ACC,XVEL,DENSITY_DT,DSTRESS_DT),
//    //            MPU_COMMA_LIST(POS,MASS,VEL,DENSITY,DSTRESS),
//
//        // code executed for each particle once before it starts interacting with neighbours
//
//    #if defined(ENABLE_SPH)
//        m3_t sigOverRho_i; // stress over density square used for acceleration
//        m3_t sigma_i; // stress tensor
//        m3_t edot(0); // strain rate tensor (edot)
//        m3_t rdot(0); // rotation rate tensor
//        f1_t vdiv{0}; // velocity divergence
//        {
//            // build stress tensor for particle i using deviatoric stress and pressure
//            sigma_i = pi.dstress;
//            const f1_t pres_i = eos::murnaghan( pi.density, rho0, BULK, dBULKdP);
//            sigma_i[0][0] -= pres_i;
//            sigma_i[1][1] -= pres_i;
//            sigma_i[2][2] -= pres_i;
//
//            sigOverRho_i = sigma_i / (pi.density*pi.density);
//
//        #if defined(ARTIFICIAL_STRESS)
//            // artificial stress from i
//            m3_t arts_i = artificialStress(mateps, sigOverRho_i);
//        #endif
//        }
//    #endif
//        ,
//        {
//            // code run for each pair of particles
//
//            const f3_t rij = pi.pos-pj.pos;
//            const f1_t r2 = dot(rij,rij);
//            if(r2>0)
//            {
//
//    #if defined(ENABLE_SELF_GRAVITY)
//                // gravity
//                const f1_t distSqr = r2 + H2;
//                const f1_t invDist = rsqrt(distSqr);
//                const f1_t invDistCube = invDist * invDist * invDist;
//                pi.acc -= rij * pj.mass * invDistCube;
//    #endif
//
//    #if defined(ENABLE_SPH)
//                if(r2 <= H2)
//                {
//                    // get the kernel gradient
//                    f1_t r = sqrt(r2);
//                    const f1_t dw = kernel::dWspline<dimension>(r, H);
//                    const f3_t gradw = (dw / r) * rij;
//
//                    // stress and pressure of j
//                    m3_t sigma_j = pj.dstress;
//                    const f1_t pres_j = eos::murnaghan(pj.density, rho0, BULK, dBULKdP);
//                    sigma_j[0][0] -= pres_j;
//                    sigma_j[1][1] -= pres_j;
//                    sigma_j[2][2] -= pres_j;
//
//                    m3_t sigOverRho_j = sigma_j / (pj.density * pj.density);
//
//                    // stress from the interaction
//                    m3_t stress = sigOverRho_i + sigOverRho_j;
//
//                    const f3_t vij = pi.vel - pj.vel;
//        #if defined(ARTIFICIAL_VISCOSITY)
//                    // acceleration from artificial viscosity
//                    pi.acc -= pj.mass *
//                              artificialViscosity(alpha, pi.density, pj.density, vij, rij, r, speedOfSound, speedOfSound) *
//                              gradw;
//        #endif
//
//        #if defined(ARTIFICIAL_STRESS)
//                    // artificial stress
//                    const f1_t f = pow(kernel::Wspline<dimension>(r, H) / kernel::Wspline<dimension>(normalsep, H) , matexp;
//                    stress += f*(arts_i + artificialStress(mateps, sigOverRho_j));
//        #endif
//
//                    // acceleration from stress
//                    pi.acc += pj.mass * (stress * gradw);
//
//                    // strain rate tensor (edot) and rotation rate tensor (rdot)
//                    addStrainRateAndRotationRate(edot,rdot,pj.mass,pi.mass,vij,gradw);
//
//                    // density time derivative
//                    vdiv += (pj.mass / pj.density) * dot(vij, gradw);
//
//        #if defined(XSPH)
//                    // xsph
//                    pi.xvel += 2 * pj.mass / (pi.density + pj.density) * (pj.vel - pi.vel) * kernel::Wspline<dimension>(r, H);
//        #endif
//                }
//    #endif // ENABLE_SPH
//            }
//        },
//        {
//            // code run for each particle once, after interacting with its neighbours
//    #if defined(ENABLE_SPH)
//            // density time derivative
//            pi.density_dt = pi.density * vdiv;
//            // deviatoric stress time derivative
//            pi.dstress_dt = dstress_dt(edot,rdot,pi.dstress,shear);
//    #endif
//        })
//    }
//
//    /**
//     * @brief perform leapfrog integration on the particles also performs the plasticity calculations
//     * @param particles the device copy to the particle buffer that stores the particles
//     * @param dt the timestep for the integration
//     * @param not_first_step set false for the first integration step of the simulation
//     * @param tanfr tangens of the internal friction angle
//     */
//    __global__ void integrateLeapfrog(DeviceParticlesType particles, f1_t dt, bool not_first_step, f1_t tanfr)
//    {
//        DO_FOR_EACH(particles, MPU_COMMA_LIST(POS,VEL,ACC,XVEL,DENSITY,DENSITY_DT,DSTRESS,DSTRESS_DT),
//                    MPU_COMMA_LIST(POS,VEL,ACC,XVEL,DENSITY,DENSITY_DT,DSTRESS,DSTRESS_DT),
//                    MPU_COMMA_LIST(POS,VEL,DENSITY,DSTRESS),
//            {
//            //   calculate velocity a_t
//            pi.vel = pi.vel + pi.acc * (dt * 0.5f);
//
//            // we could now change delta t here
//
//            // calculate velocity a_t+1/2
//            pi.vel = pi.vel + pi.acc * (dt * 0.5f) * not_first_step;
//
//            // calculate position r_t+1
//    #if defined(XSPH) && defined(ENABLE_SPH)
//            pi.pos = pi.pos + (pi.vel + xsph_factor*pi.xvel) * dt;
//    #else
//            pi.pos = pi.pos + pi.vel * dt;
//    #endif
//
//    #if defined(ENABLE_SPH)
//            // density
//            pi.density = pi.density + pi.density_dt * dt;
//            if(pi.density < 0.0f)
//                pi.density = 0.0f;
//
//            // deviatoric stress
//            pi.dstress += pi.dstress_dt * dt;
//
//        #if defined(PLASTICITY_MC)
//            plasticity(pi.dstress, mohrCoulombYieldStress( tanfr,eos::murnaghan(pi.density,rho0, BULK, dBULKdP),cohesion));
//        #elif defined(PLASTICITY_MIESE)
//            plasticity(pi.dstress,Y);
//        #endif
//
//    #endif
//        })
//    }
//
//    /**
//     * @brief The main function of the simulation. Sets up the initial conditions and frontend and then manages running the simulation.
//     *
//     */
//    int main()
//    {
//        particleTests();
//        return 0;
//
//        mpu::Log myLog( mpu::LogLvl::ALL, mpu::ConsoleSink());
//
//        std::string buildType;
//    #if defined(NDEBUG)
//        buildType = "Release";
//    #else
//        buildType = "Debug";
//    #endif
//
//        myLog.printHeader("GraSPH2",GRASPH_VERSION,GRASPH_VERSION_SHA,buildType);
//        logINFO("GraSPH2") << "Welcome to GraSPH2!";
//        assert_cuda(cudaSetDevice(0));
//
//        // set up frontend
//        fnd::initializeFrontend();
//        bool simShouldRun = false;
//        fnd::setPauseHandler([&simShouldRun](bool pause){simShouldRun = !pause;});
//
//        // generate some particles depending on options in the settings file
//        InitGenerator<HostParticlesType> generator;
//
//    #if defined(READ_FROM_FILE)
//        ps::TextFile tf(FILENAME,SEPERATOR);
//        generator.addParticles(tf);
//    #elif defined(ROTATING_UNIFORM_SPHERE)
//        ps::UniformSphere us(particle_count,1.0,tmass,rho0);
//        us.addAngularVelocity(angVel);
//        generator.addParticles(us);
//    #endif
//
//        auto hpb = generator.generate();
//
//        // create cuda buffer
//        DeviceParticlesType pb(hpb.size());
//    #if defined(FRONTEND_OPENGL)
//        pb.registerGLGraphicsResource<DEV_POSM>(fnd::getPositionBuffer(pb.size()));
//        pb.registerGLGraphicsResource<DEV_VEL>(fnd::getVelocityBuffer(pb.size()));
//        pb.registerGLGraphicsResource<DEV_DENSITY>(fnd::getDensityBuffer(pb.size()));
//        pb.mapGraphicsResource();
//    #endif
//
//        // upload particles
//        pb = hpb;
//
//        // calculate sound speed and tangents of friction angle
//        const f1_t SOUNDSPEED = sqrt(BULK / rho0);
//        const f1_t tanfr = tan(friction_angle);
//
//    #if defined(STORE_RESULTS)
//        // set up file saving engine
//        ResultStorageManager storage(RESULT_FOLDER,RESULT_PREFIX,maxJobs);
//        storage.printToFile(pb,0);
//        f1_t timeSinceStore=timestep;
//    #endif
//
//        // start simulating
//        computeDerivatives<<<NUM_BLOCKS(pb.size()),BLOCK_SIZE>>>(pb.createDeviceCopy(),SOUNDSPEED);
//        assert_cuda(cudaGetLastError());
//        integrateLeapfrog<<<NUM_BLOCKS(pb.size()),BLOCK_SIZE>>>(pb.createDeviceCopy(),timestep,false,tanfr);
//        assert_cuda(cudaGetLastError());
//
//        double simulatedTime=timestep;
//    #if defined(READ_FROM_FILE)
//        simulatedTime += startTime;
//    #endif
//
//        pb.unmapGraphicsResource(); // used for frontend stuff
//        while(fnd::handleFrontend(simulatedTime))
//        {
//            if(simShouldRun)
//            {
//                pb.mapGraphicsResource(); // used for frontend stuff
//
//                computeDerivatives<<<NUM_BLOCKS(pb.size()),BLOCK_SIZE>>>(pb.createDeviceCopy(),SOUNDSPEED);
//                assert_cuda(cudaGetLastError());
//                integrateLeapfrog<<<NUM_BLOCKS(pb.size()),BLOCK_SIZE>>>(pb.createDeviceCopy(),timestep,true,tanfr);
//                assert_cuda(cudaGetLastError());
//
//                simulatedTime += timestep;
//
//    #if defined(STORE_RESULTS)
//                timeSinceStore += timestep;
//                if( timeSinceStore >= store_intervall)
//                {
//                    storage.printToFile(pb,simulatedTime);
//                    timeSinceStore-=store_intervall;
//                }
//    #endif
//
//                pb.unmapGraphicsResource(); // used for frontend stuff
//            }
//        }
//
//        return 0;
//    }
