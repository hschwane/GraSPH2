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
#include <initialConditions/particleSources/PlummerSphere.h>

#include "initialConditions/InitGenerator.h"
#include "initialConditions/particleSources/UniformSphere.h"
#include "initialConditions/particleSources/TextFile.h"
#include "frontends/frontendInterface.h"
#include "particles/Particles.h"
#include "sph/kernel.h"
#include "sph/eos.h"
#include "sph/models.h"
#include "ResultStorageManager.h"
#include "settings/settings.h"


constexpr f1_t H2 = H*H; //!< square of the smoothing length
constexpr f1_t dW_prefactor = kernel::dsplinePrefactor<dimension>(H); //!< spline kernel prefactor
constexpr f1_t W_prefactor = kernel::splinePrefactor<dimension>(H); //!< spline kernel prefactor

/**
 * @brief first pass of derivative computation
 */
struct cdA
{
    // define particle attributes to use
    using load_type = Particle<POS,MASS,VEL,DENSITY,DSTRESS>; //!< particle attributes to load from main memory
    using store_type = Particle<BALSARA,DENSITY_DT,DSTRESS_DT>; //!< particle attributes to store to main memory
    using pi_type = merge_particles_t<load_type,store_type>; //!< the type of particle you want to work with in your job functions
    using pj_type = Particle<POS,MASS,VEL,DENSITY>; //!< the particle attributes to load from main memory of all the interaction partners j
    //!< when using do_for_each_pair_fast a SharedParticles object must be specified which can store all the attributes of particle j
    template<size_t n> using shared = SharedParticles<n,SHARED_POSM,SHARED_VEL,SHARED_DENSITY>;

    // setup some variables we need before during and after the pair interactions
    m3_t edot{0}; // strain rate tensor (edot)
    m3_t rdot{0}; // rotation rate tensor
    f1_t divv{0}; // velocity divergence

    //!< This function is executed for each particle before the interactions are computed.
    CUDAHOSTDEV void do_before(pi_type& pi, size_t id)
    {
    }

    //!< This function will be called for each pair of particles.
    CUDAHOSTDEV void do_for_each_pair(pi_type& pi, const pj_type pj)
    {
        const f3_t rij = pi.pos-pj.pos;
        const f1_t r2 = dot(rij,rij);
        if(r2 <= H2 && r2>0)
        {
            // get the kernel gradient
            f1_t r = sqrt(r2);
            const f1_t dw = kernel::dWspline(r, H, dW_prefactor);
            const f3_t gradw = (dw / r) * rij;

            // strain rate tensor (edot) and rotation rate tensor (rdot)
            const f3_t vij = pi.vel - pj.vel;
            addStrainRateAndRotationRate(edot,rdot,pj.mass,pj.density,vij,gradw);
            divv -= (pj.mass / pj.density) * dot(vij, gradw);
        }
    }

    //!< This function will be called for particle i after the interactions with the other particles are computed.
    CUDAHOSTDEV store_type do_after(pi_type& pi)
    {
        // deviatoric stress time derivative
        pi.dstress_dt = dstress_dt(edot,rdot,pi.dstress,shear);

        // density time derivative
        pi.density_dt = -pi.density * divv;

#if defined(BALSARA_SWITCH)
        // get curl from edot and compute the balsara switch value
        const f3_t curlv = f3_t{-2*rdot[1][2], -2*rdot[2][0], -2*rdot[0][1] };
        pi.balsara = balsaraSwitch(divv, curlv, SOUNDSPEED, H);
#endif

        return pi;
    }
};

/**
 * @brief second pass of derivative computation
 */
struct cdB
{
    // define particle attributes to use
    using load_type = Particle<POS,MASS,VEL,BALSARA,DENSITY,DSTRESS>; //!< particle attributes to load from main memory
    using store_type = Particle<ACC,XVEL>; //!< particle attributes to store to main memory
    using pi_type = merge_particles_t<load_type,store_type>; //!< the type of particle you want to work with in your job functions
    using pj_type = Particle<POS,MASS,VEL,BALSARA,DENSITY,DSTRESS>; //!< the particle attributes to load from main memory of all the interaction partners j
    //!< when using do_for_each_pair_fast a SharedParticles object must be specified which can store all the attributes of particle j
    template<size_t n> using shared = SharedParticles<n,SHARED_POSM,SHARED_VEL,SHARED_BALSARA,SHARED_DENSITY,SHARED_DSTRESS>;

    // setup some variables we need before during and after the pair interactions
#if defined(ENABLE_SPH)
    m3_t sigOverRho_i; // stress over density square used for acceleration
    #if defined(ARTIFICIAL_STRESS)
        m3_t arts_i; // artificial stress from i
    #endif
#endif

    //!< This function is executed for each particle before the interactions are computed.
    CUDAHOSTDEV void do_before(pi_type& pi, size_t id)
    {
#if defined(ENABLE_SPH)
        // build stress tensor for particle i using deviatoric stress and pressure
        m3_t sigma_i = pi.dstress;
        const f1_t pres_i = eos::murnaghan( pi.density, rho0, BULK, dBULKdP);
        sigma_i[0][0] -= pres_i;
        sigma_i[1][1] -= pres_i;
        sigma_i[2][2] -= pres_i;

        sigOverRho_i = sigma_i / (pi.density*pi.density);

    #if defined(ARTIFICIAL_STRESS)
        // artificial stress from i
        m3_t arts_i = artificialStress(mateps, sigOverRho_i);
    #endif
#endif
    }

    //!< This function will be called for each pair of particles.
    CUDAHOSTDEV void do_for_each_pair(pi_type& pi, const pj_type pj)
    {
        const f3_t rij = pi.pos-pj.pos;
        const f1_t r2 = dot(rij,rij);
        if(r2>0)
        {

#if defined(ENABLE_SELF_GRAVITY)
            // gravity
            const f1_t distSqr = r2 + H2;
            const f1_t invDist = rsqrt(distSqr);
            const f1_t invDistCube = invDist * invDist * invDist;
            pi.acc -= rij * pj.mass * invDistCube;
#endif

#if defined(ENABLE_SPH)
            if(r2 <= H2)
            {
                // get the kernel gradient
                f1_t r = sqrt(r2);
                const f1_t dw = kernel::dWspline(r, H, dW_prefactor);
                const f3_t gradw = (dw / r) * rij;

                // stress and pressure of j
                m3_t sigma_j = pj.dstress;
                const f1_t pres_j = eos::murnaghan(pj.density, rho0, BULK, dBULKdP);
                sigma_j[0][0] -= pres_j;
                sigma_j[1][1] -= pres_j;
                sigma_j[2][2] -= pres_j;

                m3_t sigOverRho_j = sigma_j / (pj.density * pj.density);

                // stress from the interaction
                m3_t stress = sigOverRho_i + sigOverRho_j;

                const f3_t vij = pi.vel - pj.vel;
    #if defined(ARTIFICIAL_STRESS)
                // artificial stress
                const f1_t f = pow(kernel::Wspline(r, H, W_prefactor) / kernel::Wspline(normalsep, H, W_prefactor) , matexp;
                stress += f*(arts_i + artificialStress(mateps, sigOverRho_j));
    #endif

                // acceleration from stress
                pi.acc += pj.mass * (stress * gradw);

    #if defined(ARTIFICIAL_VISCOSITY)
                // acceleration from artificial viscosity
                pi.acc -= pj.mass *
                          artificialViscosity(alpha, pi.density, pj.density, vij, rij, r, SOUNDSPEED, SOUNDSPEED
        #if defined(BALSARA_SWITCH)
                                  , pi.balsara, pj.balsara
        #endif
                          ) * gradw;
    #endif

    #if defined(XSPH)
                // xsph
                pi.xvel += 2 * pj.mass / (pi.density + pj.density) * (pj.vel - pi.vel) * kernel::Wspline<dimension>(r, H);
    #endif
            }
#endif // ENABLE_SPH
        }
    }

    //!< This function will be called for particle i after the interactions with the other particles are computed.
    CUDAHOSTDEV store_type do_after(pi_type& pi)
    {
#if defined(CLOHESSY_WILTSHIRE)
        pi.acc.x += 3*cw_n*cw_n * pi.pos.x + 2*cw_n* pi.vel.y;
        pi.acc.y += -2*cw_n * pi.vel.x;
        pi.acc.z += -cw_n*cw_n * pi.pos.z;
#endif
        return pi;
    }
};

template <typename pbT>
void computeDerivatives(pbT& particleBuffer)
{
#if defined(ENABLE_SPH)
    do_for_each_pair_fast<cdA>(particleBuffer);
#endif
    do_for_each_pair_fast<cdB>(particleBuffer);
}

/**
 * @brief perform leapfrog integration on the particles also performs the plasticity calculations
 * @param particles the device copy to the particle buffer that stores the particles
 * @param dt the timestep for the integration
 * @param not_first_step set false for the first integration step of the simulation
 * @param tanfr tangens of the internal friction angle
 */
struct integrateLeapfrog
{
    using load_type = Particle<POS,VEL,ACC,XVEL,DENSITY,DENSITY_DT,DSTRESS,DSTRESS_DT>; //!< particle attributes to load from main memory
    using store_type = Particle<POS,VEL,DENSITY,DSTRESS>; //!< particle attributes to store to main memory
    using pi_type = merge_particles_t<load_type,store_type>; //!< the type of particle you want to work with in your job functions

    //!< This function is executed for each particle. In p the current particle and in id its position in the buffer is given.
    //!< All attributes of p that are not in load_type will be initialized to some default (mostly zero)
    CUDAHOSTDEV store_type do_for_each(pi_type p, size_t id, f1_t dt, bool not_first_step)
    {
        //   calculate velocity a_t
        p.vel = p.vel + p.acc * (dt * 0.5_ft);

        // we could now change delta t here

        // calculate velocity a_t+1/2
        p.vel = p.vel + p.acc * (dt * 0.5_ft) * not_first_step;

        // calculate position r_t+1
#if defined(XSPH) && defined(ENABLE_SPH)
        p.pos = p.pos + (p.vel + xsph_factor*p.xvel) * dt;
#else
        p.pos = p.pos + p.vel * dt;
#endif

#if defined(ENABLE_SPH)
        // density
        p.density = p.density + p.density_dt * dt;
        if(p.density < 0.0_ft)
            p.density = 0.0_ft;

        // deviatoric stress
        p.dstress += p.dstress_dt * dt;

    #if defined(PLASTICITY_MC)
        plasticity(p.dstress, mohrCoulombYieldStress( tanfr,eos::murnaghan(p.density,rho0, BULK, dBULKdP),cohesion));
    #elif defined(PLASTICITY_MIESE)
        plasticity(p.dstress,Y);
    #endif

#endif
        return p; //!< return particle p, all attributes it shares with load_type will be stored in memory
    }
};

/**
 * @brief The main function of the simulation. Sets up the initial conditions and frontend and then manages running the simulation.
 *
 */
int main()
{
    mpu::Log myLog( mpu::LogLvl::ALL, mpu::ConsoleSink());

    std::string buildType;
#if defined(NDEBUG)
    buildType = "Release";
#else
    buildType = "Debug";
#endif

#if defined(STORE_RESULTS)
    // set up file saving engine
    ResultStorageManager storage(RESULT_FOLDER,RESULT_PREFIX,maxJobs);
    // setup log output file
    myLog.addSinks(mpu::FileSink( std::string(RESULT_FOLDER) + std::string(RESULT_PREFIX) + storage.getStartTime() + "_log.txt"));
    // collect all settings and print them into a file
    {
        mpu::Resource headlessSettings = LOAD_RESOURCE(src_settings_headlessSettings_h);
        mpu::Resource precisionSettings = LOAD_RESOURCE(src_settings_precisionSettings_h);
        mpu::Resource settings = LOAD_RESOURCE(src_settings_settings_h);
        std::ofstream settingsOutput(std::string(RESULT_FOLDER) + std::string(RESULT_PREFIX) + storage.getStartTime() + "_settings.txt");
        settingsOutput << "//////////////////////////\n// headlessSettigns.h \n//////////////////////////\n\n"
                        << std::string(headlessSettings.data(), headlessSettings.size())
                        << "\n\n\n//////////////////////////\n// precisionSettings.h \n//////////////////////////\n\n"
                        << std::string(precisionSettings.data(), precisionSettings.size())
                        << "\n\n\n//////////////////////////\n// settigns.h \n//////////////////////////\n\n"
                        << std::string(settings.data(), settings.size());
    }
#endif

    myLog.printHeader("GraSPH2",GRASPH_VERSION,GRASPH_VERSION_SHA,buildType);
    logINFO("GraSPH2") << "Welcome to GraSPH2!";
    assert_cuda(cudaSetDevice(0));

    logINFO("Settings") << "speed of sound: " << SOUNDSPEED << "\n"
                        << "pradius: " << pradius << "\n"
                        << "H sqared: " << H2 << "\n"
                        ;

    // set up frontend
    fnd::initializeFrontend();
    bool simShouldRun = false;
    fnd::setPauseHandler([&simShouldRun](bool pause){simShouldRun = !pause;});

    // generate some particles depending on options in the settings file
    InitGenerator<HostParticlesType> generator;

#if defined(READ_FROM_FILE)
    generator.addParticles(ps::TextFile<particleToRead>(FILENAME,SEPERATOR));
#elif defined(ROTATING_UNIFORM_SPHERE)
    generator.addParticles( ps::UniformSphere(particle_count,1.0_ft,tmass,rho0,161214).addAngularVelocity(angVel), true,true );
#elif defined(ROTATING_PLUMMER_SPHERE)
    generator.addParticles( ps::PlummerSphere(particle_count,1.0,plummerCutoff,tmass,rho0,161214).addAngularVelocity(angVel), true, true);
#endif

    auto hpb = generator.generate();

    // create cuda buffer
    DeviceParticlesType pb(hpb.size());
#if defined(FRONTEND_OPENGL)
    fnd::setParticleSize(pradius);
    pb.registerGLGraphicsResource<DEV_POSM>(fnd::getPositionBuffer(pb.size()));
    pb.registerGLGraphicsResource<DEV_VEL>(fnd::getVelocityBuffer(pb.size()));
    pb.registerGLGraphicsResource<DEV_DENSITY>(fnd::getDensityBuffer(pb.size()));
    pb.mapGraphicsResource();
#endif

    // upload particles
    pb = hpb;

#if defined(STORE_RESULTS)
    // print timestep 0
    storage.printToFile(pb,0);
    f1_t timeSinceStore=timestep;
#endif

    // start simulating
    computeDerivatives(pb);
    do_for_each<integrateLeapfrog>(pb,timestep,false);

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

            // run simulation
            computeDerivatives(pb);
            do_for_each<integrateLeapfrog>(pb,timestep,true);

            simulatedTime += timestep;

#if defined(STORE_RESULTS)
            timeSinceStore += timestep;
            if( timeSinceStore >= store_intervall)
            {
                storage.printToFile(pb,simulatedTime);
                timeSinceStore-=store_intervall;
            }
#endif

            pb.unmapGraphicsResource(); // used for frontend stuff
        }
    }

    return 0;
}
