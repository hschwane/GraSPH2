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
#include "initialConditions/particleSources/HDF5File.h"
#include "frontends/frontendInterface.h"
#include "particles/Particles.h"
#include "sph/kernel.h"
#include "sph/eos.h"
#include "sph/models.h"
#include "ResultStorageManager.h"
#include "settings.h"
#include "integration.h"

// compile setting files into resources
ADD_RESOURCE(Settings,"settings.h");
ADD_RESOURCE(HeadlessSettings,"headlessSettings.h");
ADD_RESOURCE(PrecisionSettings,"precisionSettings.h");
ADD_RESOURCE(OutputSettings,"outputSettings.h");

constexpr f1_t H2 = H*H; //!< square of the smoothing length
constexpr f1_t dW_prefactor = kernel::dsplinePrefactor<dimension>(H); //!< spline kernel prefactor
constexpr f1_t W_prefactor = kernel::splinePrefactor<dimension>(H); //!< spline kernel prefactor

/**
 * @brief calculates density using the sph method
 */
struct calcDensity
{
    // define particle attributes to use
    using load_type = Particle<POS,MASS>; //!< particle attributes to load from main memory
    using store_type = Particle<DENSITY>; //!< particle attributes to store to main memory
    using pi_type = merge_particles_t<load_type,store_type>; //!< the type of particle you want to work with in your job functions
    using pj_type = Particle<POS,MASS>; //!< the particle attributes to load from main memory of all the interaction partners j
    //!< when using do_for_each_pair_fast a SharedParticles object must be specified which can store all the attributes of particle j
    template<size_t n> using shared = SharedParticles<n,SHARED_POSM>;

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
            // get the kernel function
            const f1_t r = sqrt(r2);
            const f1_t w = kernel::Wspline(r, H, W_prefactor);

            pi.density += pj.mass * w;
        }
    }

    //!< This function will be called for particle i after the interactions with the other particles are computed.
    CUDAHOSTDEV store_type do_after(pi_type& pi)
    {
#if defined(DEAL_WITH_NO_PARTNERS)
        if(pi.density < 0.001)
            pi.density = rho0*0.01;
#endif
        return pi;
    }
};

/**
 * @brief calculates deriviatives of density, deviatoric stress as well as the balsara switch
 */
struct calcBalsaraDensityDTDStressDT
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
    f3_t curlv{0}; // velocity curl

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
            const f1_t r = sqrt(r2);
            const f1_t dw = kernel::dWspline(r, H, dW_prefactor);
            const f3_t gradw = (dw / r) * rij;

            const f3_t vij = pi.vel - pj.vel;
#if defined(SOLIDS)
            // strain rate tensor (edot) and rotation rate tensor (rdot)
            addStrainRateAndRotationRate(edot,rdot,pj.mass,pj.density,vij,gradw);
#elif defined(BALSARA_SWITCH)
            curlv += pj.mass / pj.density * cross(vij, gradw);
#endif
#if defined(BALSARA_SWITCH) || defined(INTEGRATE_DENSITY)
            divv -= (pj.mass / pj.density) * dot(vij, gradw);
#endif

        }
    }

    //!< This function will be called for particle i after the interactions with the other particles are computed.
    CUDAHOSTDEV store_type do_after(pi_type& pi)
    {
#if defined(SOLIDS)
        // deviatoric stress time derivative
        pi.dstress_dt = dstress_dt(edot,rdot,pi.dstress,shear);
        // get curl from edot and compute the balsara switch value
        curlv = f3_t{-2*rdot[1][2], -2*rdot[2][0], -2*rdot[0][1] };
#endif
#if defined(INTEGRATE_DENSITY)
        // density time derivative
        pi.density_dt = -pi.density * divv;
#endif
#if defined(BALSARA_SWITCH)
        pi.balsara = balsaraSwitch(divv, curlv, SOUNDSPEED, H);
#endif

        return pi;
    }
};

/**
 * @brief second pass of derivative computation
 */
struct calcAcceleration
{
    // define particle attributes to use
    using load_type = Particle<POS,MASS,VEL,BALSARA,DENSITY,DSTRESS>; //!< particle attributes to load from main memory
    using store_type = Particle<ACC,XVEL,MAXVSIG>; //!< particle attributes to store to main memory
    using pi_type = merge_particles_t<load_type,store_type>; //!< the type of particle you want to work with in your job functions
    using pj_type = Particle<POS,MASS,VEL,BALSARA,DENSITY,DSTRESS>; //!< the particle attributes to load from main memory of all the interaction partners j
    //!< when using do_for_each_pair_fast a SharedParticles object must be specified which can store all the attributes of particle j
    template<size_t n> using shared = SharedParticles<n,SHARED_POSM,SHARED_VEL,SHARED_BALSARA,SHARED_DENSITY,SHARED_DSTRESS>;

    // setup some variables we need before during and after the pair interactions
#if defined(ENABLE_SPH)

    #if defined(SOLIDS)
        using stress_t = m3_t;
    #else
        using stress_t = f1_t;
    #endif

    stress_t sigOverRho_i; // stress over density square used for acceleration
    #if defined(ARTIFICIAL_STRESS)
        stress_t arts_i; // artificial stress from i
    #endif
#endif

    //!< This function is executed for each particle before the interactions are computed.
    CUDAHOSTDEV void do_before(pi_type& pi, size_t id)
    {
#if defined(ENABLE_SPH)
        // build stress tensor for particle i using deviatoric stress and pressure
        stress_t sigma_i;
    #if defined(SOLIDS)
        sigma_i = pi.dstress;
        const f1_t pres_i = eos::murnaghan( pi.density, rho0, BULK, dBULKdP);
        sigma_i[0][0] -= pres_i;
        sigma_i[1][1] -= pres_i;
        sigma_i[2][2] -= pres_i;
    #else
        sigma_i =  -eos::liquid( pi.density, rho0, SOUNDSPEED*SOUNDSPEED);
    #endif

        sigOverRho_i = sigma_i / (pi.density*pi.density);

    #if defined(ARTIFICIAL_STRESS)
        // artificial stress from i
        #if defined(SOLIDS)
            arts_i = artificialStress(mateps, sigOverRho_i);
        #else
            arts_i = artificialPressure(mateps, sigOverRho_i);
        #endif
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
                stress_t sigma_j;
    #if defined(SOLIDS)
                sigma_j = pj.dstress;
                const f1_t pres_j = eos::murnaghan(pj.density, rho0, BULK, dBULKdP);
                sigma_j[0][0] -= pres_j;
                sigma_j[1][1] -= pres_j;
                sigma_j[2][2] -= pres_j;
    #else
                sigma_j = -eos::liquid( pj.density, rho0, SOUNDSPEED*SOUNDSPEED);
    #endif

                stress_t sigOverRho_j = sigma_j / (pj.density * pj.density);

                // stress from the interaction
                stress_t stress = sigOverRho_i + sigOverRho_j;

                const f3_t vij = pi.vel - pj.vel;
    #if defined(ARTIFICIAL_STRESS)
                // artificial stress
                const f1_t f = pow(kernel::Wspline(r, H, W_prefactor) / kernel::Wspline(normalsep, H, W_prefactor) , matexp);
        #if defined(SOLIDS)
                stress_t arts_j = artificialStress(mateps, sigOverRho_j)
        #else
                stress_t arts_j = artificialPressure(mateps, sigOverRho_j);
        #endif
                stress += f*(arts_i + arts_j);
    #endif

                // acceleration from stress
                pi.acc += pj.mass * (stress * gradw);

    #if defined(ARTIFICIAL_VISCOSITY)
                // acceleration from artificial viscosity
                pi.acc -= pj.mass *
                          artificialViscosity(
        #if defined(BALSARA_SWITCH)
                                  pi.max_vsig,
        #endif
                                  alpha, pi.density, pj.density, vij, rij, r, SOUNDSPEED, SOUNDSPEED
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

#if defined(VARIABLE_TIMESTEP_LEAPFROG)
        if(threadIdx.x == 0 && blockIdx.x == 0)
            nextTS = max_timestep;
#endif

        return pi;
    }
};

template <typename pbT>
void computeDerivatives(pbT& particleBuffer)
{
#if defined(ENABLE_SPH)
    #if !defined(INTEGRATE_DENSITY)
        do_for_each_pair_fast<calcDensity>(particleBuffer);
    #endif
    #if defined(SOLIDS) || defined(INTEGRATE_DENSITY) || defined(BALSARA_SWITCH)
        do_for_each_pair_fast<calcBalsaraDensityDTDStressDT>(particleBuffer);
    #endif
#endif

    do_for_each_pair_fast<calcAcceleration>(particleBuffer);
}

/**
 * @brief decides if file is hdf5 or text file and adds it to a particle generator
 * @tparam LoadParticleType The type of particles expected in the file
 * @param filename the path to the file to load
 * @param generator reference to the generator where the file will be added to
 * @param textfileSeperator in case of text file this is the seperator used
 */
template<typename LoadParticleType, typename GeneratorType>
void loadParticlesFromFile(const std::string& filename, GeneratorType& generator, char textfileSeperator = '\t')
{
    auto fileEnding = filename.substr(filename.find_last_of('.')+1);
    if(fileEnding == "h5" || fileEnding == "hdf5")
    {
        logINFO("InitialConditions") << "Input file was detected to be a hdf5 file.";
        generator.addParticles(ps::HDF5File<LoadParticleType>(filename));
    }
    else if(fileEnding == "tsv")
    {
        logINFO("InitialConditions") << "Input file treated as tsv file.";
        generator.addParticles(ps::TextFile<LoadParticleType>(filename,'\t'));
    }
    else if(fileEnding == "csv")
    {
        logINFO("InitialConditions") << "Input file treated as csv file.";
        generator.addParticles(ps::TextFile<LoadParticleType>(filename,','));
    }
    else
    {
        logINFO("InitialConditions") << "Input file treated as text file.";
        generator.addParticles(ps::TextFile<LoadParticleType>(filename,textfileSeperator));
    }
}

void printSettings(mpu::Log& log)
{
    std::string buildType;
#if defined(NDEBUG)
    buildType = "Release";
#else
    buildType = "Debug";
#endif

    log.printHeader("GraSPH2",GRASPH_VERSION,GRASPH_VERSION_SHA,buildType);
    logINFO("GraSPH2") << "Welcome to GraSPH2!";
#if defined(SINGLE_PRECISION)
    logINFO("GraSPH2") << "Running in single precision mode.";
#elif defined(DOUBLE_PRECISION)
    logINFO("GraSPH2") << "Running in double precision mode.";
#endif
#if defined(USING_CUDA_FAST_MATH)
    logWARNING("GraSPH2") << "Unsafe math optimizations enabled in CUDA code.";
#endif
    assert_cuda(cudaSetDevice(0));

    // print some important settings to the console
    log.print(mpu::LogLvl::INFO) << "\nSettings for this run:\n========================\n"
                                   << "Integration:"
                                   << "Leapfrog"
                                   #if defined(FIXED_TIMESTEP_LEAPFROG)
                                   << "Timestep: constant, " << fixed_timestep << "\n"
                                   #elif defined(VARIABLE_TIMESTEP_LEAPFROG)
                                   << "Timestep: variable \n"
                                   #endif
                                   << "Initial Conditions:\n"
                                   #if defined(READ_FROM_FILE)
                                   << "Data is read from: " << FILENAME << "\n"
                                   #elif defined(ROTATING_UNIFORM_SPHERE)
                                   << "Using a random uniform sphere with radius " << spawn_radius << "\n"
                                   << "Total mass: " << tmass << "\n"
                                   << "Number of particles: " << particle_count << "\n"
                                   << "additional angular velocity: " << angVel << "\n"
                                   #elif defined(ROTATING_PLUMMER_SPHERE)
                                   << "Using a Plummer distribution with core radius " << plummer_radius << " and cutoff " << plummer_cutoff << "\n"
                        << "Total mass: " << tmass << "\n"
                        << "Number of particles: " << particle_count << "\n"
                        << "additional angular velocity: " << angVel << "\n"
                                   #endif
                                   << "Compressed radius set to " << compressesd_radius << "\n"
                                   << "resulting in particle radius of " << pradius << "\n"
                                   << "and smoothing length " << H << "\n"
                                   << "Material Settings:\n"
                                   << "material density: " << rho0 << "\n"
                                   << "speed of sound: " << SOUNDSPEED << "\n"
                                   << "bulk-modulus: " << BULK << "\n"
                                   << "shear-modulus: " << shear << "\n"
                                   << "Environment Settings:\n"
                                   #if defined(CLOHESSY_WILTSHIRE)
                                   << "Clohessy-Wiltshire enabled with n = " << cw_n << "\n";
                                   #else
                                   << "Clohessy-Wiltshire disabled" << "\n"
                                   #endif
            ;
}

/**
 * @brief The main function of the simulation. Sets up the initial conditions and frontend and then manages running the simulation.
 *
 */
int main()
{
    mpu::Log myLog( mpu::LogLvl::ALL, mpu::ConsoleSink());

#if defined(STORE_RESULTS)
    // set up file saving engine
    ResultStorageManager storage(RESULT_FOLDER,RESULT_PREFIX,maxJobs);
    // setup log output file
    myLog.addSinks(mpu::FileSink( std::string(RESULT_FOLDER) + std::string(RESULT_PREFIX) + storage.getStartTime() + "_log.txt"));
    // collect all settings and print them into a file
    {
        mpu::Resource headlessSettings = LOAD_RESOURCE(HeadlessSettings);
        mpu::Resource precisionSettings = LOAD_RESOURCE(PrecisionSettings);
        mpu::Resource outputSettings = LOAD_RESOURCE(OutputSettings);
        mpu::Resource settings = LOAD_RESOURCE(Settings);
        std::ofstream settingsOutput(std::string(RESULT_FOLDER) + std::string(RESULT_PREFIX) + storage.getStartTime() + "_settings.txt");
        settingsOutput << "//////////////////////////\n// headlessSettigns.h \n//////////////////////////\n\n"
                        << std::string(headlessSettings.data(), headlessSettings.size())
                        << "\n\n\n//////////////////////////\n// precisionSettings.h \n//////////////////////////\n\n"
                        << std::string(precisionSettings.data(), precisionSettings.size())
                        << "\n\n\n//////////////////////////\n// outputSettings.h \n//////////////////////////\n\n"
                        << std::string(outputSettings.data(), outputSettings.size())
                        << "\n\n\n//////////////////////////\n// settigns.h \n//////////////////////////\n\n"
                        << std::string(settings.data(), settings.size());
    }
#endif

    printSettings(myLog);

    // set up frontend
    fnd::initializeFrontend();
    bool simShouldRun = false;
    fnd::setPauseHandler([&simShouldRun](bool pause){simShouldRun = !pause;});

    // generate some particles depending on options in the settings file
    InitGenerator<HostParticlesType> generator;

#if defined(READ_FROM_FILE)
    loadParticlesFromFile<particleToRead>(FILENAME,generator,SEPERATOR);
#elif defined(ROTATING_UNIFORM_SPHERE)
    generator.addParticles( ps::UniformSphere(particle_count,spawn_radius,tmass,rho0).addAngularVelocity(angVel), true,true );
#elif defined(ROTATING_PLUMMER_SPHERE)
    generator.addParticles( ps::PlummerSphere(particle_count,plummer_radius,plummer_cutoff,tmass,rho0).addAngularVelocity(angVel), true, true);
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

    // when file is dropped on window load it
    fnd::setDropHandler([&](const std::string& filename)
    {
        InitGenerator<HostParticlesType> newGenerator;
        loadParticlesFromFile<particleToRead>(filename,newGenerator,SEPERATOR);
        hpb = newGenerator.generate();
        pb = DeviceParticlesType(hpb.size());
        pb.registerGLGraphicsResource<DEV_POSM>(fnd::getPositionBuffer(pb.size()));
        pb.registerGLGraphicsResource<DEV_VEL>(fnd::getVelocityBuffer(pb.size()));
        pb.registerGLGraphicsResource<DEV_DENSITY>(fnd::getDensityBuffer(pb.size()));
        pb.mapGraphicsResource();
        pb = hpb;
    });
#endif

    // upload particles
    pb = hpb;

    // get current timestep from integrator
    float timestep = getCurrentTimestep();

#if defined(STORE_RESULTS)
    // print timestep 0
    storage.printToFile(pb,0);
    f1_t timeSinceStore=timestep;
#endif

    // start simulating
    computeDerivatives(pb);
    integrate(pb,false);

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
            integrate(pb,true);

            timestep = getCurrentTimestep();
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
