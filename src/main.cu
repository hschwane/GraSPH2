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
#include <initialConditions/particleSources/PlummerSphere.h>

#include "initialConditions/InitGenerator.h"
#include "initialConditions/particleSources/UniformSphere.h"
#include "initialConditions/particleSources/TextFile.h"
#include "initialConditions/particleSources/HDF5File.h"
#include "frontends/frontendInterface.h"
#include "particles/Particles.h"
#include "ResultStorageManager.h"
#include "settings.h"
#include "integration.h"
#include "computeDerivatives.h"

// compile setting files into resources
ADD_RESOURCE(Settings,"settings.h");
ADD_RESOURCE(HeadlessSettings,"headlessSettings.h");
ADD_RESOURCE(PrecisionSettings,"precisionSettings.h");
ADD_RESOURCE(OutputSettings,"outputSettings.h");

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


                                   # if defined (ENABLE_SELF_GRAVITY)
                                   << "Self Gravity: enabled\n"
                                   #else
                                   << "Self Gravity: disabledņ\n"
                                   #endif
                                   # if defined (ENABLE_SPH)
                                   << "SPH: enabled "
                                       #if defined(INTEGRATE_DENSITY)
                                       << "with density integration "
                                       #else
                                       << "with density summation "
                                       #endif
                                       #if defined(SOLIDS)
                                       << " and stress tensor support "
                                       #endif
                                   << "\n"
                                   #else
                                   << "SPH: disabledņ\n"
                                   #endif
                                    << "Integration:"
                                   << "Leapfrog\n"
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
