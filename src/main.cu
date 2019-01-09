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


/**
 * @brief computes derivatives of particle attributes
 * @param particles device copy of the particle Buffer
 * @param speedOfSound your materials sound speed
 */
struct computeDerivatives
{
    // define particle attributes to use
    using load_type = Particle<POS,MASS,VEL,DENSITY,DSTRESS>; //!< particle attributes to load from main memory
    using store_type = Particle<ACC,XVEL,DENSITY_DT,DSTRESS_DT>; //!< particle attributes to store to main memory
    using pi_type = merge_particles_t<load_type,store_type>; //!< the type of particle you want to work with in your job functions
    using pj_type = Particle<POS,MASS,VEL,DENSITY,DSTRESS>; //!< the particle attributes to load from main memory of all the interaction partners j
    //!< when using do_for_each_pair_fast a SharedParticles object must be specified which can store all the attributes of particle j
    template<size_t n>
    using shared = SharedParticles<n,SHARED_POSM,SHARED_VEL,SHARED_DENSITY,SHARED_DSTRESS>;

    // setup some variables we need before during and after the pair interactions
#if defined(ENABLE_SPH)
    m3_t sigOverRho_i; // stress over density square used for acceleration
    m3_t sigma_i; // stress tensor
    m3_t edot{0}; // strain rate tensor (edot)
    m3_t rdot{0}; // rotation rate tensor
    f1_t vdiv{0}; // velocity divergence
    #if defined(ARTIFICIAL_STRESS)
        m3_t arts_i; // artificial stress from i
    #endif
#endif

    //!< This function is executed for each particle before the interactions are computed.
    CUDAHOSTDEV void do_before(pi_type& pi, size_t id, f1_t speedOfSound)
    {
#if defined(ENABLE_SPH)
        // build stress tensor for particle i using deviatoric stress and pressure
        sigma_i = pi.dstress;
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
    CUDAHOSTDEV void do_for_each_pair(pi_type& pi, const pj_type pj, f1_t speedOfSound)
    {
        // code run for each pair of particles

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
                const f1_t dw = kernel::dWspline<dimension>(r, H);
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
    #if defined(ARTIFICIAL_VISCOSITY)
                // acceleration from artificial viscosity
                pi.acc -= pj.mass *
                          artificialViscosity(alpha, pi.density, pj.density, vij, rij, r, speedOfSound, speedOfSound) *
                          gradw;
    #endif

    #if defined(ARTIFICIAL_STRESS)
                // artificial stress
                const f1_t f = pow(kernel::Wspline<dimension>(r, H) / kernel::Wspline<dimension>(normalsep, H) , matexp;
                stress += f*(arts_i + artificialStress(mateps, sigOverRho_j));
    #endif

                // acceleration from stress
                pi.acc += pj.mass * (stress * gradw);

                // strain rate tensor (edot) and rotation rate tensor (rdot)
                addStrainRateAndRotationRate(edot,rdot,pj.mass,pi.mass,vij,gradw);

                // density time derivative
                vdiv += (pj.mass / pj.density) * dot(vij, gradw);

    #if defined(XSPH)
                // xsph
                pi.xvel += 2 * pj.mass / (pi.density + pj.density) * (pj.vel - pi.vel) * kernel::Wspline<dimension>(r, H);
    #endif
            }
#endif // ENABLE_SPH
        }
    }

    //!< This function will be called for particle i after the interactions with the other particles are computed.
    CUDAHOSTDEV store_type do_after(pi_type& pi, f1_t speedOfSound)
    {
#if defined(ENABLE_SPH)
        // density time derivative
        pi.density_dt = pi.density * vdiv;
        // deviatoric stress time derivative
        pi.dstress_dt = dstress_dt(edot,rdot,pi.dstress,shear);
#endif
        return pi;
    }
};

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
    CUDAHOSTDEV store_type do_for_each(pi_type p, size_t id, f1_t dt, bool not_first_step, f1_t tanfr)
    {
        //   calculate velocity a_t
        p.vel = p.vel + p.acc * (dt * 0.5f);

        // we could now change delta t here

        // calculate velocity a_t+1/2
        p.vel = p.vel + p.acc * (dt * 0.5f) * not_first_step;

        // calculate position r_t+1
#if defined(XSPH) && defined(ENABLE_SPH)
        p.pos = p.pos + (p.vel + xsph_factor*p.xvel) * dt;
#else
        p.pos = p.pos + p.vel * dt;
#endif

#if defined(ENABLE_SPH)
        // density
        p.density = p.density + p.density_dt * dt;
        if(p.density < 0.0f)
            p.density = 0.0f;

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

    myLog.printHeader("GraSPH2",GRASPH_VERSION,GRASPH_VERSION_SHA,buildType);
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

    // create cuda buffer
    DeviceParticlesType pb(hpb.size());
#if defined(FRONTEND_OPENGL)
    pb.registerGLGraphicsResource<DEV_POSM>(fnd::getPositionBuffer(pb.size()));
    pb.registerGLGraphicsResource<DEV_VEL>(fnd::getVelocityBuffer(pb.size()));
    pb.registerGLGraphicsResource<DEV_DENSITY>(fnd::getDensityBuffer(pb.size()));
    pb.mapGraphicsResource();
#endif

    // upload particles
    pb = hpb;

    // calculate sound speed and tangents of friction angle
    const f1_t SOUNDSPEED = sqrt(BULK / rho0);
    const f1_t tanfr = tan(friction_angle);

#if defined(STORE_RESULTS)
    // set up file saving engine
    ResultStorageManager storage(RESULT_FOLDER,RESULT_PREFIX,maxJobs);
    storage.printToFile(pb,0);
    f1_t timeSinceStore=timestep;
#endif

    // start simulating
    do_for_each_pair_fast<computeDerivatives>(pb,SOUNDSPEED);
    do_for_each<integrateLeapfrog>(pb,timestep,false,tanfr);

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
            do_for_each_pair_fast<computeDerivatives>(pb,SOUNDSPEED);
            do_for_each<integrateLeapfrog>(pb,timestep,true,tanfr);

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
