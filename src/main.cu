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
#include <bitset>

#include "initialConditions/InitGenerator.h"
#include "initialConditions/particleSources/UniformSphere.h"
#include "initialConditions/particleSources/TextFile.h"
#include "frontends/frontendInterface.h"
#include "particles/Particles.h"
#include "sph/kernel.h"
#include "sph/eos.h"
#include "sph/models.h"
#include "ResultStorageManager.h"
#include "settings.h"

// compile setting files into resources
ADD_RESOURCE(Settings,"settings.h");
ADD_RESOURCE(HeadlessSettings,"headlessSettings.h");
ADD_RESOURCE(PrecisionSettings,"precisionSettings.h");

constexpr f1_t H2 = H*H; //!< square of the smoothing length
constexpr f1_t dW_prefactor = kernel::dsplinePrefactor<dimension>(H); //!< spline kernel prefactor
constexpr f1_t W_prefactor = kernel::splinePrefactor<dimension>(H); //!< spline kernel prefactor



using spaceKey = unsigned long long int;
/**
 * @brief Expands a 21-bit integer into 64 bits by inserting 2 zeros after each bit.
 *          https://stackoverflow.com/questions/1024754/how-to-compute-a-3d-morton-number-interleave-the-bits-of-3-ints
 * @param x integer to expand
 * @return expanded integer
 */
CUDAHOSTDEV unsigned long long int expandBits( unsigned long long int x)
{
    x = (x | x << 32u) & 0x1f00000000ffffu;
    x = (x | x << 16u) & 0x1f0000ff0000ffu;
    x = (x | x << 8u) & 0x100f00f00f00f00fu;
    x = (x | x << 4u) & 0x10c30c30c30c30c3u;
    x = (x | x << 2u) & 0x1249249249249249u;
    return x;
}

/**
 * @brief Calculates a 64-bit Morton code for the given 3D point located within the unit cube [0,1].
 *          https://devblogs.nvidia.com/thinking-parallel-part-iii-tree-construction-gpu/ on april 29 2019
 * @param x first spacial coordinate
 * @param y second spacial coordinate
 * @param z third spacial coordinate
 * @return resulting 30 bit morton code
 */
CUDAHOSTDEV spaceKey mortonKey(f1_t x, f1_t y, f1_t z)
{
    // float in [0,1] to 21 bit integer
    x = min(max(x * 2'097'151.0_ft, 0.0_ft), 2'097'151.0_ft);
    y = min(max(y * 2'097'151.0_ft, 0.0_ft), 2'097'151.0_ft);
    z = min(max(z * 2'097'151.0_ft, 0.0_ft), 2'097'151.0_ft);
    spaceKey xx = expandBits(static_cast<spaceKey>(x));
    spaceKey yy = expandBits(static_cast<spaceKey>(y));
    spaceKey zz = expandBits(static_cast<spaceKey>(z));
    return xx | (yy << 1u) | (zz << 2u);
}

CUDAHOSTDEV spaceKey calculatePositionKey(const f3_t& pos, const f3_t& domainMin, const f3_t& domainFactor)
{
    const f3_t normalizedPos = (pos - domainMin) * domainFactor;
    return mortonKey(normalizedPos.x,normalizedPos.y,normalizedPos.z);
}

struct Node
{
    bool isLeaf{false};
    virtual ~Node() = default;
};

struct LNode : public Node
{
    explicit LNode(int i)
    {
        id = i;
        isLeaf=true;
    }

    int id;
};

struct INode : public Node
{
    INode(std::shared_ptr<Node> l, std::shared_ptr<Node> r)
    {
        left = std::move(l);
        right = std::move(r);
    }
    std::shared_ptr<Node> left;
    std::shared_ptr<Node> right;
};

void printTree(Node* root)
{
    int level = 0;

    std::queue<std::pair<Node*,bool>> queue;
    queue.push({root,false});
    queue.push({nullptr,false});

    while(!queue.empty())
    {
        Node* node = queue.front().first;
        if(node->isLeaf)
        {
            std::cout << dynamic_cast<LNode *>(node)->id << "\t";
        }
        else
        {
            if(queue.front().second)
                std::cout << "r" << level << "\t";
            else
                std::cout << "l" << level << "\t";
            queue.push( {dynamic_cast<INode*>(node)->left.get(), false});
            queue.push( {dynamic_cast<INode*>(node)->right.get(), true});
        }
        queue.pop();

        if(queue.front().first == nullptr)
        {
            queue.pop();
            if(!queue.empty())
                queue.push({nullptr,false});
            std::cout << "\n";
            level++;
        }
    }
    std::cout << std::endl;
}

int findSplit(const spaceKey* sortedKeys, int first, int last)
{

    spaceKey firstCode = sortedKeys[first];
    spaceKey lastCode = sortedKeys[last];

    // Identical Morton codes => split the range in the middle.
    if (firstCode == lastCode)
        return (first + last) >> 1u;

    // number of highest bits that are the same for all objects in the range
    // get number of leading bits that are the same
    int commonPrefix = __builtin_clz(first ^ last);

    // now search for the id where the code changes
    // use binary search to find the where the highest different morten key bit changes first
    // meaning the highest object that shares more then commonPrefix bits with the first one
    int split = first;
    int step = last - first;

    do
    {
        step = (step+1u) >> 1u; // divide step by 2
        int newSplit = split + step; // new guess for the split

        if(newSplit < last)
        {
            spaceKey splitCode = sortedKeys[newSplit];
            int splitPrefix = __builtin_clz(firstCode ^ splitCode);
            if(splitPrefix > commonPrefix)
                split = newSplit; // except guess if it shares more bits with the first element
        }
    }
    while(step > 1);

    return split;
}

std::shared_ptr<Node> generateHierarchy(spaceKey* sortedKeys, int first, int last)
{
    // single object ==> leaf
    if(first == last)
        return std::make_shared<LNode>(first);

    int split = findSplit(sortedKeys, first, last);

    // recurse sub ranges:
    std::shared_ptr<Node> cA = generateHierarchy(sortedKeys, first, split);
    std::shared_ptr<Node> cB = generateHierarchy(sortedKeys, split+1, last);

    return std::make_shared<INode>(std::move(cA),std::move(cB));
}

// tree settings
/////////////////
constexpr f3_t domainMin = {-2,-2,-2};
constexpr f3_t domainMax = {2,2,2};
//#define DEBUG_PRINTS
/////////////////

void buildMeATree(HostParticleBuffer<HOST_POSM>& pb)
{
    // generate morton keys for all particles
    spaceKey mKeys[pb.size()];
    f3_t domainFactor = 1.0_ft / (domainMax - domainMin);
    for(int i =0; i < pb.size(); i++)
    {
        Particle<POS> p = pb.loadParticle<POS>(i);
        mKeys[i] = calculatePositionKey(p.pos,domainMin,domainFactor);
    }

#ifdef DEBUG_PRINTS
    std::cout << "morton keys generated:\n";
    for(int i =0; i < pb.size(); i++)
    {
        std::bitset<64> x(mKeys[i]);
        std::cout << x << "\n";
    }
    std::cout << std::endl;
#endif

    // sort particles by morton key
    for(int j=0; j< pb.size();j++)
    {
        bool swapped = false;
        for(int i = 0; i < pb.size() - 1; i++)
        {
            if(mKeys[i] > mKeys[i + 1])
            {
                auto p1 = pb.loadParticle(i);
                auto p2 = pb.loadParticle(i + 1);
                pb.storeParticle(i, p2);
                pb.storeParticle(i + 1, p1);
                auto key = mKeys[i];
                mKeys[i] = mKeys[i + 1];
                mKeys[i + 1] = key;
                swapped = true;
            }
        }
        if(!swapped)
            break;
    }

#ifdef DEBUG_PRINTS
    std::cout << "morton keys sorted:\n";
    for(int i =0; i < pb.size(); i++)
        std::cout << mKeys[i] << "\n";
    std::cout << std::endl;
#endif

    // generate nodes and leafes
    std::shared_ptr<Node> tree= generateHierarchy(mKeys,0,pb.size()-1);
    printTree(tree.get());

    // calculate node and leaf data
    // profit
}

int main()
{
    HostParticleBuffer<HOST_POSM> pb(100);

    std::default_random_engine rng(mpu::getRanndomSeed());
    std::uniform_real_distribution<f1_t > dist(-2,2);

    for(int i =0; i<pb.size(); i++)
    {
        Particle<POS,MASS> p;
        p.mass = 1/pb.size();
        p.pos = f3_t{0,dist(rng),0};
        pb.storeParticle(i,p);
    }

#ifdef DEBUG_PRINTS
    for(int i =0; i<pb.size(); i++)
        std::cout << pb.loadParticle(i).pos << "\n";
    std::cout << std::endl;
#endif

    mpu::HRStopwatch sw;
    buildMeATree(pb);
    sw.pause();
    std::cout << "Tree generation took " << sw.getSeconds() *1000 << "ms" << std::endl;

#ifdef DEBUG_PRINTS
    for(int i =0; i<pb.size(); i++)
        std::cout << pb.loadParticle(i).pos << "\n";
    std::cout << std::endl;
#endif
}



///**
// * @brief first pass of derivative computation
// */
//struct cdA
//{
//    // define particle attributes to use
//    using load_type = Particle<POS,MASS,VEL,DENSITY,DSTRESS>; //!< particle attributes to load from main memory
//    using store_type = Particle<BALSARA,DENSITY_DT,DSTRESS_DT>; //!< particle attributes to store to main memory
//    using pi_type = merge_particles_t<load_type,store_type>; //!< the type of particle you want to work with in your job functions
//    using pj_type = Particle<POS,MASS,VEL,DENSITY>; //!< the particle attributes to load from main memory of all the interaction partners j
//    //!< when using do_for_each_pair_fast a SharedParticles object must be specified which can store all the attributes of particle j
//    template<size_t n> using shared = SharedParticles<n,SHARED_POSM,SHARED_VEL,SHARED_DENSITY>;
//
//    // setup some variables we need before during and after the pair interactions
//    m3_t edot{0}; // strain rate tensor (edot)
//    m3_t rdot{0}; // rotation rate tensor
//    f1_t divv{0}; // velocity divergence
//
//    //!< This function is executed for each particle before the interactions are computed.
//    CUDAHOSTDEV void do_before(pi_type& pi, size_t id)
//    {
//    }
//
//    //!< This function will be called for each pair of particles.
//    CUDAHOSTDEV void do_for_each_pair(pi_type& pi, const pj_type pj)
//    {
//        const f3_t rij = pi.pos-pj.pos;
//        const f1_t r2 = dot(rij,rij);
//        if(r2 <= H2 && r2>0)
//        {
//            // get the kernel gradient
//            f1_t r = sqrt(r2);
//            const f1_t dw = kernel::dWspline(r, H, dW_prefactor);
//            const f3_t gradw = (dw / r) * rij;
//
//            // strain rate tensor (edot) and rotation rate tensor (rdot)
//            const f3_t vij = pi.vel - pj.vel;
//            addStrainRateAndRotationRate(edot,rdot,pj.mass,pj.density,vij,gradw);
//            divv -= (pj.mass / pj.density) * dot(vij, gradw);
//        }
//    }
//
//    //!< This function will be called for particle i after the interactions with the other particles are computed.
//    CUDAHOSTDEV store_type do_after(pi_type& pi)
//    {
//        // deviatoric stress time derivative
//        pi.dstress_dt = dstress_dt(edot,rdot,pi.dstress,shear);
//
//        // density time derivative
//        pi.density_dt = -pi.density * divv;
//
//#if defined(BALSARA_SWITCH)
//        // get curl from edot and compute the balsara switch value
//        const f3_t curlv = f3_t{-2*rdot[1][2], -2*rdot[2][0], -2*rdot[0][1] };
//        pi.balsara = balsaraSwitch(divv, curlv, SOUNDSPEED, H);
//#endif
//
//        return pi;
//    }
//};
//
///**
// * @brief second pass of derivative computation
// */
//struct cdB
//{
//    // define particle attributes to use
//    using load_type = Particle<POS,MASS,VEL,BALSARA,DENSITY,DSTRESS>; //!< particle attributes to load from main memory
//    using store_type = Particle<ACC,XVEL>; //!< particle attributes to store to main memory
//    using pi_type = merge_particles_t<load_type,store_type>; //!< the type of particle you want to work with in your job functions
//    using pj_type = Particle<POS,MASS,VEL,BALSARA,DENSITY,DSTRESS>; //!< the particle attributes to load from main memory of all the interaction partners j
//    //!< when using do_for_each_pair_fast a SharedParticles object must be specified which can store all the attributes of particle j
//    template<size_t n> using shared = SharedParticles<n,SHARED_POSM,SHARED_VEL,SHARED_BALSARA,SHARED_DENSITY,SHARED_DSTRESS>;
//
//    // setup some variables we need before during and after the pair interactions
//#if defined(ENABLE_SPH)
//    m3_t sigOverRho_i; // stress over density square used for acceleration
//    #if defined(ARTIFICIAL_STRESS)
//        m3_t arts_i; // artificial stress from i
//    #endif
//#endif
//
//    //!< This function is executed for each particle before the interactions are computed.
//    CUDAHOSTDEV void do_before(pi_type& pi, size_t id)
//    {
//#if defined(ENABLE_SPH)
//        // build stress tensor for particle i using deviatoric stress and pressure
//        m3_t sigma_i = pi.dstress;
//        const f1_t pres_i = eos::murnaghan( pi.density, rho0, BULK, dBULKdP);
//        sigma_i[0][0] -= pres_i;
//        sigma_i[1][1] -= pres_i;
//        sigma_i[2][2] -= pres_i;
//
//        sigOverRho_i = sigma_i / (pi.density*pi.density);
//
//    #if defined(ARTIFICIAL_STRESS)
//        // artificial stress from i
//        m3_t arts_i = artificialStress(mateps, sigOverRho_i);
//    #endif
//#endif
//    }
//
//    //!< This function will be called for each pair of particles.
//    CUDAHOSTDEV void do_for_each_pair(pi_type& pi, const pj_type pj)
//    {
//        const f3_t rij = pi.pos-pj.pos;
//        const f1_t r2 = dot(rij,rij);
//        if(r2>0)
//        {
//
//#if defined(ENABLE_SELF_GRAVITY)
//            // gravity
//            const f1_t distSqr = r2 + H2;
//            const f1_t invDist = rsqrt(distSqr);
//            const f1_t invDistCube = invDist * invDist * invDist;
//            pi.acc -= rij * pj.mass * invDistCube;
//#endif
//
//#if defined(ENABLE_SPH)
//            if(r2 <= H2)
//            {
//                // get the kernel gradient
//                f1_t r = sqrt(r2);
//                const f1_t dw = kernel::dWspline(r, H, dW_prefactor);
//                const f3_t gradw = (dw / r) * rij;
//
//                // stress and pressure of j
//                m3_t sigma_j = pj.dstress;
//                const f1_t pres_j = eos::murnaghan(pj.density, rho0, BULK, dBULKdP);
//                sigma_j[0][0] -= pres_j;
//                sigma_j[1][1] -= pres_j;
//                sigma_j[2][2] -= pres_j;
//
//                m3_t sigOverRho_j = sigma_j / (pj.density * pj.density);
//
//                // stress from the interaction
//                m3_t stress = sigOverRho_i + sigOverRho_j;
//
//                const f3_t vij = pi.vel - pj.vel;
//    #if defined(ARTIFICIAL_STRESS)
//                // artificial stress
//                const f1_t f = pow(kernel::Wspline(r, H, W_prefactor) / kernel::Wspline(normalsep, H, W_prefactor) , matexp;
//                stress += f*(arts_i + artificialStress(mateps, sigOverRho_j));
//    #endif
//
//                // acceleration from stress
//                pi.acc += pj.mass * (stress * gradw);
//
//    #if defined(ARTIFICIAL_VISCOSITY)
//                // acceleration from artificial viscosity
//                pi.acc -= pj.mass *
//                          artificialViscosity(alpha, pi.density, pj.density, vij, rij, r, SOUNDSPEED, SOUNDSPEED
//        #if defined(BALSARA_SWITCH)
//                                  , pi.balsara, pj.balsara
//        #endif
//                          ) * gradw;
//    #endif
//
//    #if defined(XSPH)
//                // xsph
//                pi.xvel += 2 * pj.mass / (pi.density + pj.density) * (pj.vel - pi.vel) * kernel::Wspline<dimension>(r, H);
//    #endif
//            }
//#endif // ENABLE_SPH
//        }
//    }
//
//    //!< This function will be called for particle i after the interactions with the other particles are computed.
//    CUDAHOSTDEV store_type do_after(pi_type& pi)
//    {
//#if defined(CLOHESSY_WILTSHIRE)
//        pi.acc.x += 3*cw_n*cw_n * pi.pos.x + 2*cw_n* pi.vel.y;
//        pi.acc.y += -2*cw_n * pi.vel.x;
//        pi.acc.z += -cw_n*cw_n * pi.pos.z;
//#endif
//        return pi;
//    }
//};
//
//template <typename pbT>
//void computeDerivatives(pbT& particleBuffer)
//{
//#if defined(ENABLE_SPH)
//    do_for_each_pair_fast<cdA>(particleBuffer);
//#endif
//    do_for_each_pair_fast<cdB>(particleBuffer);
//}
//
///**
// * @brief perform leapfrog integration on the particles also performs the plasticity calculations
// * @param particles the device copy to the particle buffer that stores the particles
// * @param dt the timestep for the integration
// * @param not_first_step set false for the first integration step of the simulation
// * @param tanfr tangens of the internal friction angle
// */
//struct integrateLeapfrog
//{
//    using load_type = Particle<POS,VEL,ACC,XVEL,DENSITY,DENSITY_DT,DSTRESS,DSTRESS_DT>; //!< particle attributes to load from main memory
//    using store_type = Particle<POS,VEL,DENSITY,DSTRESS>; //!< particle attributes to store to main memory
//    using pi_type = merge_particles_t<load_type,store_type>; //!< the type of particle you want to work with in your job functions
//
//    //!< This function is executed for each particle. In p the current particle and in id its position in the buffer is given.
//    //!< All attributes of p that are not in load_type will be initialized to some default (mostly zero)
//    CUDAHOSTDEV store_type do_for_each(pi_type p, size_t id, f1_t dt, bool not_first_step)
//    {
//        //   calculate velocity a_t
//        p.vel = p.vel + p.acc * (dt * 0.5_ft);
//
//        // we could now change delta t here
//
//        // calculate velocity a_t+1/2
//        p.vel = p.vel + p.acc * (dt * 0.5_ft) * not_first_step;
//
//        // calculate position r_t+1
//#if defined(XSPH) && defined(ENABLE_SPH)
//        p.pos = p.pos + (p.vel + xsph_factor*p.xvel) * dt;
//#else
//        p.pos = p.pos + p.vel * dt;
//#endif
//
//#if defined(ENABLE_SPH)
//        // density
//        p.density = p.density + p.density_dt * dt;
//        if(p.density < 0.0_ft)
//            p.density = 0.0_ft;
//
//        // deviatoric stress
//        p.dstress += p.dstress_dt * dt;
//
//    #if defined(PLASTICITY_MC)
//        plasticity(p.dstress, mohrCoulombYieldStress( tanfr,eos::murnaghan(p.density,rho0, BULK, dBULKdP),cohesion));
//    #elif defined(PLASTICITY_MIESE)
//        plasticity(p.dstress,Y);
//    #endif
//
//#endif
//        return p; //!< return particle p, all attributes it shares with load_type will be stored in memory
//    }
//};
//
///**
// * @brief The main function of the simulation. Sets up the initial conditions and frontend and then manages running the simulation.
// *
// */
//
//
//int main()
//{
//    mpu::Log myLog( mpu::LogLvl::ALL, mpu::ConsoleSink());
//
//    std::string buildType;
//#if defined(NDEBUG)
//    buildType = "Release";
//#else
//    buildType = "Debug";
//#endif
//
//#if defined(STORE_RESULTS)
//    // set up file saving engine
//    ResultStorageManager storage(RESULT_FOLDER,RESULT_PREFIX,maxJobs);
//    // setup log output file
//    myLog.addSinks(mpu::FileSink( std::string(RESULT_FOLDER) + std::string(RESULT_PREFIX) + storage.getStartTime() + "_log.txt"));
//    // collect all settings and print them into a file
//    {
//        mpu::Resource headlessSettings = LOAD_RESOURCE(HeadlessSettings);
//        mpu::Resource precisionSettings = LOAD_RESOURCE(PrecisionSettings);
//        mpu::Resource settings = LOAD_RESOURCE(Settings);
//        std::ofstream settingsOutput(std::string(RESULT_FOLDER) + std::string(RESULT_PREFIX) + storage.getStartTime() + "_settings.txt");
//        settingsOutput << "//////////////////////////\n// headlessSettigns.h \n//////////////////////////\n\n"
//                        << std::string(headlessSettings.data(), headlessSettings.size())
//                        << "\n\n\n//////////////////////////\n// precisionSettings.h \n//////////////////////////\n\n"
//                        << std::string(precisionSettings.data(), precisionSettings.size())
//                        << "\n\n\n//////////////////////////\n// settigns.h \n//////////////////////////\n\n"
//                        << std::string(settings.data(), settings.size());
//    }
//#endif
//
//    myLog.printHeader("GraSPH2",GRASPH_VERSION,GRASPH_VERSION_SHA,buildType);
//    logINFO("GraSPH2") << "Welcome to GraSPH2!";
//#if defined(SINGLE_PRECISION)
//    logINFO("GraSPH2") << "Running in single precision mode.";
//#elif defined(DOUBLE_PRECISION)
//    logINFO("GraSPH2") << "Running in double precision mode.";
//#endif
//#if defined(USING_CUDA_FAST_MATH)
//    logWARNING("GraSPH2") << "Unsafe math optimizations enabled in CUDA code.";
//#endif
//    assert_cuda(cudaSetDevice(0));
//
//    // print some important settings to the console
//    myLog.print(mpu::LogLvl::INFO) << "\nSettings for this run:\n========================\n"
//                        << "Integration:"
//                        << "Leapfrog"
//                        << "Timestep: constant, " << timestep << "\n"
//                        << "Initial Conditions:\n"
//                #if defined(READ_FROM_FILE)
//                        << "Data is read from: " << FILENAME << "\n"
//                #elif defined(ROTATING_UNIFORM_SPHERE)
//                        << "Using a random uniform sphere with radius " << spawn_radius << "\n"
//                        << "Total mass: " << tmass << "\n"
//                        << "Number of particles: " << particle_count << "\n"
//                        << "additional angular velocity: " << angVel << "\n"
//                #elif defined(ROTATING_PLUMMER_SPHERE)
//                        << "Using a Plummer distribution with core radius " << plummer_radius << " and cutoff " << plummer_cutoff << "\n"
//                        << "Total mass: " << tmass << "\n"
//                        << "Number of particles: " << particle_count << "\n"
//                        << "additional angular velocity: " << angVel << "\n"
//                #endif
//                        << "Compressed radius set to " << compressesd_radius << "\n"
//                        << "resulting in particle radius of " << pradius << "\n"
//                        << "and smoothing length " << H << "\n"
//                        << "Material Settings:\n"
//                        << "material density: " << rho0 << "\n"
//                        << "speed of sound: " << SOUNDSPEED << "\n"
//                        << "bulk-modulus: " << BULK << "\n"
//                        << "shear-modulus: " << shear << "\n"
//                        << "Environment Settings:\n"
//                #if defined(CLOHESSY_WILTSHIRE)
//                        << "Clohessy-Wiltshire enabled with n = " << cw_n << "\n";
//                #else
//                        << "Clohessy-Wiltshire disabled" << "\n"
//                #endif
//                        ;
//
//    // set up frontend
//    fnd::initializeFrontend();
//    bool simShouldRun = false;
//    fnd::setPauseHandler([&simShouldRun](bool pause){simShouldRun = !pause;});
//
//    // generate some particles depending on options in the settings file
//    InitGenerator<HostParticlesType> generator;
//
//#if defined(READ_FROM_FILE)
//    generator.addParticles(ps::TextFile<particleToRead>(FILENAME,SEPERATOR));
//#elif defined(ROTATING_UNIFORM_SPHERE)
//    generator.addParticles( ps::UniformSphere(particle_count,spawn_radius,tmass,rho0).addAngularVelocity(angVel), true,true );
//#elif defined(ROTATING_PLUMMER_SPHERE)
//    generator.addParticles( ps::PlummerSphere(particle_count,plummer_radius,plummer_cutoff,tmass,rho0).addAngularVelocity(angVel), true, true);
//#endif
//
//    auto hpb = generator.generate();
//
//    // create cuda buffer
//    DeviceParticlesType pb(hpb.size());
//#if defined(FRONTEND_OPENGL)
//    fnd::setParticleSize(pradius);
//    pb.registerGLGraphicsResource<DEV_POSM>(fnd::getPositionBuffer(pb.size()));
//    pb.registerGLGraphicsResource<DEV_VEL>(fnd::getVelocityBuffer(pb.size()));
//    pb.registerGLGraphicsResource<DEV_DENSITY>(fnd::getDensityBuffer(pb.size()));
//    pb.mapGraphicsResource();
//#endif
//
//    // upload particles
//    pb = hpb;
//
//#if defined(STORE_RESULTS)
//    // print timestep 0
//    storage.printToFile(pb,0);
//    f1_t timeSinceStore=timestep;
//#endif
//
//    // start simulating
//    computeDerivatives(pb);
//    do_for_each<integrateLeapfrog>(pb,timestep,false);
//
//    double simulatedTime=timestep;
//#if defined(READ_FROM_FILE)
//    simulatedTime += startTime;
//#endif
//
//    pb.unmapGraphicsResource(); // used for frontend stuff
//    while(fnd::handleFrontend(simulatedTime))
//    {
//        if(simShouldRun)
//        {
//            pb.mapGraphicsResource(); // used for frontend stuff
//
//            // run simulation
//            computeDerivatives(pb);
//            do_for_each<integrateLeapfrog>(pb,timestep,true);
//
//            simulatedTime += timestep;
//
//#if defined(STORE_RESULTS)
//            timeSinceStore += timestep;
//            if( timeSinceStore >= store_intervall)
//            {
//                storage.printToFile(pb,simulatedTime);
//                timeSinceStore-=store_intervall;
//            }
//#endif
//
//            pb.unmapGraphicsResource(); // used for frontend stuff
//        }
//    }
//
//    return 0;
//}
//
