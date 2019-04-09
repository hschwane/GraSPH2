/*
 * GraSPH2
 * PlummerSphere.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the PlummerSphere class
 *
 * Copyright (c) 2019 Hendrik Schwanekamp
 *
 */

// includes
//--------------------
#include "PlummerSphere.h"
//--------------------

// namespace
//--------------------
namespace ps {
//--------------------

// function definitions of the PlummerSphere class
//-------------------------------------------------------------------

PlummerSphere::PlummerSphere(size_t particleCount, f1_t plummerRadius, f1_t cutoffRadius, f1_t totalMass,
                             f1_t materialDensity, unsigned int seed)
        : m_plummerRadius(plummerRadius), m_totalMass(totalMass), m_particleMass(totalMass/particleCount), m_matDensity(materialDensity), m_rng(seed), m_cutoffRadius(cutoffRadius)
{
    logINFO("InitialConditions") << "PlummerSphere using random seed: " << seed;
    m_numberOfParticles = particleCount;
}

Particle<POS, MASS, DENSITY> PlummerSphere::generateParticle(size_t id)
{
    Particle<POS,MASS,DENSITY> p;

    mpu::randSphereShell(m_dist(m_rng),  m_dist(m_rng), p.pos.x, p.pos.y, p.pos.z);

    // according to Art of computational science volume 11:
    // (http://www.artcompsci.org/kali/vol/plummer/volume11.pdf)
    // m(r) = M * (1+ (a^2 / r^2))^(-3/2)
    // therefore: r(m) = a * ( (m/M)^(-2/3) -1 )^(-1/2)
    // where M is total mass of the system and a is the structural length scale / plummer radius
    // choosing a random value for m in range [0,M] <=> choosing random value for m/M in range [0,1]

    // now find radius within reasonable bounds
    f1_t radius = m_cutoffRadius+1;
    while(radius > m_cutoffRadius)
        radius = m_plummerRadius / std::sqrt( std::pow(  m_dist(m_rng), -2.0_ft/3.0_ft) -1 );

    p.pos *= radius;
    p.mass = m_particleMass;
    p.density = m_matDensity;

    return p;
}

PlummerSphere &PlummerSphere::addRandomPlummerVelocity(f1_t G, unsigned int seed)
{
    f1_t M = m_totalMass;
    f1_t a = m_plummerRadius;
    m_operations.push_back([G,M,a,seed](full_particle &p)
                           {
                               static std::default_random_engine rng(seed);
                               static std::uniform_real_distribution<f1_t> dist01(0,1);
                               static std::uniform_real_distribution<f1_t> dist001(0,0.1);

                               f1_t r = length(p.pos);

                               f1_t x = 0;
                               f1_t y = 0.1;

                               while ( y > x*x * std::pow( 1- x*x, 3.5) )
                               {
                                   x = dist01(rng);
                                   y = dist001(rng);
                               }

                               f3_t v;
                               mpu::randSphereShell(dist01(rng),  dist01(rng), v.x, v.y, v.z);
                               v *= x * std::sqrt( 2 * G * M / std::sqrt(r*r + a*a));

                               p.vel += v;
                           });
    return *this;
}

}