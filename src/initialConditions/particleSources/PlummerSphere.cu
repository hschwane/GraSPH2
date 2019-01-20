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

PlummerSphere::PlummerSphere(size_t particleCount, f1_t plummerRadius, f1_t totalMass, f1_t materialDensity)
        : m_plummerRadius(plummerRadius), m_totalMass(totalMass), m_particleMass(totalMass/particleCount), m_matDensity(materialDensity)
{
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
    p.pos *= m_plummerRadius / std::sqrt( std::pow( m_dist(m_rng) / m_totalMass, -2.0/3.0) -1 );

    p.mass = m_particleMass;
    p.density = m_matDensity;

    return p;
}

}