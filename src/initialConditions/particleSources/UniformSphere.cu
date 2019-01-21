/*
 * GraSPH2
 * UniformSphere.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the UniformSphere class
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */

// includes
//--------------------
#include "UniformSphere.h"
#include <mpUtils/mpUtils.h>
//--------------------

// namespace
//--------------------
namespace ps {
//--------------------

// function definitions of the UniformSphere class
//-------------------------------------------------------------------

UniformSphere::UniformSphere(size_t particleCount, f1_t radius, f1_t totalMass, f1_t materialDensity, unsigned int seed)
    : m_radius(radius), m_particleMass(totalMass/particleCount), m_matDensity(materialDensity), m_rng(seed)
{
    m_numberOfParticles = particleCount;
}

Particle<POS,MASS,DENSITY> UniformSphere::generateParticle(size_t id)
{
    Particle<POS,MASS,DENSITY> p;

    mpu::randUniformSphere(  m_dist(m_rng),  m_dist(m_rng),  m_dist(m_rng), m_radius, p.pos.x, p.pos.y, p.pos.z);
    p.mass = m_particleMass;
    p.density = m_matDensity;

    return p;
}

}