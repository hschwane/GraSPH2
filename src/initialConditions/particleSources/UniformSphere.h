/*
 * GraSPH2
 * UniformSphere.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the UniformSphere class
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */

#ifndef GRASPH2_UNIFORMSPHERE_H
#define GRASPH2_UNIFORMSPHERE_H

// includes
//--------------------
#include "../ParticleSource.h"
#include <particles/Particles.h>
#include <random>
//--------------------

// namespace
//--------------------
namespace ps {
//--------------------

//-------------------------------------------------------------------
/**
 * class UniformSphere
 * using pseudo-random point picking to generate particles in a sphere with uniform density
 *
 * usage:
 *
 *
 */
class UniformSphere : public ParticleSource<Particle<POS,MASS,DENSITY>,UniformSphere>
{
public:
    UniformSphere(size_t particleCount, f1_t radius, f1_t totalMass, f1_t materialDensity);
    ~UniformSphere() override = default;

private:
    Particle<POS,MASS,DENSITY> generateParticle(size_t id) override;

    f1_t m_radius;
    f1_t m_matDensity;
    f1_t m_particleMass;
    std::default_random_engine m_rng{};
    std::uniform_real_distribution<f1_t> m_dist{0.0,1.0};
};

}
#endif //GRASPH2_UNIFORMSPHERE_H
