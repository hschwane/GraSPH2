/*
 * GraSPH2
 * PlummerSphere.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the PlummerSphere class
 *
 * Copyright (c) 2019 Hendrik Schwanekamp
 *
 */

#ifndef GRASPH2_PLUMMERSPHERE_H
#define GRASPH2_PLUMMERSPHERE_H

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
 * class PlummerSphere
 *
 * usage:
 *
 */
class PlummerSphere : public ParticleSource<Particle<POS, MASS, DENSITY>, PlummerSphere>
{
public:
    PlummerSphere(size_t particleCount, f1_t plummerRadius, f1_t totalMass, f1_t materialDensity);
    ~PlummerSphere() override = default;

private:
    Particle<POS, MASS, DENSITY> generateParticle(size_t id) override;

    f1_t m_plummerRadius;
    f1_t m_matDensity;
    f1_t m_totalMass;
    f1_t m_particleMass;
    std::default_random_engine m_rng{};
    std::uniform_real_distribution<f1_t> m_dist{0.0, 1.0};

};

}

#endif //GRASPH2_PLUMMERSPHERE_H
