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
 * using pseudo random point picking to generate particles in sphere with e density profile according to the plummer model
 *
 * usage:
 *
 */
class PlummerSphere : public ParticleSource<Particle<POS, MASS, DENSITY>, PlummerSphere>
{
public:
    PlummerSphere(size_t particleCount, f1_t plummerRadius, f1_t totalMass, f1_t materialDensity);
    ~PlummerSphere() override = default;

    PlummerSphere &addRandomPlummerVelocity(f1_t G = 1); //! add random velocity according to plummer distribution to each particle but make sure it is smaller then escape velocity
    PlummerSphere &addAngularPlummerVelocity(f3_t axis, f1_t G = 1); //! add rotational velocity according to plummer distribution to each particle but make sure it is smaller then escape velocity

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
