/*
 * GraSPH2
 * ParticleSource.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the ParticleSource class
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 * Interface for particle generators, classes that generate a set of particles, or an object.
 *
 */

#ifndef GRASPH2_PARTICLEGENERATOR_H
#define GRASPH2_PARTICLEGENERATOR_H

// includes
//--------------------
#include <types.h>
#include <random>
#include <particles/Particles.h>
//--------------------

// namespace
//--------------------
namespace ps {
//--------------------

namespace detail {
    //!< class to check if something is derived from any instantiation of ParticleSource
    class ParticleSourceBaseFlag
    {
    };
}

//-------------------------------------------------------------------
/**
 * class particleGenerator
 * virtual base class for all particle sources allows the user to apply modifiers to the particles read from a source
 *
 * usage:
 * If you are using a derived class of this to load/generate particles this allows you to add modifiers to the particle,
 * e.g. move the particles to another position or add velocity.
 *
 * Derive a class from one a specialization of this class depending on which particle attributes your particle source can provide.
 * Implement a constructor that takes all required parameters and prepares the state of your source. You also need to set
 * m_numberOfParticles to the number of particles your source will provide.
 * Then override the generateParticle(size_t id) where you return the particle with the index id.
 *
 */
template<typename particleType, typename Derived>
class ParticleSource : public detail::ParticleSourceBaseFlag
{
public:
    virtual ~ParticleSource() = default;

    size_t getNumParticles() { return m_numberOfParticles; } //!< the number of particles provided by this source

    // transformation (affecting )
    template<typename vecType>
    Derived &move(vecType position); //!< move the generated particles to a position
    Derived &rotate(f3_t axis, f1_t angle); //!< rotate object and velocities around axis by angle
    Derived &addPositionalNoise(f1_t strength, int seed = 10258); //!< add a values from a uniform random distribution to each position

    Derived &addAngularVelocity(f3_t omega); //!< add angular velocity around axis omega with strength length(omega)
    Derived &addLinearVelocity(f3_t v); //!< add linear velocity to particles
    Derived &addRandomVelocity(f1_t strength, int seed = 4568); //!< adds a value from a uniform random distribution to the velocity

    template <typename Attrib>
    Derived &setConstant(typename Attrib::type v); //!< set the attribute "Attrib" of all particles to the value v

    template<typename particleBufferType>
    void operator()(particleBufferType &particles, size_t id, size_t pos); //!< function to generate a single id < getNumparticles(), pos is the position in the buffer where the particle is saved
protected:
    size_t m_numberOfParticles{0}; //!< number of particles provided by this source

private:
    virtual particleType
    generateParticle(size_t id) = 0; //!< function to be overridden ny derived classes to provide particles

    std::vector<std::function<void(full_particle &)>> m_operations; //!< a list of modifiers that are performed on the particles after generation e.g. add velocity
};

// template function definitions of the ParticleSource class
//-------------------------------------------------------------------

template<typename particleType, typename Derived>
template<typename vecType>
Derived &ParticleSource<particleType,Derived>::move(vecType position)
{
    m_operations.push_back([position](full_particle &p)
                           {
                                p.pos += position;
                           });
    return *static_cast<Derived*>(this);
}

template<typename particleType, typename Derived>
Derived &ParticleSource<particleType,Derived>::addAngularVelocity(f3_t omega)
{
    m_operations.push_back([omega](full_particle &p)
                           {
                                p.vel += cross(omega,p.pos);
                           });
    return *static_cast<Derived*>(this);
}

template<typename particleType, typename Derived>
Derived &ParticleSource<particleType,Derived>::addLinearVelocity(f3_t v)
{
    m_operations.push_back([v](full_particle &p)
                           {
                               p.vel += v;
                           });
    return *static_cast<Derived*>(this);
}

template<typename particleType, typename Derived>
Derived &ParticleSource<particleType, Derived>::rotate(f3_t axis, f1_t angle)
{
    // build rotation matrix according to http://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToMatrix/
    axis = normalize(axis);
    f1_t c = std::cos(angle);
    m3_t M1 = c * m3_t(1.0);
    m3_t M2 = (1-c) * m3_t( axis.x*axis.x, axis.x*axis.y, axis.x*axis.z,
                           axis.x*axis.y, axis.y*axis.y, axis.y*axis.z,
                           axis.x*axis.z, axis.z*axis.y, axis.z*axis.z);
    m3_t M3 = std::sin(angle) * m3_t( 0, -axis.z, axis.y,
                                      axis.z, 0, -axis.x,
                                      -axis.y, axis.x, 0);
    m3_t R = M1 + M2 + M3;

    m_operations.push_back([R](full_particle &p)
                           {
                               p.pos = R * p.pos;
                               p.vel = R * p.vel;
                           });

    return *static_cast<Derived*>(this);
}

template<typename particleType, typename Derived>
template<typename Attrib>
Derived &ParticleSource<particleType, Derived>::setConstant(typename Attrib::type v)
{
    m_operations.push_back([v](full_particle &p)
                           {
                               p.template setAttribute<Attrib>(v);
                           });
    return *static_cast<Derived*>(this);
}

template<typename particleType, typename Derived>
Derived &ParticleSource<particleType, Derived>::addPositionalNoise(f1_t strength, int seed)
{
    m_operations.push_back([strength,seed](full_particle &p)
                           {
                                static std::default_random_engine rng(seed);
                                static std::uniform_real_distribution<f1_t> dist(-strength,strength);
                                p.pos += f3_t{dist(rng),dist(rng),dist(rng)};
                           });
    return *static_cast<Derived*>(this);
}

template<typename particleType, typename Derived>
Derived &ParticleSource<particleType, Derived>::addRandomVelocity(f1_t strength, int seed)
{
    m_operations.push_back([strength,seed](full_particle &p)
                           {
                               static std::default_random_engine rng(seed);
                               static std::uniform_real_distribution<f1_t> dist(-strength,strength);
                               p.vel += f3_t{dist(rng),dist(rng),dist(rng)};
                           });
    return *static_cast<Derived*>(this);
}

template<typename particleType, typename Derived>
template<typename particleBufferType>
void ParticleSource<particleType, Derived>::operator()(particleBufferType &particles, size_t id, size_t pos)
{
    full_particle p{};
    p = generateParticle(id);
    for(const auto &operation : m_operations)
    {
        operation(p);
    }
    particles.storeParticle(pos, p);
}

}

#endif //GRASPH2_PARTICLEGENERATOR_H
