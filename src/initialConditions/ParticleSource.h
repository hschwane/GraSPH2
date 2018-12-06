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
#include <glm/glm.hpp>
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
template<typename particleType>
class ParticleSource : public detail::ParticleSourceBaseFlag
{
public:
    virtual ~ParticleSource() = default;

    size_t getNumParticles() { return m_numberOfParticles; } //!< the number of particles provided by this source

    // transformation
    template<typename vecType>
    ParticleSource &move(vecType position); //!< move the generated particles to a position

    //ParticleSource rotate(f3_t axis, f1_t angle); //!< rotate object and velocities around axis by angle

    // initial values for other properties
    ParticleSource &addAngularVelocity(f3_t omega); //!< add angular velocity around axis omega with strength length(omega)

    template<typename particleBufferType>
    void operator()(particleBufferType &particles, size_t id,
                    size_t pos); //!< function to generate a single id < getNumparticles(), pos is the position in the buffer where the particle is saved
protected:
    size_t m_numberOfParticles{0}; //!< number of particles provided by this source
    using ptType=particleType; //!< the type of particle created by this source (shorthand to be used in derived classes)

private:
    virtual particleType
    generateParticle(size_t id) = 0; //!< function to be overridden ny derived classes to provide particles

    std::vector<std::function<void(
            particleType &)>> m_operations; //!< a list of modifiers that are performed on the particles after generation e.g. add velocity
};

//!< bases that every particle should have, so modifiers can work
#define PS_DEFAULT_PARTICLE_BASES POS,MASS,VEL

// template function definitions of the ParticleSource class
//-------------------------------------------------------------------

template<typename particleType>
template<typename vecType>
ParticleSource<particleType> &ParticleSource<particleType>::move(vecType position)
{
    m_operations.push_back([position](particleType &p)
                           {
                                p.pos += position;
                           });
    return *this;
}

template<typename particleType>
ParticleSource<particleType> &ParticleSource<particleType>::addAngularVelocity(f3_t omega)
{
    m_operations.push_back([omega](particleType &p)
                           {
                                p.vel = cross(omega,p.pos);
                           });
    return *this;
}

template<typename particleType>
template<typename particleBufferType>
void ParticleSource<particleType>::operator()(particleBufferType &particles, size_t id, size_t pos)
{
    particleType p = generateParticle(id);
    for(const auto &operation : m_operations)
    {
        operation(p);
    }
    particles.storeParticle(pos, p);
}

}

#endif //GRASPH2_PARTICLEGENERATOR_H
