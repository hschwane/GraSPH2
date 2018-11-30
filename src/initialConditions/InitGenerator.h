/*
 * GraSPH2
 * InitGenerator.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the InitGenerator class
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */

#ifndef GRASPH2_INITGENERATOR_H
#define GRASPH2_INITGENERATOR_H

// includes
//--------------------
#include <vector>
#include <functional>
#include <utility>
#include "ParticleSource.h"
//--------------------

//-------------------------------------------------------------------
/**
 * class initGenerator
 * Generates initial conditions for the simulation.
 *
 * usage:
 * Use the addParticles function to pass particle sources with modifiers to get particles from.
 * When all sources are added generate a particle buffer containing the particles from all sources using generate().
 *
 */
template <typename hostParticleType>
class InitGenerator
{
public:
    template <typename ParticleSourceType>
    void addParticles(ParticleSourceType particleSource); //!< adds a particle source to get particles from

    hostParticleType generate(); //!< generates the initial conditions and returns a host buffer with particle data

    size_t numberOfParticles(){return m_totalNumOfParticles;} //!< shows how many particles will be generated once "generate()" is called

private:
    std::vector<std::pair<size_t,size_t>> m_genParameters; //!< contains start and end particle for each source
    std::vector<std::function< void(hostParticleType&,size_t,size_t) >> m_genFuncs; //!< vector containing all particle sources

    size_t m_totalNumOfParticles{0}; //!< the total number of particles generated by this generator
};


// template function definitions of the InitGenerator class
//-------------------------------------------------------------------

template<typename hostParticleType>
template<typename ParticleSourceType>
void InitGenerator<hostParticleType>::addParticles(ParticleSourceType particleSource)
{
    static_assert(std::is_base_of<ps::detail::ParticleSourceBaseFlag,ParticleSourceType>::value, "particleSource must be an object of a class derived from the ParticleSource class!");

    size_t numParticles = particleSource.getNumParticles();
    m_genFuncs.push_back(particleSource);
    m_genParameters.emplace_back(m_totalNumOfParticles,m_totalNumOfParticles+numParticles);
    m_totalNumOfParticles += numParticles;
}

template<typename hostParticleType>
hostParticleType InitGenerator<hostParticleType>::generate()
{
    logINFO("InitalConditions") << "Getting " << m_totalNumOfParticles << " particles from particle sources.";

    // create a buffer to hold the particles
    hostParticleType hpb(m_totalNumOfParticles);
//    hpb.initialize();

    size_t p=0; // we count the overall number of particles created over all sources
    // loop over all particle sources
    for(int i=0; i<m_genFuncs.size(); ++i)
    {
        // all particles that need to be generated by this source
        for(size_t j = m_genParameters[i].first; j < m_genParameters[i].second; ++j, ++p)
        {
            // generate particle p using the generation function i with index j
            m_genFuncs[i](hpb, j, p);
        }
    }
    return hpb;
}


#endif //GRASPH2_INITGENERATOR_H
