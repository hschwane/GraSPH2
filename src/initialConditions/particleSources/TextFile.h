/*
 * GraSPH2
 * TextFile.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the TextFile class
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */

#ifndef GRASPH2_TEXTFILE_H
#define GRASPH2_TEXTFILE_H

// includes
//--------------------
#include "../ParticleSource.h"
#include <particles/Particles.h>
//--------------------

// namespace
//--------------------
namespace ps {
//--------------------

//-------------------------------------------------------------------
/**
 * class TextFile
 * Read particles from a text file.
 *
 * usage:
 * Pass the filename and a seperator character to the constructor default seperator is a tab.
 * Add modifiers as desiered and pass the particle source to the generator.
 * The File format is expected to have one line per particle containing 9 floats seperated by the seperator character.
 * The Order of the parameter is assumed to be as follows:
 * POS_x | POS_y | POS_z | MASS | VEL_x | VEL_y | VEL_z | DENSITY
 * Spaces and empty lines are ignored.
 *
 */
class TextFile : public ParticleSource<Particle<POS,MASS,VEL,DENSITY>,TextFile>
{
public:
    TextFile(std::string filename, char seperator='\t'); //!< construct this using a filename and a seperator
    ~TextFile() override = default;

private:
    Particle<POS,MASS,VEL,DENSITY> generateParticle(size_t id) override; //!< function used by the base class to generate a particle

    std::vector<std::string> m_data; //!< all lines from the file
    const char m_seperator; //!< character seperating the values
};

}

#endif //GRASPH2_TEXTFILE_H
