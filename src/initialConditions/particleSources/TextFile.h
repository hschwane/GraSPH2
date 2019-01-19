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
#include <fstream>
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
 * The template parameter specifies the particle attributes to be read. For example Particle<POS,MASS,VEL,DENSITY> would assume the following file structure:
 * POS_x | POS_y | POS_z | MASS | VEL_x | VEL_y | VEL_z | DENSITY
 * Spaces and empty lines are ignored.
 *
 */
template <typename ParticleToRead>
class TextFile : public ParticleSource<ParticleToRead,TextFile<ParticleToRead>>
{
public:
    TextFile(std::string filename, char seperator='\t'); //!< construct this using a filename and a seperator
    ~TextFile() override = default;

private:
    ParticleToRead generateParticle(size_t id) override; //!< function used by the base class to generate a particle

    std::vector<std::string> m_data; //!< all lines from the file
    const char m_seperator; //!< character seperating the values

    // functor that reads the attributes
    struct attributeReader
    {
    public:
        CUDAHOSTDEV void operator()(f2_t& v);
        CUDAHOSTDEV void operator()(f3_t& v);
        CUDAHOSTDEV void operator()(f4_t& v);
        CUDAHOSTDEV void operator()(m3_t& v);
        template <typename T, std::enable_if_t < !(std::is_same<T,f2_t>::value || std::is_same<T,f3_t>::value || std::is_same<T,f4_t>::value || std::is_same<T,m3_t>::value) ,int> = 0>
        CUDAHOSTDEV void operator()(T& v);
        explicit attributeReader(std::istream& s, char seperator);
    private:
        std::istream& m_stream;
        char m_seperator;
    };
};

// function definitions of the TextFile class
//-------------------------------------------------------------------

template <typename ParticleToRead>
TextFile<ParticleToRead>::TextFile(std::string filename, char seperator) : m_data(), m_seperator(seperator)
{
    std::ifstream file(filename);
    if(!file.is_open())
    {
        logERROR("InitialConditions") << "Could not open file " << filename;
        logFlush();
        throw std::runtime_error("Error opening file.");
    }

    // load all lines
    std::string line;
    while(std::getline(file,line))
    {
        // check if the line actually contains anything
        if(!std::all_of( line.begin(),line.end(),[](char c){return std::isspace(static_cast<unsigned char>(c));}))
            m_data.push_back(line);
    }
    logINFO("InitialConditions") << "Found " << m_data.size() << " particles in file " << filename;

    // figure out how long many lines we got
    TextFile<ParticleToRead>::m_numberOfParticles = m_data.size();
}

template <typename ParticleToRead>
ParticleToRead TextFile<ParticleToRead>::generateParticle(size_t id)
{
    ParticleToRead p;

    // read the line into a stringstream
    std::istringstream line(m_data[id]);

    // now tokenize the string and build the particle
    p.doForEachAttribute(attributeReader(line,m_seperator));

    if(!line)
    {
        logERROR("InitialConditions") << "Error parsing input file at line " << id;
        logFlush();
        throw std::runtime_error("Error parsing file.");
    }

    return p;
}

template <typename ParticleToRead>
template <typename T, std::enable_if_t < !(std::is_same<T,f2_t>::value || std::is_same<T,f3_t>::value || std::is_same<T,f4_t>::value || std::is_same<T,m3_t>::value) ,int>>
void TextFile<ParticleToRead>::attributeReader::operator()(T& v)
{
#ifndef __CUDA_ARCH__ // protection against calling from device code (mostly to shut up compiler warning)
    std::string token;
    std::getline(m_stream,token,m_seperator);
    v = std::stod(token);
#endif
}

template <typename ParticleToRead>
void TextFile<ParticleToRead>::attributeReader::operator()(f2_t& v)
{
    (*this)(v.x);
    (*this)(v.y);
}

template <typename ParticleToRead>
void TextFile<ParticleToRead>::attributeReader::operator()(f3_t& v)
{
    (*this)(v.x);
    (*this)(v.y);
    (*this)(v.z);
}

template <typename ParticleToRead>
void TextFile<ParticleToRead>::attributeReader::operator()(f4_t& v)
{
    (*this)(v.x);
    (*this)(v.y);
    (*this)(v.z);
    (*this)(v.w);
}

template <typename ParticleToRead>
void TextFile<ParticleToRead>::attributeReader::operator()(m3_t& v)
{
    for(int i = 0; i < 9; i++)
        (*this)(v(i));
}


template<typename ParticleToRead>
TextFile<ParticleToRead>::attributeReader::attributeReader(std::istream &s, char seperator) : m_stream(s), m_seperator(seperator)
{
}

}

#endif //GRASPH2_TEXTFILE_H
