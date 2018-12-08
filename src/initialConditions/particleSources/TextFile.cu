/*
 * GraSPH2
 * TextFile.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the TextFile class
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */

// includes
//--------------------
#include "TextFile.h"
#include <algorithm>
#include <fstream>
//--------------------

// namespace
//--------------------
namespace ps {
//--------------------

// function definitions of the TextFile class
//-------------------------------------------------------------------

TextFile::TextFile(std::string filename, char seperator) : m_data(), m_seperator(seperator)
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
    m_numberOfParticles = m_data.size();
}

TextFile::ptType TextFile::generateParticle(size_t id)
{
    ptType p;
    std::string token;

    // read the line into a stringstream
    std::istringstream line(m_data[id]);

    // now tokenize the string and build the particle

    std::getline(line,token,m_seperator);
    p.pos.x = std::stod(token);
    std::getline(line,token,m_seperator);
    p.pos.y = std::stod(token);
    std::getline(line,token,m_seperator);
    p.pos.z = std::stod(token);

    std::getline(line,token,m_seperator);
    p.vel.x = std::stod(token);
    std::getline(line,token,m_seperator);
    p.vel.y = std::stod(token);
    std::getline(line,token,m_seperator);
    p.vel.z = std::stod(token);

    std::getline(line,token,m_seperator);
    p.mass = std::stod(token);
    std::getline(line,token,m_seperator);
    p.density = std::stod(token);

    if(!line)
    {
        logERROR("InitialConditions") << "Error parsing input file at line " << id;
        logFlush();
        throw std::runtime_error("Error parsing file.");
    }

    return p;
}

}