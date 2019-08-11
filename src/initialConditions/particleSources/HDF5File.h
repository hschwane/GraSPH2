/*
 * GraSPH2
 * HDF5File.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the HDF5File class
 *
 * Copyright (c) 2019 Hendrik Schwanekamp
 *
 */

#ifndef GRASPH2_HDF5FILE_H
#define GRASPH2_HDF5FILE_H

// includes
//--------------------
#include "../ParticleSource.h"
#include <particles/Particles.h>
#include <highfive/H5File.hpp>
#include <unordered_map>
#include <vector>
//--------------------

// namespace
//--------------------
namespace ps {
//--------------------

//-------------------------------------------------------------------
/**
 * class HDF5File
 *
 * usage:
 *
 */
template <typename ParticleToRead>
class HDF5File : public ParticleSource<ParticleToRead,HDF5File<ParticleToRead>>
{
public:
    explicit HDF5File(const std::string& filename); //!< construct this using a filename and a seperator
    ~HDF5File() override = default;

private:
    ParticleToRead generateParticle(size_t id) override; //!< function used by the base class to generate a particle

    HighFive::File m_file; //!< all lines from the file

    using DsIdMap = std::unordered_map<std::string, int>;
    using DsStorage = std::vector<HighFive::DataSet>;
    DsIdMap m_datasetId; //!< map dataset names to ids in vector
    DsStorage m_datasets; //!< preload all datasets here

    struct datasetLoader
    {
    public:
        explicit datasetLoader(const HighFive::File& file, DsStorage& datasets, DsIdMap& datasetId);
        bool allAttributesInFile() {return m_allThere;}

        template<typename T> void operator()();
    private:
        bool m_allThere;
        const HighFive::File& m_fileToCheck;
        DsStorage& m_datasets;
        DsIdMap& m_datasetId;
    };

    struct attributeLoader
    {
    public:
        explicit attributeLoader(const HighFive::File& file, ParticleToRead& p, size_t id, DsStorage& datasets, DsIdMap& datasetId);
        template<typename T> void operator()();
    private:
        size_t m_id;
        ParticleToRead& m_particle;
        const HighFive::File& m_hdf5file;
        DsStorage& m_datasets;
        DsIdMap& m_datasetId;
    };
};

// function definitions of the HDF5File class
//-------------------------------------------------------------------

template <typename ParticleToRead>
HDF5File<ParticleToRead>::HDF5File(const std::string& filename)
    : m_file(filename, HighFive::File::ReadOnly)
{
    datasetLoader ac(m_file, m_datasets,m_datasetId);
    ParticleToRead::doForEachAttributeType(ac);
    if( !ac.allAttributesInFile())
    {
        logERROR("InitialConditions") << "File does not contain all required particle attributes. Filename: " << filename;
        logFlush();
        throw std::runtime_error("File " + filename + " does not contain all required particle attributes.");
    }

    HighFive::DataSet ds = m_file.getDataSet( std::tuple_element_t<0,typename ParticleToRead::attributes>::debugName());
    std::vector<size_t> dim = ds.getDimensions();
    ParticleSource<ParticleToRead,HDF5File<ParticleToRead>>::m_numberOfParticles = dim[0];
}

template <typename ParticleToRead>
HDF5File<ParticleToRead>::datasetLoader::datasetLoader(const HighFive::File& file, DsStorage& datasets, DsIdMap& datasetId)
        : m_allThere(true), m_fileToCheck(file), m_datasets(datasets), m_datasetId(datasetId)
{
}

template <typename ParticleToRead>
template <typename T>
void HDF5File<ParticleToRead>::datasetLoader::operator()()
{
    if( !m_fileToCheck.exist( T::debugName()) )
    {
        logWARNING("InitialConditions") << "Attribute " << T::debugName << " does not exist in file";
        m_allThere = false;
    }

    HighFive::DataSet ds = m_fileToCheck.getDataSet(T::debugName());
    std::pair<std::string,int> sdsp(T::debugName(), m_datasets.size());
    m_datasets.push_back(ds);
    m_datasetId.insert( sdsp );
}

template <typename ParticleToRead>
ParticleToRead HDF5File<ParticleToRead>::generateParticle(size_t id)
{
    ParticleToRead p;
    attributeLoader al(m_file,p,id,m_datasets,m_datasetId);
    ParticleToRead::doForEachAttributeType(al);
    return p;
}

template <typename ParticleToRead>
HDF5File<ParticleToRead>::attributeLoader::attributeLoader(const HighFive::File& file, ParticleToRead& p, size_t id, DsStorage& datasets, DsIdMap& datasetId)
    : m_hdf5file(file), m_particle(p), m_id(id), m_datasets(datasets), m_datasetId(datasetId)
{
}

template <typename ParticleToRead>
template <typename T>
void HDF5File<ParticleToRead>::attributeLoader::operator()()
{
    m_datasets[m_datasetId[T::debugName()]].select({m_id,0},{1,getDim<typename T::type>()}).read( reinterpret_cast<f1_t*>(&m_particle.template getAttributeRef<T>()));
}

}


#endif //GRASPH2_HDF5FILE_H
