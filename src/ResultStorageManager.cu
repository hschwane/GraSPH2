/*
 * GraSPH2
 * ResultStorageManager.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the ResultStorageManager class, which saves simulation results into files.
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */

// includes
//--------------------
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>
#include "ResultStorageManager.h"
//--------------------


// function definitions of the ResultStorageManager class
//-------------------------------------------------------------------

ResultStorageManager::ResultStorageManager(std::string directory, std::string prefix, int maxJobs)
    : m_directory(directory), m_prefix(prefix), m_terminateWorker(false), m_maxQueue(maxJobs),
    m_numberJobs(0), m_startTime(mpu::timestamp("%Y-%m-%d_%H:%M"))

{
    assert_critical(m_maxQueue>1, "ResultStorageManager", "Can't work with Maximum job number below 1.")
    m_workerThread = std::thread(&ResultStorageManager::worker, this);
}

void ResultStorageManager::worker()
{
    std::unique_lock<std::mutex> lck(m_workerMutex);
    HostDiscPT hostData;
    while(!m_terminateWorker)
    {
        // wait until there is work to do
        m_workSignal.wait(lck);

        while(m_numberJobs > 0)
        {
            // download data from gpu to cpu if any
            {
                std::unique_lock<std::mutex> hdc_lck(m_queueMutex);
                if(!m_deviceDiskCopy.empty())
                {
                    ddcQueueType deviceJob = std::move(m_deviceDiskCopy.front());
                    m_deviceDiskCopy.pop();
                    hdc_lck.unlock();

                    if(hostData.size() != deviceJob.first->size())
                    {
                        hostData = HostDiscPT(deviceJob.first->size());
                        hostData.pinMemory();
                    }
                    hostData = *deviceJob.first;
                    assert_cuda(cudaGetLastError());

                    printHDF5File(hostData,deviceJob.second);
                    m_numberJobs--;
                    logDEBUG("ResultStorageManager") << "Results stored for t= " << deviceJob.second;
                }
            }

            {
                std::unique_lock<std::mutex> hdc_lck(m_queueMutex);
                if(!m_hostDiskCopy.empty())
                {
                    hdcQueueType hostJob = std::move(m_hostDiskCopy.front());
                    m_hostDiskCopy.pop();
                    hdc_lck.unlock();

                    printHDF5File(*hostJob.first, hostJob.second);
                    m_numberJobs--;
                    logDEBUG("ResultStorageManager") << "Results stored for t= " << hostJob.second;
                }
            }
        }
    }
}

ResultStorageManager::~ResultStorageManager()
{
    {
        std::lock_guard<std::mutex> lck(m_workerMutex);
        m_terminateWorker = true;
        m_workSignal.notify_one();
    }
    m_workerThread.join();
}

template<typename T>
void ResultStorageManager::attributePrinter::operator()(T v)
{
#ifndef __CUDA_ARCH__ // protection against calling from device code (mostly to shut up compiler warning)
    m_stream << v << "\t";
#endif
}

template<>
void ResultStorageManager::attributePrinter::operator()(f2_t v)
{
#ifndef __CUDA_ARCH__
    m_stream << v.x << "\t"
             << v.y << "\t";
#endif
}

template<>
void ResultStorageManager::attributePrinter::operator()(f3_t v)
{
#ifndef __CUDA_ARCH__
    m_stream << v.x << "\t"
             << v.y << "\t"
             << v.z << "\t";
#endif
}

template<>
void ResultStorageManager::attributePrinter::operator()(f4_t v)
{
#ifndef __CUDA_ARCH__
    m_stream << v.x << "\t"
             << v.y << "\t"
             << v.z << "\t"
             << v.w << "\t";
#endif
}

template<>
void ResultStorageManager::attributePrinter::operator()(m3_t v)
{
#ifndef __CUDA_ARCH__
    for(int i = 0; i < 9; ++i)
    {
        m_stream << v(i) << "\t";
    }
#endif
}

ResultStorageManager::attributePrinter::attributePrinter(std::ostream& s) : m_stream(s)
{
}

ResultStorageManager::attributePrinter::~attributePrinter()
{
    m_stream << "\n"; // particle finished, end the line
}

void ResultStorageManager::printTextFile(HostDiscPT& data, f1_t time)
{
    std::ostringstream filename;
    filename << m_directory << m_prefix << m_startTime << "_" << std::fixed << std::setprecision(std::numeric_limits<f1_t>::digits10 + 1) << time << ".tsv";
    std::ofstream file(filename.str());
    file << std::fixed << std::setprecision(std::numeric_limits<f1_t>::digits10 + 1);

    if(!file.is_open())
    {
        logERROR("ResultStorageManager") << "Could not open output file: " << filename.str() << " Make sure the path actually exists.";
        logFlush();
        throw std::runtime_error("Could not open output file.");
    }

    for(int i = 0; i < data.size(); ++i)
    {
        auto p = data.loadParticle(i);
        p.doForEachAttribute(attributePrinter(file));
    }

    if(file.fail())
    {
        logERROR("ResultStorageManager") << "Error writing output file: " << filename.str();
        logFlush();
        throw std::runtime_error("Could not write to output file.");
    }
}

writeP(f3_t vec3, dataset) ...
writeP(f2_t vec2, dataset) ...
writeP(f1_t vec2, dataset) ...

template<typename T>
size_t getDimension()
{return 0;}

template<>
size_t getDimension<f1_t>()
{return 1;}

template<>
size_t getDimension<f3_t>()
{return 3;}

template<>
size_t getDimension<f2_t>()
{return 2;}

template <typename A>
void writeAttributeDataset(const HostDiscPT& data, HighFive::File file)
{
    //For the Dataset we need information about the dimensions of the data type, e.g. if its vec3, we want to get 3
    size_t dim = getDimension<A::type>();
    std::vector<size_t> dims(2);
    dims[0] = data.size();
    dims[1] = dim;

    //Create DataSpace for DataSet
    HighFive::DataSpace dspace = DataSpace({data.size(),1}, {data.size(), 19})
    // Create a new Dataset
    HighFive::DataSet dset = file.createDataSet(A::debugName(), dspace, dims);

    //Since we've saved the dimensions of data in the variable dims, we cann just write the whole data at once without for loop
    dset.write(data);

    // create dataset ... A::debugName();
    /*for (int i = 0; i < data.size(); ++i)
    {
        auto p = data.loadParticle<A>(i);
        dset.write(p);
    }*/
}

template<typename ...Args>
struct writeAllParticles
{
    void operator()(const HostDiscPT &data, HighFive::File file)
    {
        int t[] = {0, ((void) (writeAttributeDataset<Args>(data, file)), 1)...};
        (void) t[0]; // silence compiler warning about t being unused
    }
};

void ResultStorageManager::printHDF5File(HostDiscPT& data, f1_t time)
{
    std::ostringstream filename;
    filename << m_directory << m_prefix << m_startTime << "_" << std::fixed << std::setprecision(std::numeric_limits<f1_t>::digits10 + 1) << time << ".h5";

    //Create HDF5 File and DataSet which stores the result of one time step (all attributes of all particles at this timestep)
    HighFive::File file(filename.str(), HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Truncate);

//    HighFive::DataSet dset = file.createDataSet(dataset_name, HighFive::DataSpace(data.size(),HostDiscPT::particleType::numAttributes()));

    mpu::instantiate_from_tuple_t<writeAllParticles, HostDiscPT::particleType::attributes> myWriteFunction;
    myWriteFunction(data, file);
}
