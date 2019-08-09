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

void ResultStorageManager::printFile(HostDiscPT& data, f1_t time)
{
#if defined(STORE_HDF5)
    printHDF5File(data,time);
#else
    printTextFile(data,time);
#endif
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

                    printFile(hostData,deviceJob.second);
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

                    printFile(*hostJob.first, hostJob.second);
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

// functions for hdf5 output
#if defined(STORE_HDF5)

template<typename T>
void ResultStorageManager::attributeHDF5Printer::operator()(T v)
{
#ifndef __CUDA_ARCH__ // protection against calling from device code (mostly to shut up compiler warning)
    assert_critical(false,"ResultStorageManager","implement the attribute hdf5 printer for your data type");
#endif
}

template<>
void ResultStorageManager::attributeHDF5Printer::operator()(f1_t v)
{
#ifndef __CUDA_ARCH__
    m_data.push_back(v);
#endif
}

template<>
void ResultStorageManager::attributeHDF5Printer::operator()(f2_t v)
{
#ifndef __CUDA_ARCH__
    m_data.push_back(v.x);
    m_data.push_back(v.y);
#endif
}

template<>
void ResultStorageManager::attributeHDF5Printer::operator()(f3_t v)
{
#ifndef __CUDA_ARCH__
    m_data.push_back(v.x);
    m_data.push_back(v.y);
    m_data.push_back(v.z);
#endif
}

template<>
void ResultStorageManager::attributeHDF5Printer::operator()(f4_t v)
{
#ifndef __CUDA_ARCH__
    m_data.push_back(v.x);
    m_data.push_back(v.y);
    m_data.push_back(v.z);
    m_data.push_back(v.w);
#endif
}

template<>
void ResultStorageManager::attributeHDF5Printer::operator()(m3_t v)
{
#ifndef __CUDA_ARCH__
    for(int i = 0; i < 9; ++i)
    {
        m_data.push_back(v(i));
    }
#endif
}

ResultStorageManager::attributeHDF5Printer::attributeHDF5Printer(std::vector<float>& s) : m_data(s)
{
}

template <typename A>
void ResultStorageManager::writeAttributeDataset(const HostDiscPT& data, HighFive::File& file)
{
    using namespace HighFive;

    //Create DataSpace for DataSet (min size and max size);
    DataSpace dspace = HighFive::DataSpace({data.size(), static_cast<size_t>(getDim<typename A::type>())});

    // Create a new Dataset
    DataSet dset = file.createDataSet(std::string(A::debugName()), dspace, AtomicType<float>());

    // fill the dataset
    for (int i = 0; i < data.size(); ++i)
    {
        auto p = data.loadParticle<A>(i);
        std::vector<float> res;
        p.doForEachAttribute(ResultStorageManager::attributeHDF5Printer(res));
        dset.select({size_t(i),0},{1,res.size()}).write(res);
    }
}

template<typename ...Args>
void ResultStorageManager::writeAllParticles<Args...>::operator()(const HostDiscPT &data, HighFive::File& file)
{
    int t[] = {0, ((void) (writeAttributeDataset<Args>(data, file)), 1)...};
    (void) t[0]; // silence compiler warning about t being unused
}

void ResultStorageManager::printHDF5File(HostDiscPT& data, f1_t time)
{
    std::ostringstream filename;
    filename << m_directory << m_prefix << m_startTime << "_" << std::fixed << std::setprecision(std::numeric_limits<f1_t>::digits10 + 1) << time << ".h5";

    //Create HDF5 File and DataSet which stores the result of one time step (all attributes of all particles at this timestep)
    HighFive::File file(filename.str(), HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Truncate);

    mpu::instantiate_from_tuple_t<writeAllParticles, HostDiscPT::particleType::attributes> myWriteFunction;
    myWriteFunction(data, file);
}

#endif