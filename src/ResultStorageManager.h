/*
 * GraSPH2
 * ResultStorageManager.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the ResultStorageManager class, which saves simulation results into files.
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */

#ifndef GRASPH2_RESULTSTORAGEMANAGER_H
#define GRASPH2_RESULTSTORAGEMANAGER_H

// includes
//--------------------
#include <thread>
#include <queue>
#include <mutex>
#include "particles/Particles.h"

//--------------------

// particle parameters that are saved to disk
using HostDiscPT = HostParticleBuffer<HOST_POSM,HOST_VEL,HOST_DENSITY>;

//-------------------------------------------------------------------
/**
 * class ResultStorageManager
 * Stores Simulation results in a file. The file is a tab seperated text file with one line for each particle.
 * Position, mass, velocity and density are stored for each particle. An output file can be used directly as an
 * input file for the "TextFile" particle source.
 * Filenames consist of the start time of the simulation and the current simulation time of the snapshot that is
 * being saved in the file. File i/o operation as well as copying from device to host are
 * handled in asyncronus to the simulation.
 *
 * usage:
 * Create an object of this class specifying an existing folder where simulation results are stored.
 * You also pass a prefix for the filename of this simulation and a maximum number of storage jobs that can live in
 * RAM at any given time.
 * To save a snapshot call printToFile() with your particle buffer. The function will duplicate the data that is needed
 * for storing and then return, while the actual file operations are handled in a different thread.
 *
 */
class ResultStorageManager
{
public:
    ResultStorageManager(std::string directory, std::string prefix, int maxJobs);
    ~ResultStorageManager();

    template<typename deviceParticleType, std::enable_if_t< mpu::is_instantiation_of<DeviceParticleBuffer,deviceParticleType>::value,int> =0>
    void printToFile(deviceParticleType particles, f1_t time); //!< add a simulation snapshot from the device to the storage queue

    template<typename hostParticleType, std::enable_if_t< mpu::is_instantiation_of<HostParticleBuffer,hostParticleType>::value,int> =0>
    void printToFile(hostParticleType particles, f1_t time); //!< add a simulation snapshot from the host to the storage queue

    std::string getStartTime() const {return m_startTime;} //!< get the timestep when the simulation was started

private:

    const std::string m_directory; //!< the directory where all the snapshots are saved
    const std::string m_prefix; //!< prefix for files of the current simulation
    const std::string m_startTime; //!< the timestamp of when the simulation was started

    const int m_maxQueue; //!< the maximum number of jobs waiting in the queue before the simulation is paused

    using hdcQueueType=std::pair<std::unique_ptr<HostDiscPT>,f1_t >; //!< type of elements in first queue
    using ddcQueueType=std::pair<std::unique_ptr<HostDiscPT::deviceType>,f1_t >; //!< type of elements in second queue

    std::queue<hdcQueueType> m_hostDiskCopy; //!< device data that is waiting to be transfered to host memory
    std::queue<ddcQueueType> m_deviceDiskCopy; //!< host data waiting to be written to a file

    std::mutex m_queueMutex; //!< mutex for first queue
    std::mutex m_workerMutex; //!< mutex for the conditional variable of the worker thread
    std::condition_variable m_workSignal; //!< conditional variable to signal the worker thread when work is availible

    bool m_terminateWorker; //!< set to true to terminate the worker thread once all jobs are done
    std::atomic_int m_numberJobs; //!< the number of jobs currently in all queues
    std::thread m_workerThread; //!< handle to the worker thread
    void worker(); //!< main function of the worker thread

    struct attributePrinter
    {
    public:
        template <typename T> CUDAHOSTDEV void operator()(T v);
        explicit attributePrinter(std::ostream& s);
        ~attributePrinter();
    private:
        std::ostream& m_stream;
    };

    void printTextFile(HostDiscPT& data, f1_t time); //!< function to actually print data to a file
    void printHDF5File(HostDiscPT& data, f1_t time); //!< function to actually print data to a HDF5 file
};


//-------------------------------------------------------------------
// definitions of template functions of the ResultStorageManager

template<typename deviceParticleType, std::enable_if_t< mpu::is_instantiation_of<DeviceParticleBuffer,deviceParticleType>::value,int>>
void ResultStorageManager::printToFile(deviceParticleType particles, f1_t time)
{
    // wait for things to be written so memory frees up
    if(m_numberJobs > m_maxQueue)
    {
        logWARNING("ResultStorageManager") << "Storage Queue full. Pausing simulation until data is written to file.";
        while( m_numberJobs>1)
        {

            mpu::yield();
            mpu::sleep_ms(10);
        }
    }
    m_numberJobs++;

    // try to buffer in gpu memory
    try
    {
        std::unique_ptr<HostDiscPT::deviceType> downloadCopy(new HostDiscPT::deviceType(particles));
        assert_cuda(cudaGetLastError());

        std::lock_guard<std::mutex> lock(m_queueMutex);
        m_deviceDiskCopy.emplace(std::move(downloadCopy),time);
        m_workSignal.notify_one();
    }
    catch (const std::exception& e)
    {
        logWARNING("ResultStorage") << "Could not duplicate simulation data on the device. Copying directly to main memory.";

        std::unique_ptr<HostDiscPT> hostData(new HostDiscPT(particles, true));
        assert_cuda(cudaGetLastError());

        std::lock_guard<std::mutex> lock(m_queueMutex);
        m_hostDiskCopy.emplace(std::move(hostData),time);
        m_workSignal.notify_one();
    }
}

template<typename hostParticleType, std::enable_if_t< mpu::is_instantiation_of<HostParticleBuffer,hostParticleType>::value,int>>
void ResultStorageManager::printToFile(hostParticleType particles, f1_t time)
{
    // wait for things to be written so memory frees up
    if(m_numberJobs > m_maxQueue)
    {
        logWARNING("ResultStorageManager") << "Storage Queue full. Pausing simulation until data is written to file.";
        while( m_numberJobs>1)
        {
            mpu::yield();
            mpu::sleep_ms(10);
        }
    }
    m_numberJobs++;

    std::unique_ptr<HostDiscPT> hostData(new HostDiscPT(particles));

    std::lock_guard<std::mutex> lock(m_queueMutex);
    m_hostDiskCopy.emplace(std::move(hostData),time);
    m_workSignal.notify_one();

}

#endif //GRASPH2_RESULTSTORAGEMANAGER_H
