/*
 * GraSPH2
 * ResultStorageManager.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the ResultStorageManager class
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
using DeviceDiscPT = Particles<DEV_POSM,DEV_VEL,DEV_DENSITY>;
using HostDiscPT = Particles<HOST_POSM,HOST_VEL,HOST_DENSITY>;
#define DiscPbases POS,MASS,VEL,DENSITY
using DiscPT = Particle<DiscPbases>;

/**
 * @brief store information about ongoing memory transfer
 */
struct OngoingTransfer
{
    std::unique_ptr<DeviceDiscPT> source;
    std::unique_ptr<HostDiscPT> target;
    f1_t time;
    cudaEvent_t event;

    OngoingTransfer(std::unique_ptr<DeviceDiscPT> s, std::unique_ptr<HostDiscPT> t, f1_t ti, cudaEvent_t e)
        : source(std::move(s)), target(std::move(t)), time(ti), event(e)
    {
    }
};

//-------------------------------------------------------------------
/**
 * class ResultStorageManager
 * Stores Simulation results in a file.
 *
 * usage:
 *
 */
class ResultStorageManager
{
public:
    ResultStorageManager(std::string directory, std::string prefix, int maxJobs);
    ~ResultStorageManager();

    template<typename deviceParticleType>
    void printToFile(deviceParticleType particles, f1_t time);

private:

    std::string m_directory;
    std::string m_prefix;
    std::string m_startTime;

    const int m_maxQueue;

    using hdcQueueType=std::pair<std::unique_ptr<DeviceDiscPT>,f1_t >;
    using ddcQueueType=std::pair<std::unique_ptr<HostDiscPT>,f1_t >;

    std::queue<hdcQueueType> m_hostDeviceCopy;
    std::queue<OngoingTransfer> m_ongoingTransfers;
    std::queue<ddcQueueType> m_deviceDiskCopy;

    std::mutex m_hdcMutex;
    std::mutex m_ddcMutex;
    std::mutex m_workerMutex;
    std::condition_variable m_workSignal;

    bool m_terminateWorker;
    std::atomic_int m_numberJobs;
    std::thread m_workerThread;
    void worker();

    void printTextFile(ddcQueueType data);
};


//-------------------------------------------------------------------
// definitions of template functions of the ResultStorageManager

template<typename deviceParticleType>
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
        std::unique_ptr<DeviceDiscPT> downloadCopy(new DeviceDiscPT(particles));
        assert_cuda(cudaGetLastError());
        assert_cuda(cudaStreamSynchronize(nullptr));

        std::lock_guard<std::mutex> lock(m_hdcMutex);
        m_hostDeviceCopy.emplace(std::move(downloadCopy),time);
        m_workSignal.notify_one();
    }
    catch (const std::exception& e)
    {
        logWARNING("ResultStorage") << "Could not duplicate simulation data on the device. Copying directly to main memory.";

        std::unique_ptr<HostDiscPT> hostData(new HostDiscPT(particles));
        assert_cuda(cudaGetLastError());
        assert_cuda(cudaStreamSynchronize(nullptr));

        std::lock_guard<std::mutex> lock(m_ddcMutex);
        m_deviceDiskCopy.emplace(std::move(hostData),time);
        m_workSignal.notify_one();
    }
}

#endif //GRASPH2_RESULTSTORAGEMANAGER_H
