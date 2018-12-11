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

void ResultStorageManager::worker()
{
    std::unique_lock<std::mutex> lck(m_workerMutex);
    while(!m_terminateWorker)
    {
        // wait until there is work to do
        m_workSignal.wait(lck);

        while(m_numberJobs > 0)
        {
            // download data from gpu to cpu if any
            {
                std::unique_lock<std::mutex> hdc_lck(m_hdcMutex);
                if(!m_hostDeviceCopy.empty())
                {
                    hdcQueueType deviceData = std::move(m_hostDeviceCopy.front());
                    m_hostDeviceCopy.pop();
                    hdc_lck.unlock();

                    std::unique_ptr<HostDiscPT> hostData(new HostDiscPT(*(deviceData.first)));
                    cudaEvent_t event;
                    cudaEventCreate(&event);
                    cudaEventRecord(event, 0);
                    m_ongoingTransfers.emplace(std::move(deviceData.first), std::move(hostData), deviceData.second,
                                               event);
                }
            }

            // handle finished memory transfers if any
            if(!m_ongoingTransfers.empty() && cudaEventQuery(m_ongoingTransfers.front().event) != cudaErrorNotReady)
            {
                OngoingTransfer transfer = std::move(m_ongoingTransfers.front());
                m_ongoingTransfers.pop();

                std::lock_guard<std::mutex> ddc_lck(m_ddcMutex);
                m_deviceDiskCopy.emplace(std::move(transfer.target),transfer.time);
            }

            // put data from cpu to files in memory
            {
                std::unique_lock<std::mutex> ddc_lck(m_ddcMutex);
                if(!m_deviceDiskCopy.empty())
                {
                    ddcQueueType hostData = std::move(m_deviceDiskCopy.front());
                    m_deviceDiskCopy.pop();
                    ddc_lck.unlock();

                    printTextFile(std::move(hostData));
                    m_numberJobs--;
                    logDEBUG("ResultStorageManager") << "Results stored for t= " << hostData.second;
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

void ResultStorageManager::printTextFile(ResultStorageManager::ddcQueueType data)
{
    std::string filename = m_directory + m_prefix + m_startTime + "_" + mpu::toString(data.second)+".tsv";
    std::ofstream file(filename);

    if(!file.is_open())
    {
        logERROR("ResultStorageManager") << "Could not open output file: " << filename << " Make sure the path actually exists.";
        logFlush();
        throw std::runtime_error("Could not open output file.");
    }

    for(int i = 0; i < data.first->size(); ++i)
    {
        auto p = data.first->loadParticle<DiscPbases>(i);

        file << p.pos.x << "\t"
             << p.pos.y << "\t"
             << p.pos.z << "\t"
             << p.vel.x << "\t"
             << p.vel.y << "\t"
             << p.vel.z << "\t"
             << p.mass << "\t"
             << p.density << "\n";
    }

    if(file.fail())
    {
        logERROR("ResultStorageManager") << "Error writing output file: " << filename;
        logFlush();
        throw std::runtime_error("Could not write to output file.");
    }
}
