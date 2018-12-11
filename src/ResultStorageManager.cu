/*
 * GraSPH2
 * ResultStorageManager.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the ResultStorageManager class
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

ResultStorageManager::ResultStorageManager(std::string directory, std::string prefix) : m_terminateWorker(false)
{
    m_workerThread = std::thread(&ResultStorageManager::worker, this);
}

void ResultStorageManager::worker()
{
    std::unique_lock<std::mutex> lck(m_workerMutex);
    while(!m_terminateWorker)
    {
        // wait until there is work to do
        m_workSignal.wait(lck);

        std::unique_lock<std::mutex> hdc_lck(m_hdcMutex);
        std::unique_lock<std::mutex> ddc_lck(m_ddcMutex);
        while(!m_hostDeviceCopy.empty() || !m_ongoingTransfers.empty() || !m_deviceDiskCopy.empty())
        {
            ddc_lck.unlock();

            // download data from gpu to cpu if any
            // hdc_lck is already locked
            if(!m_hostDeviceCopy.empty())
            {
                hdcQueueType deviceData = std::move(m_hostDeviceCopy.front());
                m_hostDeviceCopy.pop();
                hdc_lck.unlock();

                std::unique_ptr<HostDiscPT> hostData(new HostDiscPT(*(deviceData.first)));
                cudaEvent_t event;
                cudaEventCreate(&event);
                cudaEventRecord(event, 0);
                m_ongoingTransfers.emplace(std::move(deviceData.first), std::move(hostData), deviceData.second, event);
            }
            hdc_lck.unlock();

            // handle finished memory transfers if any
            if(!m_ongoingTransfers.empty() && cudaEventQuery(m_ongoingTransfers.front().event) != cudaErrorNotReady)
            {
                OngoingTransfer transfer = std::move(m_ongoingTransfers.front());
                m_ongoingTransfers.pop();

                ddc_lck.lock();
                m_deviceDiskCopy.emplace(std::move(transfer.target),transfer.time);
            }

            // put data from cpu to memory
            ddc_lck.lock();
            if(!m_deviceDiskCopy.empty())
            {
                ddcQueueType hostData = std::move(m_deviceDiskCopy.front());
                m_deviceDiskCopy.pop();
                ddc_lck.unlock();

                logINFO("ResultStorageManager") << "Results stored for t= " << hostData.second;
                // writeToFile(hostData);
            }
            ddc_lck.unlock();

            hdc_lck.lock();
            ddc_lck.lock();
        }
    }
}
