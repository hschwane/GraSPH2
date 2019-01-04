/*
 * mpUtils
 * algorithms.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */
#ifndef MPUTILS_ALGORITHMS_H
#define MPUTILS_ALGORITHMS_H

// includes
//--------------------
#include <mpUtils/mpUtils.h>
#include <mpUtils/mpCuda.h>
#include "partice_attributes.h"
#include "particle_buffer_impl.h"
#include "Particle.h"
#include "HostParticleBuffer.h"
#include "DeviceParticleBuffer.h"
#include "DeviceParticleReference.h"
#include "SharedParticles.h"
//--------------------

//-------------------------------------------------------------------
// template functions to execute some common algorithms on particles

/**
 * @brief Allows to execute the function specified by job on each particle in the HostParticleBuffer pb on the Host. (Pass a DeviceParticleBuffer to execute on the device)
 * @tparam job the type of job struct that specifies what function to execute
 * @tparam blockSize the cuda block size when executing on the device
 * @tparam particleBuffer the type of HostParticleBuffer on which the function is executed
 * @param pb the HostParticleBuffer on which the function is executed
 */
template<typename job, size_t blockSize=0, typename hostBuffer, typename ... Ts, std::enable_if_t< mpu::is_instantiation_of< HostParticleBuffer,hostBuffer>::value, int> =0>
void do_for_each(hostBuffer& pb, Ts...args);

/**
 * @brief Allows to execute the function specified by job on each particle in the DeviceParticleBuffer pb on the device. (Pass a HostParticleBuffer to execute on the host)
 * @tparam job the type of job struct that specifies what function to execute
 * @tparam blockSize the cuda block size when executing on the device
 * @tparam particleBuffer the type of DeviceParticleBuffer on which the function is executed
 * @param pb the DeviceParticleBuffer on which the function is executed
 */
template<typename job, size_t blockSize=256, typename deviceBuffer, typename ... Ts, std::enable_if_t< mpu::is_instantiation_of< DeviceParticleBuffer,deviceBuffer>::value, int> =0>
void do_for_each(deviceBuffer& pb, Ts...args);

/**
 * @brief Allows to execute the function specified by job on each pair of particles in the HostParticleBuffer pb on the Host. (Pass a DeviceParticleBuffer to execute on the device)
 * @tparam job the type of job struct that specifies what function to execute
 * @tparam blockSize the cuda block size when executing on the device
 * @tparam particleBuffer the type of HostParticleBuffer on which the function is executed
 * @param pb the HostParticleBuffer on which the function is executed
 */
template<typename job, size_t blockSize=0, typename hostBuffer, typename ... Ts, std::enable_if_t< mpu::is_instantiation_of< HostParticleBuffer,hostBuffer>::value, int> =0>
void do_for_each_pair(hostBuffer& pb, Ts...args);

/**
 * @brief Allows to execute the function specified by job on each pair of particles in the DeviceParticleBuffer pb on the device. (Pass a HostParticleBuffer to execute on the host)
 * @tparam job the type of job struct that specifies what function to execute
 * @tparam blockSize the cuda block size when executing on the device
 * @tparam particleBuffer the type of DeviceParticleBuffer on which the function is executed
 * @param pb the DeviceParticleBuffer on which the function is executed
 */
template<typename job, size_t blockSize=256, typename deviceBuffer, typename ... Ts, std::enable_if_t< mpu::is_instantiation_of< DeviceParticleBuffer,deviceBuffer>::value, int> =0>
void do_for_each_pair(deviceBuffer& pb, Ts...args);

/**
 * @brief Same as do_for_each_pair. Only exists for compatibility
 * @tparam job the type of job struct that specifies what function to execute
 * @tparam blockSize the cuda block size when executing on the device
 * @tparam particleBuffer the type of HostParticleBuffer on which the function is executed
 * @param pb the HostParticleBuffer on which the function is executed
 */
template<typename job, size_t blockSize=0, typename hostBuffer, typename ... Ts, std::enable_if_t< mpu::is_instantiation_of< HostParticleBuffer,hostBuffer>::value, int> =0>
void do_for_each_pair_fast(hostBuffer& pb, Ts...args);

/**
 * @brief Allows to execute the function specified by job on each pair of particles in the DeviceParticleBuffer pb on the device. (Pass a HostParticleBuffer to execute on the host)
 *          Can only be used if job has using declaration "shared" with shared memory to store particle j in, shared memory optimization is used during the interaction.
 * @tparam job the type of job struct that specifies what function to execute
 * @tparam blockSize the cuda block size when executing on the device
 * @tparam particleBuffer the type of DeviceParticleBuffer on which the function is executed
 * @param pb the DeviceParticleBuffer on which the function is executed
 */
template<typename job, size_t blockSize=256, typename deviceBuffer, typename ... Ts, std::enable_if_t< mpu::is_instantiation_of< DeviceParticleBuffer,deviceBuffer>::value, int> =0>
void do_for_each_pair_fast(deviceBuffer& pb, Ts...args);

//-------------------------------------------------------------------
// template functions definitions

namespace detail {

    template<typename ParticleType, typename buffer>
    struct load_helper;

    template<typename ...Args, typename buffer>
    struct load_helper<Particle<Args...>, buffer>
    {
        CUDAHOSTDEV static auto load(const buffer &pb, size_t i)
        {
            return pb.template loadParticle<Args...>(i);
        }
    };

    template<typename job, typename deviceReference, typename ... Ts>
    __global__ void do_for_each_impl(deviceReference pb, Ts ... args)
    {
        for(const auto &i : mpu::gridStrideRange(pb.size()))
        {
            auto pi = load_helper<typename job::load_type, deviceReference>::load(pb, i);

            job job_i(args...);
            auto result = job_i.do_for_each(pi, i);

            pb.storeParticle(i, result);
        }
    }

}

template<typename job, size_t blockSize, typename hostBuffer, typename ... Ts, std::enable_if_t< mpu::is_instantiation_of< HostParticleBuffer,hostBuffer>::value, int>>
void do_for_each(hostBuffer& pb, Ts...args)
{
    #pragma omp parallel for
    for(int i = 0; i < pb.size(); i++)
    {
        job job_i(args...);

        typename job::pi_type pi{};
        pi = detail::load_helper< typename job::load_type, hostBuffer>::load(pb,i);

        typename job::store_type result = job_i.do_for_each( pi, i);
        pb.storeParticle( i, result);
    }
}

template<typename job, size_t blockSize, typename deviceBuffer, typename ... Ts, std::enable_if_t< mpu::is_instantiation_of< DeviceParticleBuffer,deviceBuffer>::value, int>>
void do_for_each(deviceBuffer& pb, Ts...args)
{
    detail::do_for_each_impl<job><<< mpu::numBlocks(pb.size(),blockSize), blockSize>>>( pb.getDeviceReference(), args...);
}

namespace detail {

    template<typename job, typename deviceReference, typename ... Ts >
    __global__ void do_for_each_pair_impl(deviceReference pb, Ts ... args)
    {
        for(const auto &i : mpu::gridStrideRange(pb.size()))
        {
            job job_i(args...);

            typename job::pi_type pi{};
            pi = detail::load_helper< typename job::load_type, deviceReference>::load(pb,i);
            job_i.do_before( pi, i);

            for(int j = 0; j < pb.size(); j++)
            {
                const auto pj = detail::load_helper< typename job::pj_type, deviceReference>::load(pb,j);
                job_i.do_for_each_pair( pi, pj);
            }

            typename job::store_type result = job_i.do_after( pi);
            pb.storeParticle( i, result);
        }
    }
}

template<typename job, size_t blockSize, typename hostBuffer, typename ... Ts, std::enable_if_t< mpu::is_instantiation_of< HostParticleBuffer,hostBuffer>::value, int>>
void do_for_each_pair(hostBuffer& pb, Ts...args)
{
    #pragma omp parallel for
    for(int i = 0; i < pb.size(); i++)
    {
        job job_i(args...);

        typename job::pi_type pi{};
        pi = detail::load_helper< typename job::load_type, hostBuffer>::load(pb,i);
        job_i.do_before( pi, i);

        for(int j = 0; j < pb.size(); j++)
        {
            const auto pj = detail::load_helper< typename job::pj_type, hostBuffer>::load(pb,j);
            job_i.do_for_each_pair( pi, pj);
        }

        typename job::store_type result = job_i.do_after( pi);
        pb.storeParticle( i, result);
    }
}

template<typename job, size_t blockSize, typename deviceBuffer, typename ... Ts, std::enable_if_t< mpu::is_instantiation_of< DeviceParticleBuffer,deviceBuffer>::value, int>>
void do_for_each_pair(deviceBuffer& pb, Ts...args)
{
    detail::do_for_each_pair_impl<job><<< mpu::numBlocks(pb.size(),blockSize), blockSize>>>( pb.getDeviceReference(), args...);
}

namespace detail {

    template <typename T>
    using shared_detector_t = typename T::template shared<0>::particleType;

    template<typename job, size_t tileSize, typename deviceReference, typename ... Ts >
    __global__ void do_for_each_pair_fast_impl(deviceReference pb, Ts ... args)
    {
        using SharedType = typename job::template shared<tileSize>;
        SharedType shared;
        const int remain = pb.size() % tileSize;
        for(const auto &i : mpu::gridStrideRange( pb.size()))
        {
            job job_i(args...);
            typename job::pi_type pi{};
            pi = detail::load_helper< typename job::load_type, deviceReference>::load(pb,i);
            job_i.do_before( pi, i);

            const int thisTileSize = (i < pb.size() - remain) ? tileSize : remain;
            const int numTiles = (pb.size() + thisTileSize - 1) / thisTileSize;
            for (int tile = 0; tile < numTiles; tile++)
            {
                int loadIndex = tile*tileSize+threadIdx.x;
                if( loadIndex < pb.size())
                {
                    const auto p = detail::load_helper< typename job::pj_type, deviceReference>::load(pb,loadIndex);
                    shared.storeParticle(threadIdx.x,p);
                }
                __syncthreads();

                for(int j = 0; j < thisTileSize && (j+remain < thisTileSize || tile < numTiles); j++)
                {
                    const auto pj = detail::load_helper< typename job::pj_type, SharedType>::load(shared,loadIndex);
                    job_i.do_for_each_pair( pi, pj);
                }
                __syncthreads();
            }

            typename job::store_type result = job_i.do_after( pi);
            pb.storeParticle( i, result);
        }
    }
}

template<typename job, size_t blockSize, typename hostBuffer, typename ... Ts, std::enable_if_t< mpu::is_instantiation_of< HostParticleBuffer,hostBuffer>::value, int>>
void do_for_each_pair_fast(hostBuffer& pb, Ts...args)
{
    do_for_each_pair<job>(pb, std::forward<Ts>(args)...);
}

template<typename job, size_t blockSize, typename deviceBuffer, typename ... Ts, std::enable_if_t< mpu::is_instantiation_of< DeviceParticleBuffer,deviceBuffer>::value, int>>
void do_for_each_pair_fast(deviceBuffer& pb, Ts...args)
{
    static_assert( mpu::is_detected<detail::shared_detector_t, job>::value,
                   R"(Job does not specify a type "shared", or shared is not an instantiation of SharedParticles.
                        In order to use do_for_each_pair_fast on the GPU "job" needs to
                        specify a type "shared" which is a instantiation of SharedParticles and fits an entire particle j.
                        See the job example in algorithms.h.)");

    static_assert( std::is_same< typename job::template shared<0>::particleType, typename job::pj_type>::value,
                   R"(Job does specify a type "shared", but its particleType is not equal to pj_type. In order to use do_for_each_pair_fast on the GPU "job" needs to
                        specify a type "shared" which is a instantiation of SharedParticles and fits an entire particle j.
                        See the job example in algorithms.h.)");

    detail::do_for_each_pair_fast_impl<job,blockSize><<< mpu::numBlocks(pb.size(),blockSize), blockSize>>>( pb.getDeviceReference(), args...);
}

#endif //MPUTILS_ALGORITHMS_H
