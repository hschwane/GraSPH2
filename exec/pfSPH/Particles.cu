/*
 * mpUtils
 * Particles.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the Particles class
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */

// includes
//--------------------
#include <cuda_gl_interop.h>
#include "Particles.h"
//--------------------

// namespace
//--------------------

//--------------------

// function definitions of the Particles class
//-------------------------------------------------------------------
Particles::Particles() : m_numParticles(0), m_hpos(nullptr), m_hvel(nullptr), m_hacc(nullptr), m_dpos(nullptr),
                         m_dvel(nullptr), m_dacc(nullptr), registeredPosBuffer(false), registeredVelBuffer(false),
                         m_isDeviceCopy(false)
{
    VBO_CUDA[0] = nullptr;
    VBO_CUDA[1] = nullptr;
}

Particles::Particles(size_t n) : Particles()
{
    reallocate(n);
}

Particles::~Particles()
{
    // if this is just a device copy do nothing
    if(m_isDeviceCopy)
        return;
    free();
    unmapRegisteredBuffes();
    unregisterBuffers();
}

void Particles::reallocate(size_t n)
{
    free();
    allocate(n);
    for(int i=0; i<m_numParticles; i++)
    {
        m_hpos[i] = {0,0,0,0};
        m_hvel[i] = {0,0,0,0};
        m_hacc[i] = {0,0,0,0};
    }
}

void Particles::allocate(size_t n)
{
    m_numParticles = n;
    m_hpos = new f4_t[n];
    m_hvel = new f4_t[n];
    m_hacc = new f4_t[n];

    assert_cuda(cudaMalloc(&m_dpos, m_numParticles*sizeof(f4_t)));
    assert_cuda(cudaMalloc(&m_dvel, m_numParticles*sizeof(f4_t)));
    assert_cuda(cudaMalloc(&m_dacc, m_numParticles*sizeof(f4_t)));
}

void Particles::free()
{
    delete[] m_hpos;
    m_hpos = nullptr;
    delete[] m_hvel;
    m_hvel = nullptr;
    delete[] m_hacc;
    m_hacc = nullptr;

    if(!registeredPosBuffer)
    {
        assert_cuda(cudaFree(m_dpos));
        m_dpos = nullptr;
    }
    if(!registeredVelBuffer)
    {
        assert_cuda(cudaFree(m_dvel));
        m_dvel = nullptr;
    }
    assert_cuda(cudaFree(m_dacc));
    m_dacc = nullptr;
}

void Particles::copyToDevice()
{
    assert_cuda(cudaMemcpy(m_dpos, m_hpos, m_numParticles*sizeof(f4_t), cudaMemcpyHostToDevice));
    assert_cuda(cudaMemcpy(m_dvel, m_hvel, m_numParticles*sizeof(f4_t), cudaMemcpyHostToDevice));
    assert_cuda(cudaMemcpy(m_dacc, m_hacc, m_numParticles*sizeof(f4_t), cudaMemcpyHostToDevice));
}

void Particles::copyFromDevice()
{
    if(m_dpos) assert_cuda(cudaMemcpy(m_hpos, m_dpos, m_numParticles*sizeof(f4_t), cudaMemcpyDeviceToHost));
    if(m_dvel) assert_cuda(cudaMemcpy(m_hvel, m_dvel, m_numParticles*sizeof(f4_t), cudaMemcpyDeviceToHost));
    if(m_dpos) assert_cuda(cudaMemcpy(m_hacc, m_dacc, m_numParticles*sizeof(f4_t), cudaMemcpyDeviceToHost));
}

void Particles::registerGLPositionBuffer(uint32_t posBufferID)
{
    assert_cuda(cudaFree(m_dpos));
    m_dpos = nullptr;
    assert_cuda(cudaGraphicsGLRegisterBuffer(&VBO_CUDA[0], posBufferID, cudaGraphicsMapFlagsWriteDiscard));
    registeredPosBuffer = true;
}

void Particles::registerGLVelocityBuffer(uint32_t velBufferID)
{
    assert_cuda(cudaFree(m_dvel));
    m_dpos = nullptr;
    assert_cuda(cudaGraphicsGLRegisterBuffer(&VBO_CUDA[1], velBufferID, cudaGraphicsMapFlagsWriteDiscard));
    registeredVelBuffer = true;
}

void Particles::mapRegisteredBuffers()
{
    size_t mappedBufferSize;

    if(registeredPosBuffer)
    {
        assert_cuda(cudaGraphicsMapResources(1, &VBO_CUDA[0]));
        assert_cuda(cudaGraphicsResourceGetMappedPointer((void**)&m_dpos, &mappedBufferSize, VBO_CUDA[0]));
        assert_true(mappedBufferSize == m_numParticles*sizeof(f4_t), "Paticles",
                    "opengl buffer size is not equal to particle number");
    }
    if(registeredVelBuffer)
    {
        assert_cuda(cudaGraphicsMapResources(1, &VBO_CUDA[1]));
        assert_cuda(cudaGraphicsResourceGetMappedPointer((void**)&m_dvel, &mappedBufferSize, VBO_CUDA[1]));
        assert_true(mappedBufferSize == m_numParticles*sizeof(f4_t), "Paticles",
                    "opengl buffer size is not equal to particle number");
    }
}

void Particles::unmapRegisteredBuffes()
{
    if(registeredPosBuffer)
    {
        assert_cuda(cudaGraphicsUnmapResources(1, &VBO_CUDA[0]));
        m_dpos = nullptr;
    }
    if(registeredVelBuffer)
    {
        assert_cuda(cudaGraphicsUnmapResources(1, &VBO_CUDA[1]));
        m_dvel = nullptr;
    }
}

void Particles::unregisterBuffers()
{
    if(registeredPosBuffer)
    {
        cudaGraphicsUnregisterResource(VBO_CUDA[0]);
        m_dvel=nullptr;
        registeredPosBuffer = false;
    }
    if(registeredVelBuffer)
    {
        cudaGraphicsUnregisterResource(VBO_CUDA[1]);
        m_dvel=nullptr;
        registeredVelBuffer = false;
    }
}

Particles Particles::createDeviceClone() const
{
    Particles other;
    // only copy device pointer to the device copy of particles
    other.m_dpos = m_dpos;
    other.m_dvel = m_dvel;
    other.m_dacc = m_dacc;
    other.m_numParticles = m_numParticles;
    other.m_isDeviceCopy = true;
    return std::move(other);
}