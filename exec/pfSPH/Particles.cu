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
#include "Particles.h"
//--------------------

// namespace
//--------------------

//--------------------

// function definitions of the Particles class
//-------------------------------------------------------------------
Particles::Particles() : m_numParticles(0), m_hpos(nullptr), m_hvel(nullptr), m_hacc(nullptr), m_dpos(nullptr),
                         m_dvel(nullptr), m_dacc(nullptr)
{
}

Particles::Particles(size_t n) : Particles()
{
    reallocate(n);
}

Particles::~Particles()
{
    free();
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

    assert_cuda(cudaFree(m_dpos));
    m_dpos = nullptr;
    assert_cuda(cudaFree(m_dvel));
    m_dvel = nullptr;
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
