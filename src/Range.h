/*
 * mpUtils
 * range.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */
#ifndef MPUTILS_RANGE_H
#define MPUTILS_RANGE_H

// includes
//--------------------
#include <iterator>
//--------------------

// this file contains device/host functions that also need to compile when using gcc
//--------------------
#ifndef CUDAHOSTDEV
    #ifdef __CUDACC__
        #define CUDAHOSTDEV __host__ __device__
    #else
        #define CUDAHOSTDEV
    #endif
#endif
//--------------------

// namespace
//--------------------
namespace mpu {
//--------------------

/**
 * template class Range
 *
 * @brief class to be used in for each loops for python style iteration
 * usage:
 * Generate a range using one of the constructors and use it in the for each loop.
 * If start is not specified it defaults to zero, if step is not specified it defaults to 1.
 * Using float values for the step is possible (see below).
 * @tparam T type to be used for iteration, needs to support +=, >=, <=, >, <
 */
template<typename T>
class Range
{
public:
    class iterator : std::iterator<std::forward_iterator_tag, T, T&, T*>
    {
    public:
        CUDAHOSTDEV T operator *() const { return  m_i;}
        CUDAHOSTDEV const iterator & operator ++() {m_i+=m_step; return *this;}
        CUDAHOSTDEV const iterator operator ++(int) {iterator copy(*this); m_i+=m_step; return copy;}

        CUDAHOSTDEV bool operator ==(const iterator& other) const {return m_isStepPoitive ? (m_i >= other.m_i) : (m_i <= other.m_i);}
        CUDAHOSTDEV bool operator !=(const iterator& other) const {return m_isStepPoitive ? (m_i < other.m_i) : (m_i > other.m_i);}

        CUDAHOSTDEV explicit iterator(T begin, T stride = 1) : m_i(begin), m_step(stride), m_isStepPoitive(stride>0) {}
    private:
        T m_i;
        const T m_step;
        const bool m_isStepPoitive;
    };

    using const_iterator=iterator;

    CUDAHOSTDEV explicit Range(const T& stop) : m_start(0), m_stop(stop), m_step(1) {}
    CUDAHOSTDEV explicit Range(const T& start,const T& stop,const T& step=1) : m_start(start), m_stop(stop), m_step(step) {}

    CUDAHOSTDEV iterator begin() { return iterator(m_start,m_step);}
    CUDAHOSTDEV iterator end() {return iterator(m_stop,m_step);}
    CUDAHOSTDEV const_iterator cbegin() const { return const_iterator(m_start,m_step);}
    CUDAHOSTDEV const_iterator cend() const {return const_iterator(m_stop,m_step);}

private:
    const T m_start;
    const T m_stop;
    const T m_step;
};

}

#endif //MPUTILS_RANGE_H
