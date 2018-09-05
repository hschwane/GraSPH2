/*
 * mpUtils
 * matrix3x3.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */
#ifndef MPUTILS_MATRIX3X3_H
#define MPUTILS_MATRIX3X3_H

// includes
//--------------------
#include <iostream>
#include <type_traits>
#include <sstream>
#include "../type_traitUtils.h"
#ifdef USE_GLM
    #include <glm/glm.hpp>
#endif
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
 * Class Template Mat
 * This is a class template for small matrices to be used with cuda on host or device.
 * On host, glm matrix will be faster due to vectorization.
 * @tparam T the internal data type needs to be an arithmetic type
 * @tparam rows number of rows
 * @tparam cols number of columns
 */
template<typename T, size_t rows, size_t cols>
class Mat
{
    static_assert(std::is_arithmetic<T>::value, "Non arithmetic type used for matrix.");
public:
    // default constructors
    Mat() = default;

    // additional construction
    CUDAHOSTDEV explicit Mat(const T v); //!< constructor fills the diagonal with v

    template <typename... cArgs, std::enable_if_t< (sizeof...(cArgs) > 1) && (sizeof...(cArgs) == rows*cols), int> = 0>
    CUDAHOSTDEV explicit Mat(const cArgs... v) : m_data{static_cast<T>(v)...} {} //!< constructs matrix with a value for each element


#ifdef USE_GLM
    // conversion to glm
    template<glm::qualifier Q>
    explicit Mat(glm::mat<rows, cols, T, Q> &glmat); //!< constructs this from glm matrix
    template<glm::qualifier Q>
    explicit operator glm::mat<rows, cols, T, Q>(); //!< convert to glm matrix
#endif

    // data access
    CUDAHOSTDEV T *operator[](size_t row) { return &m_data[cols * row]; } //!< access a row
    CUDAHOSTDEV const T *operator[](size_t row) const { return &m_data[cols * row]; } //!< access a row

    CUDAHOSTDEV T &operator()(size_t idx) { return m_data[idx]; } //!< access value
    CUDAHOSTDEV const T &operator()(size_t idx) const { return m_data[idx]; } //!< access value

    // logical operators
    CUDAHOSTDEV bool operator==(const Mat &other) const;
    CUDAHOSTDEV bool operator!=(const Mat &other) const;

    // arithmetic operators
    CUDAHOSTDEV Mat &operator+=(const Mat &other); //!< component wise addition
    CUDAHOSTDEV Mat &operator-=(const Mat &other); //!< component wise subtraction
    CUDAHOSTDEV Mat operator+(const Mat &other) const; //!< component wise addition
    CUDAHOSTDEV Mat operator-(const Mat &other) const; //!< component wise subtraction

    CUDAHOSTDEV Mat &operator*=(const T &v); //!< scalar multiply
    CUDAHOSTDEV Mat &operator/=(const T &v); //!< scalar divide
    CUDAHOSTDEV Mat operator*(const T &v) const; //!< scalar multiply
    CUDAHOSTDEV Mat operator/(const T &v) const; //!< scalar divide

    CUDAHOSTDEV Mat &operator*=(const Mat &rhs); //!< matrix multiplication
    template<size_t rhsRows, size_t rhsCols>
    CUDAHOSTDEV Mat<T, rows, rhsCols> operator*(const Mat<T, rhsRows, rhsCols> &rhs) const; //!< matrix multiplication

    static constexpr size_t size = rows * cols;
private:
    T m_data[size];
};

/**
 * @brief calculate the transpose of matrix m
 */
template<typename T, size_t rows, size_t cols>
CUDAHOSTDEV Mat<T, cols, rows> transpose(Mat<T, rows, cols> &m);

/**
 * @brief performes component wise multiplication of two matrices of same size
 */
template<typename T, size_t rows, size_t cols>
CUDAHOSTDEV Mat<T, rows, cols> compWiseMult(Mat<T, rows, cols> &first, Mat<T, rows, cols> &second);

/**
 * @brief calculates the inverse matrix undefined if determinant is zero
 */
template <typename T>
CUDAHOSTDEV Mat<T,2,2> invert(const Mat<T,2,2> &m);

// /**
//  * @brief calculates the inverse matrix undefined if determinant is zero
//  */
//template <typename T>
//Mat<T,3,3> invert(Mat<T,3,3> &m);

/**
 * @brief calculates the inverse matrix undefined if determinant is zero
 */
template <typename T>
CUDAHOSTDEV Mat<T,4,4> invert(const Mat<T,4,4> &m);

// helper to check if T has a x attribute
namespace detail {
    template<class T>
    using hasx_t = decltype(std::declval<T>().x);
}

/**
 * @brief scalar multiplication is order independent
 */
template<typename T, size_t rows, size_t cols>
CUDAHOSTDEV Mat<T, rows, cols> operator*(const T& lhs, const Mat<T,rows,cols>& rhs);

/**
 * @brief multiply a 2D vector with a 2x2 matrix
 */
template<typename T, typename vT, std::enable_if_t<!std::is_same<T,vT>::value && mpu::is_detected<detail::hasx_t,vT>(), int> = 0>
CUDAHOSTDEV vT operator*(Mat<T, 2, 2> lhs, vT &rhs);

/**
 * @brief multiply a 3D vector with a 3x3 matrix
 */
template<typename T, typename vT, std::enable_if_t<!std::is_same<T,vT>::value && mpu::is_detected<detail::hasx_t,vT>(), int> = 0>
CUDAHOSTDEV vT operator*(Mat<T, 3, 3> lhs, vT &rhs);

/**
 * @brief multiply a 4D vector with a 4x4 matrix
 */
template<typename T, typename vT, std::enable_if_t<!std::is_same<T,vT>::value && mpu::is_detected<detail::hasx_t,vT>(), int> = 0>
CUDAHOSTDEV vT operator*(Mat<T, 4, 4> lhs, vT &rhs);

/**
 * @brief convert a matrix to string for debugging
 */
template<typename T, size_t rows, size_t cols>
std::string toString(Mat<T,rows,cols>& mat);

// define all the template functions of the matrix class
//-------------------------------------------------------------------

template<typename T, size_t rows, size_t cols>
CUDAHOSTDEV Mat<T, rows, cols>::Mat(const T v)
{
    for(int i = 0; i < rows; i++)
        for(int j = 0; j < cols; j++)
        {
            if(i == j)
                (*this)[i][j] = v;
            else
                (*this)[i][j] = 0;
        }
}

#ifdef USE_GLM
template<typename T, size_t rows, size_t cols>
template<glm::qualifier Q>
Mat<T, rows, cols>::Mat(glm::mat<rows, cols, T, Q> &glmat)
{
    for(int i = 0; i < rows; i++)
        for(int j = 0; j < cols; j++)
        {
            (*this)[i][j] = glmat[i][j];
        }
}

template<typename T, size_t rows, size_t cols>
template<glm::qualifier Q>
Mat<T, rows, cols>::operator glm::mat<rows, cols, T, Q>()
{
    glm::mat<rows, cols, T, Q> r;

    for(int i = 0; i < rows; i++)
        for(int j = 0; j < cols; j++)
        {
            r[i][j] = (*this)[i][j];
        }

    return r;
}
#endif

template<typename T, size_t rows, size_t cols>
CUDAHOSTDEV bool Mat<T, rows, cols>::operator==(const Mat &other) const
{
    for(int i = 0; i < size; ++i)
    {
        if(m_data[i] != other.m_data[i])
            return false;
    }
    return true;
}

template<typename T, size_t rows, size_t cols>
CUDAHOSTDEV bool Mat<T, rows, cols>::operator!=(const Mat &other) const
{
    return !((*this) == other);
}

template<typename T, size_t rows, size_t cols>
CUDAHOSTDEV Mat<T, rows, cols> &Mat<T, rows, cols>::operator+=(const Mat &other)
{
    for(int i = 0; i < size; ++i)
    {
        m_data[i] += other.m_data[i];
    }
    return *this;
}

template<typename T, size_t rows, size_t cols>
CUDAHOSTDEV Mat<T, rows, cols> &Mat<T, rows, cols>::operator-=(const Mat &other)
{
    for(int i = 0; i < size; ++i)
    {
        m_data[i] -= other.m_data[i];
    }
    return *this;
}

template<typename T, size_t rows, size_t cols>
CUDAHOSTDEV Mat<T, rows, cols> Mat<T, rows, cols>::operator+(const Mat &other) const
{
    Mat<T, rows, cols> temp(*this);
    temp += other;
    return temp;
}

template<typename T, size_t rows, size_t cols>
CUDAHOSTDEV Mat<T, rows, cols> Mat<T, rows, cols>::operator-(const Mat &other) const
{
    Mat<T, rows, cols> temp(*this);
    temp -= other;
    return temp;
}

template<typename T, size_t rows, size_t cols>
CUDAHOSTDEV Mat<T, rows, cols> &Mat<T, rows, cols>::operator*=(const T &v)
{
    for(int i = 0; i < size; ++i)
    {
        m_data[i] *= v;
    }
    return *this;
}

template<typename T, size_t rows, size_t cols>
CUDAHOSTDEV Mat<T, rows, cols> &Mat<T, rows, cols>::operator/=(const T &v)
{
    for(int i = 0; i < size; ++i)
    {
        m_data[i] /= v;
    }
    return *this;
}

template<typename T, size_t rows, size_t cols>
CUDAHOSTDEV Mat<T, rows, cols> Mat<T, rows, cols>::operator*(const T &v) const
{
    Mat<T, rows, cols> temp(*this);
    temp *= v;
    return temp;
}

template<typename T, size_t rows, size_t cols>
CUDAHOSTDEV Mat<T, rows, cols> Mat<T, rows, cols>::operator/(const T &v) const
{
    Mat<T, rows, cols> temp(*this);
    temp /= v;
    return temp;
}

template<typename T, size_t rows, size_t cols>
CUDAHOSTDEV Mat<T, rows, cols> &Mat<T, rows, cols>::operator*=(const Mat &rhs)
{
    Mat tmp(*this);

    for(int i = 0; i < rows; ++i)
        for(int k = 0; k < cols; ++k)
        {
            (*this)[i][k] = tmp[i][0] * rhs[0][k];
            for(int j = 1; j < cols; ++j)
            {
                (*this)[i][k] += tmp[i][j] * rhs[j][k];
            }
        }

    return *this;
}

template<typename T, size_t rows, size_t cols>
template<size_t rhsRows, size_t rhsCols>
CUDAHOSTDEV Mat<T, rows, rhsCols> Mat<T, rows, cols>::operator*(const Mat<T, rhsRows, rhsCols> &rhs) const
{
    static_assert(cols == rhsRows, "Matrices of these sizes can not be multiplied.");
    Mat<T, rows, rhsCols> result(0);

    for(int i = 0; i < rows; ++i)
        for(int k = 0; k < rhsCols; ++k)
        {
            result[i][k] = (*this)[i][0] * rhs[0][k];
            for(int j = 1; j < cols; ++j)
            {
                result[i][k] += (*this)[i][j] * rhs[j][k];
            }
        }

    return result;
}

// define all the helper functions
//-------------------------------------------------------------------

template<typename T, size_t rows, size_t cols>
CUDAHOSTDEV Mat<T, cols, rows> transpose(Mat<T, rows, cols> &m)
{
    Mat<T, cols, rows> result;
    for(int i = 0; i < rows; i++)
        for(int j = 0; j < cols; j++)
            result[j][i] = m[i][j];
    return result;
}

template<typename T, size_t rows, size_t cols>
CUDAHOSTDEV Mat<T, rows, cols> compWiseMult(Mat<T, rows, cols> &first, Mat<T, rows, cols> &second)
{
    Mat<T,rows,cols> r;

    for(int i = 0; i < r.size; ++i)
        r(i) = first(i)*second(i);

    return r;
}

template<typename T>
CUDAHOSTDEV Mat<T, 2, 2> invert(Mat<T, 2, 2> &m)
{
    Mat<T,2,2> r;

    r(0) = m(3);
    r(1) = -m(1);
    r(2) = -m(2);
    r(3) = m(0);

    T det = m(0)*m(3) - m(1)*m(2);
    det = 1.0/det;

    r(0) *= det;
    r(1) *= det;
    r(2) *= det;
    r(3) += det;

    return r;
}

template<typename T>
CUDAHOSTDEV Mat<T, 4, 4> invert(Mat<T, 4, 4> &m)
{
    Mat<T, 4, 4> inv;

    inv(0) = m(5)  * m(10) * m(15) -
             m(5)  * m(11) * m(14) -
             m(9)  * m(6)  * m(15) +
             m(9)  * m(7)  * m(14) +
             m(13) * m(6)  * m(11) -
             m(13) * m(7)  * m(10);

    inv(4) = -m(4)  * m(10) * m(15) +
             m(4)  * m(11) * m(14) +
             m(8)  * m(6)  * m(15) -
             m(8)  * m(7)  * m(14) -
             m(12) * m(6)  * m(11) +
             m(12) * m(7)  * m(10);

    inv(8) = m(4)  * m(9) * m(15) -
             m(4)  * m(11) * m(13) -
             m(8)  * m(5) * m(15) +
             m(8)  * m(7) * m(13) +
             m(12) * m(5) * m(11) -
             m(12) * m(7) * m(9);

    inv(12) = -m(4)  * m(9) * m(14) +
              m(4)  * m(10) * m(13) +
              m(8)  * m(5) * m(14) -
              m(8)  * m(6) * m(13) -
              m(12) * m(5) * m(10) +
              m(12) * m(6) * m(9);

    inv(1) = -m(1)  * m(10) * m(15) +
             m(1)  * m(11) * m(14) +
             m(9)  * m(2) * m(15) -
             m(9)  * m(3) * m(14) -
             m(13) * m(2) * m(11) +
             m(13) * m(3) * m(10);

    inv(5) = m(0)  * m(10) * m(15) -
             m(0)  * m(11) * m(14) -
             m(8)  * m(2) * m(15) +
             m(8)  * m(3) * m(14) +
             m(12) * m(2) * m(11) -
             m(12) * m(3) * m(10);

    inv(9) = -m(0)  * m(9) * m(15) +
             m(0)  * m(11) * m(13) +
             m(8)  * m(1) * m(15) -
             m(8)  * m(3) * m(13) -
             m(12) * m(1) * m(11) +
             m(12) * m(3) * m(9);

    inv(13) = m(0)  * m(9) * m(14) -
              m(0)  * m(10) * m(13) -
              m(8)  * m(1) * m(14) +
              m(8)  * m(2) * m(13) +
              m(12) * m(1) * m(10) -
              m(12) * m(2) * m(9);

    inv(2) = m(1)  * m(6) * m(15) -
             m(1)  * m(7) * m(14) -
             m(5)  * m(2) * m(15) +
             m(5)  * m(3) * m(14) +
             m(13) * m(2) * m(7) -
             m(13) * m(3) * m(6);

    inv(6) = -m(0)  * m(6) * m(15) +
             m(0)  * m(7) * m(14) +
             m(4)  * m(2) * m(15) -
             m(4)  * m(3) * m(14) -
             m(12) * m(2) * m(7) +
             m(12) * m(3) * m(6);

    inv(10) = m(0)  * m(5) * m(15) -
              m(0)  * m(7) * m(13) -
              m(4)  * m(1) * m(15) +
              m(4)  * m(3) * m(13) +
              m(12) * m(1) * m(7) -
              m(12) * m(3) * m(5);

    inv(14) = -m(0)  * m(5) * m(14) +
              m(0)  * m(6) * m(13) +
              m(4)  * m(1) * m(14) -
              m(4)  * m(2) * m(13) -
              m(12) * m(1) * m(6) +
              m(12) * m(2) * m(5);

    inv(3) = -m(1) * m(6) * m(11) +
             m(1) * m(7) * m(10) +
             m(5) * m(2) * m(11) -
             m(5) * m(3) * m(10) -
             m(9) * m(2) * m(7) +
             m(9) * m(3) * m(6);

    inv(7) = m(0) * m(6) * m(11) -
             m(0) * m(7) * m(10) -
             m(4) * m(2) * m(11) +
             m(4) * m(3) * m(10) +
             m(8) * m(2) * m(7) -
             m(8) * m(3) * m(6);

    inv(11) = -m(0) * m(5) * m(11) +
              m(0) * m(7) * m(9) +
              m(4) * m(1) * m(11) -
              m(4) * m(3) * m(9) -
              m(8) * m(1) * m(7) +
              m(8) * m(3) * m(5);

    inv(15) = m(0) * m(5) * m(10) -
              m(0) * m(6) * m(9) -
              m(4) * m(1) * m(10) +
              m(4) * m(2) * m(9) +
              m(8) * m(1) * m(6) -
              m(8) * m(2) * m(5);

    T det = m(0) * inv(0) + m(1) * inv(4) + m(2) * inv(8) + m(3) * inv(12);
    det = 1.0 / det;

    for (int i = 0; i < 16; i++)
        inv(i) = inv(i) * det;

    return inv;
}

template<typename T, size_t rows, size_t cols>
CUDAHOSTDEV Mat<T, rows, cols> operator*(const T &lhs, const Mat<T, rows, cols>& rhs)
{
    return rhs*lhs;
}

template<typename T, typename vT, std::enable_if_t<!std::is_same<T,vT>::value && mpu::is_detected<detail::hasx_t,vT>(), int>>
CUDAHOSTDEV vT operator*(Mat<T, 2, 2> lhs, vT &rhs)
{
    return vT{lhs(0) * rhs.x + lhs(1) * rhs.y,
              lhs(2) * rhs.x + lhs(3) * rhs.y};
}

template<typename T, typename vT, std::enable_if_t<!std::is_same<T,vT>::value && mpu::is_detected<detail::hasx_t,vT>(), int>>
CUDAHOSTDEV vT operator*(Mat<T, 3, 3> lhs, vT &rhs)
{
    return vT{lhs(0) * rhs.x + lhs(1) * rhs.y + lhs(2) * rhs.z,
              lhs(3) * rhs.x + lhs(4) * rhs.y + lhs(5) * rhs.z,
              lhs(6) * rhs.x + lhs(7) * rhs.y + lhs(8) * rhs.z};
}

template<typename T, typename vT, std::enable_if_t<!std::is_same<T,vT>::value && mpu::is_detected<detail::hasx_t,vT>(), int>>
CUDAHOSTDEV vT operator*(Mat<T, 4, 4> lhs, vT &rhs)
{
    return vT{lhs(0) * rhs.x + lhs(1) * rhs.y + lhs(2) * rhs.z + lhs(3) * rhs.z,
              lhs(4) * rhs.x + lhs(5) * rhs.y + lhs(6) * rhs.z + lhs(7) * rhs.z,
              lhs(8) * rhs.x + lhs(9) * rhs.y + lhs(10) * rhs.z + lhs(11) * rhs.z,
              lhs(12) * rhs.x + lhs(13) * rhs.y + lhs(14) * rhs.z + lhs(15) * rhs.z};
}

template<typename T, size_t rows, size_t cols>
std::string toString(Mat<T,rows,cols>& mat)
{
    std::ostringstream ss;
    for(int i = 0; i < rows; ++i)
    {
        ss << "| " << mat[i][0];
        for(int j = 1; j < cols; ++j)
        {
            ss << ",  " << mat[i][j];
        }
        ss << " |\n";
    }
    return ss.str();
}

}

#endif //MPUTILS_MATRIX3X3_H
