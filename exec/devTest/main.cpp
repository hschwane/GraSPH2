/*
 * mpUtils
 * main.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail: hendrik.schwanekamp@gmx.net
 *
 * mpUtils = my personal Utillities
 * A utility library for my personal c++ projects
 *
 * Copyright 2016 Hendrik Schwanekamp
 *
 */

#include <mpUtils.h>
#include <Graphics/Graphics.h>
#include <chrono>
#include <Cuda/Matrix.h>

using namespace mpu;
using namespace std;
using namespace std::chrono;

int main()
{

    Mat<float,3,3> m(5);
    Mat<float,3,3> m2(5);

    auto m3 = m * m2;

    return 0;
}