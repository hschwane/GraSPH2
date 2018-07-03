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
#include <chrono>
#include "cudaTest.h"

using namespace mpu;
using namespace std;
using namespace std::chrono;

int main()
{
    CpuStopwatch timer;

//        Log myLog(LogPolicy::CONSOLE, LogLvl::ALL);

        Log myLog( LogLvl::ALL, ConsoleSink());

        myLog(LogLvl::INFO, MPU_FILEPOS , "TEST") << "Hi, a log";

        logINFO("TEST") << "Testing Cuda.";
        testCuda();

    logINFO("TEST") << "It took me " << timer.getSeconds() << " seconds" << endl;
    myLog.close();
    return 0;
}