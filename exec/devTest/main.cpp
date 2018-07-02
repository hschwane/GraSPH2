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

#include <stringUtils.h>
#include <Timer/Stopwatch.h>
#include <iostream>
#include <Log/Log.h>
#include <Log/ConsoleSink.h>
#include <Log/FileSink.h>
#include <Log/SyslogSink.h>
#include <chrono>

#include <typeinfo>
#include <ctime>
#include "Timer/DeltaTimer.h"
#include <thread>
#include "Timer/Timer.h"
#include "Timer/AsyncTimer.h"
#include <thread>             // std::thread
#include <mutex>              // std::mutex, std::unique_lock
#include <condition_variable> // std::condition_variable
#include <syslog.h>


using namespace mpu;
using namespace std;
using namespace std::chrono;

constexpr int numRuns = 10;
double dTime =0;

int main()
{
    CpuStopwatch timer;

//        Log myLog(LogPolicy::CONSOLE, LogLvl::ALL);

        Log myLog( LogLvl::ALL, ConsoleSink());

        myLog(LogLvl::INFO, MPU_FILEPOS , "TEST") << "Hi, a log";

        logINFO("TEST") << "Some generic Info";
        logWARNING("TEST") << "Some log warning";
        logERROR("MODULE_TEST") << "some stuff has happend";
        logDEBUG("some stuff") << "some stuff is debugging stuff";
        logDEBUG2("some stuff") << "more debugging stuff";

    timer.getSeconds();

    yield();
    sleep(2);
    yield();

    myLog.close();
    cout << "It took me " << dTime << " seconds" << endl;
    return 0;
}