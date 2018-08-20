/*
 * mpUtils
 * consoleFrontend.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */

#include "frontendInterface.h"
#include <mpUtils.h>
#include <iostream>

namespace fnd {

namespace cmdFrontend {

    // sttings
    const double maxTime = 10.0f;

    // variables
    double totalTime = 0;
    double timeSincePrint = 0;
}

void initializeFrontend()
{
    logINFO("Frontend")
        << "You have compiled the console frontend, which is just a dummy and does not provide any interactive features.";
}

uint32_t getPositionBuffer(size_t n)
{
    return 0;
}

uint32_t getVelocityBuffer(size_t n)
{
    return 0;
}

void setPauseHandler(std::function<void(bool)> f)
{
}

bool handleFrontend()
{
    using namespace cmdFrontend;
    static mpu::DeltaTimer dtime;
    double dt = dtime.getDeltaTime();

    totalTime += dt;
    timeSincePrint += dt;

    if(timeSincePrint > 4.0f)
    {
        timeSincePrint = 0;
        logINFO("Frontend") << "Simulation ran for " << totalTime << "seconds.";
    }

    if(totalTime < maxTime)
        return true;
    else
        return false;

}

}