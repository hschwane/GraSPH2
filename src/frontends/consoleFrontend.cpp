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
#include "settings/headlessSettings.h"
#include <mpUtils/mpUtils.h>
#include <iostream>

namespace fnd {

namespace cmdFrontend {

    // variables
    double totalRuntime{0};
    double timeSincePrint{0};
    int frames{0};
}

void initializeFrontend()
{
    using namespace cmdFrontend;
    logINFO("Console Frontend")
        << "You have compiled the console frontend. There are no interactive features.";
    logINFO("Console Frontend") << "The Simulation will shut down automatically after "
        << maxRuntime/(60*60) << " hours, or when "
        << maxSimtime << " time units have been simulated.";
    logINFO("Console Frontend") << "I will keep you updated on the console.";
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
    f(false); // unpause simulation
}

bool handleFrontend(double t)
{
    using namespace cmdFrontend;
    static mpu::DeltaTimer timer;
    double delta = timer.getDeltaTime();

    totalRuntime += delta;
    timeSincePrint += delta;
    frames++;
    if(timeSincePrint > printIntervall)
    {
        logINFO("Console Frontend") << "Simulation ran for " << totalRuntime/(60*60) << " hours. Simulated Time: " << t << " fps: " << frames/timeSincePrint << " ms/f: " << timeSincePrint/frames * 1000;
        frames=0;
        timeSincePrint = 0;
    }

    if(totalRuntime > maxRuntime || t > maxSimtime)
    {
        logINFO("Console Frontend") << "Simulated " << t << " time units during " << totalRuntime/(60*60) << " hours of runtime. Shutting down simulation.";
        return false;
    }
    else
        return true;


}

}