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

namespace fnd {

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

bool handleFrontend(double dt)
{
    return true;
}

}