/*
 * mpUtils
 * oglInterface.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */
#ifndef MPUTILS_OGLINTERFACE_H
#define MPUTILS_OGLINTERFACE_H

// includes
//--------------------
#include <cinttypes>
#include <cstdio>
#include <functional>
//--------------------

namespace fnd {

void initializeFrontend(); //!< initialize the frontend
uint32_t getPositionBuffer(size_t n);   //!< generate a position buffer and get its openGL id (only defined for the opgenGL frontend)
uint32_t getVelocityBuffer(size_t n);    //!< generate a velocity buffer and get its openGL id (only defined for the opgenGL frontend)
uint32_t getDensityBuffer(size_t n);    //!< generate a density buffer and get its openGL id (only defined for the opgenGL frontend)
void setPauseHandler(std::function<void(bool)> f); //!< the function f is called with true when the simulation should pause and with false when it should resume
bool handleFrontend(double t); //!< allow the frontend to do all its regular tasks. call in the main loop. pass the total simulated time in t. returns false if the app was terminated by the user
void setParticleSize(float pradius); //!< set the particle size for openGL rendering

}

#endif //MPUTILS_OGLINTERFACE_H
