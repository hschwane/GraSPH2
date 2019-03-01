/*
 * GraSPH2
 * headlessSettings.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */
#ifndef GRASPH2_HEADLESSSETTINGS_H
#define GRASPH2_HEADLESSSETTINGS_H

// Settings for headless simulations. No effect when using with the openGL-frontend.
const double maxRuntime = 1.0*60*60; // maximum wall-clock runtime of the simulation in seconds
const double maxSimtime = 0.5; // maximum simulated time in internal time units
const double printIntervall = 4.0; // time interval where progress information is printed to the console

#endif //GRASPH2_HEADLESSSETTINGS_H
