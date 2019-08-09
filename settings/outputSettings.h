/*
 * GraSPH2
 * outputSettings.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2019 Hendrik Schwanekamp
 *
 */
#ifndef GRASPH2_OUTPUTSETTINGS_H
#define GRASPH2_OUTPUTSETTINGS_H


// --------------------------------
// settings about how output data is stored

// storing results as file
#define STORE_RESULTS // should results be stored at all?
constexpr char RESULT_FOLDER[] = "/home/hendrik/test/"; // results will be stored in this folder
constexpr char RESULT_PREFIX[] = "graSPH2_"; // prefix for filename
constexpr f1_t store_intervall=0.03; // simulation time between files (should be bigger then the simulation timestep)
constexpr int maxJobs=10; // maximum number of snapshots to be stored in RAM, before simulation will be paused to save the files to disk
using HostDiscPT = HostParticleBuffer<HOST_POSM,HOST_VEL,HOST_DENSITY>; // particle attributes to be stored
//#define STORE_HDF5 // use hdf5 files instead of

// DO NOT MODIFY BELOW HERE
//-------------------------------------------------------------------------
// DO NOT MODIFY BELOW HERE
// check if options are valid...

#if defined(STORE_HDF5) && !(HDF5_AVAILABLE)
    #warning "App was not compiled with hdf5 libhdf5. Output will use text files instead."
    #undef STORE_HDF5
#endif


#endif //GRASPH2_OUTPUTSETTINGS_H
