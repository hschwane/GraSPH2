/*
 * GraSPH2
 * settings.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */
#ifndef GRASPH2_SETTINGS_H
#define GRASPH2_SETTINGS_H

// includes
// -------------------
#include "types.h"
#include "particles/Particles.h"
// -------------------

// This file contains all important settings for the GraSPH2 simulation code.
// Entries of the style "constexpr type name = value" can be just be set to the value you need.
// Entries of the style "#define NAME" are on/off switches. Comment out a line to turn the feature off.
// You will find some additional settings in the headlessSettings.h and precisionSettings.h files.

// -------------------
// general

// 2D or 3D simulation (for 2D simulation, make sure all particles have 0 values in z components of position and velocity))
//constexpr Dim dimension=Dim::two;
constexpr Dim dimension=Dim::three;

// the integration timestep for the constant timestep leapfrog integrator
constexpr f1_t timestep=0.0003;

// storing results as file
//#define STORE_RESULTS
constexpr char RESULT_FOLDER[] = "/home/hendrik/test/"; // results will be stored in this folder
constexpr char RESULT_PREFIX[] = "graSPH2_"; // prefix for filename
constexpr f1_t store_intervall=0.03; // simulation time between files (should be bigger then the simulation timestep)
constexpr int maxJobs=10; // maximum number of snapshots to be stored in RAM, before simulation will be paused to save the files to disk

// enable / disable self gravity
#define ENABLE_SELF_GRAVITY

// enable / disable SPH simulation
#define ENABLE_SPH

//--------------------
// initial conditions

// set a value for the smoothing length H
// you can also define a radius for a single particle
constexpr f1_t pradius = 0.1 / 25.4; // "radius" of a particle
constexpr f1_t H = pradius*2.5; // the smoothing length H of a particle

// read data from a file
// one line in the file is one particle, column are seperated using the SEPERATOR character and
// represent the different particle attributes
// The Order of the parameter is assumed to be as follows:
// POS_x | POS_y | POS_z | VEL_x | VEL_y | VEL_z | MASS | DENSITY
//#define READ_FROM_FILE
#define FILENAME "/home/hendrik/inputData.tsv"
constexpr char SEPERATOR='\t';
constexpr double startTime = 0; // if you continue a old simulation you can set the start time to match displayed simulation times

// generate a rotating sphere with uniform density
// only use this with 3D simulations
#define ROTATING_UNIFORM_SPHERE
constexpr f1_t tmass = 0.5; // total mass of the sphere
constexpr f1_t particle_count=1<<14; // number of particles
constexpr f3_t angVel=f3_t{0,0,0.57735}; // angular velocity of the cloud omega


// --------------------
// Material settings

// parameters of the equation of state
constexpr f1_t rho0 = tmass /particle_count / (4.0/3.0 * pradius * pradius * pradius * M_PI); // the materials rest density
constexpr f1_t BULK = 64; // the materials bulk modulus
constexpr f1_t dBULKdP = 16; // the materials change of the bulk modulus with pressure

// parameters for solid bodys
constexpr f1_t shear = 92; // the materials shear modulus

// choose one of the plasticity models, if you disable both the material is purely elastic

// von miese plasticity using a yield stress Y
//#define PLASTICITY_MIESE
constexpr f1_t Y =0.5;

// mohr-coulomb plasticity, using friction angle and cohesion
#define PLASTICITY_MC
constexpr f1_t friction_angle = 55.0f * (M_PI/180.0f); // the materials internal friction angle in radians
constexpr f1_t cohesion = 0.8; // the materials cohesion


//--------------------
// artificial correction

// use artificial viscosity
#define ARTIFICIAL_VISCOSITY
constexpr f1_t alpha = 1; // strength of artificial viscosity

// artificial stress to prevent particle clumps
// not needed for most simulations
//#define ARTIFICIAL_STRESS
constexpr f1_t mateps = 0.4;
constexpr f1_t matexp = 4;
constexpr f1_t normalsep = H*0.3;

// enable XSPH, a technique to smooth the velocity field
//#define XSPH
constexpr f1_t xsph_factor = 0.5;


//--------------------
// advanced options
// (only change this if you know what you are doing)

// cuda block size
constexpr size_t BLOCK_SIZE = 256;

// types for particle buffer. you can remove things you don't need to save memory
using DeviceParticlesType = DeviceParticleBuffer<DEV_POSM,DEV_VEL,DEV_ACC,DEV_XVEL,DEV_DENSITY,DEV_DENSITY_DT,DEV_DSTRESS,DEV_DSTRESS_DT>;
using HostParticlesType = HostParticleBuffer<HOST_POSM,HOST_VEL,HOST_ACC,HOST_XVEL,HOST_DENSITY,HOST_DENSITY_DT,HOST_DSTRESS,HOST_DSTRESS_DT>;


// DO NOT MODIFY BELOW HERE
//-------------------------------------------------------------------------
// DO NOT MODIFY BELOW HERE
// check if options are valid...

#if defined(READ_FROM_FILE) && defined(ROTATING_UNIFORM_SPHERE)
    #error "Choose one input method, not both!"
#endif

#if defined(PLASTICITY_MIESE) && defined(PLASTICITY_MC)
    #error "Choose one plasticity model, not both!"
#endif

#endif //GRASPH2_SETTINGS_H
