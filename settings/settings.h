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

// This file contains most important settings for the GraSPH2 simulation code.
// see also other settings files in this directory
// Entries of the style "constexpr type name = value" can be just be set to the value you need.
// Entries of the style "#define NAME" are on/off switches. Comment out a line to turn the feature off.
// You will find some additional settings in the headlessSettings.h and precisionSettings.h files.

// -------------------
// general

// 2D or 3D simulation (for 2D simulation, make sure all particles have 0 values in z components of position and velocity))
//constexpr Dim dimension=Dim::two;
constexpr Dim dimension=Dim::three;

// enable / disable self gravity (gravitational constant is 1)
#define ENABLE_SELF_GRAVITY

// enable / disable SPH simulation
#define ENABLE_SPH

// -------------------
// time integration

// kick drift kick leapfrog using fixed timestep
// density and deviatoric stress are updated during drift step
//#define FIXED_TIMESTEP_LEAPFROG
constexpr f1_t fixed_timestep=0.0002; // timestep for fixed timestep leapfrog integrator

// kick drift kick leapfrog using variable timestep
// density and deviatoric stress are updated during drift step
#define VARIABLE_TIMESTEP_LEAPFROG
constexpr f1_t initial_timestep=0.0002;
constexpr f1_t min_timestep=0.000002;
constexpr f1_t max_timestep=0.02;

//--------------------
// initial conditions

// read data from a file
// one line in the file is one particle, column are seperated using the SEPERATOR character and
// represent the different particle attributes
// The option "particleToRead" specifies the particle attributes to be read. For example Particle<POS,MASS,VEL,DENSITY> would assume the following file structure:
// POS_x | POS_y | POS_z | MASS | VEL_x | VEL_y | VEL_z | DENSITY
// hdf5 files are also supported and will be detected according to file ending
//#define READ_FROM_FILE
using particleToRead = Particle<POS,MASS,VEL,DENSITY>;
constexpr char FILENAME[] = "/home/hendrik/inputData.tsv";
constexpr char SEPERATOR='\t';
constexpr double startTime = 0; // if you continue a old simulation you can set the start time to match displayed simulation times

// generate a rotating sphere of radius 1 with uniform density
// only use this with 3D simulations
#define ROTATING_UNIFORM_SPHERE
constexpr f1_t spawn_radius = 1.0_ft; // the radius particles are spawned in

// generate a rotating sphere with density distribution according to plummers law
// only use this with 3D simulations
//#define ROTATING_PLUMMER_SPHERE
constexpr f1_t plummer_cutoff = 1.0_ft; // all particles outside the cutoff will be repicked until they fall inside the radius
constexpr f1_t plummer_radius = 1.0_ft; // plummer core radius

// parameter for generated initial conditions
constexpr f1_t tmass = 1.0_ft; // total mass of the sphere
constexpr size_t particle_count=1<<14; // number of particles
constexpr f3_t angVel=f3_t{0,0,1}; // angular velocity of the cloud omega

// set a value for the smoothing length H
// you can also define a radius for a single particle
constexpr f1_t compressesd_radius = 0.1_ft;// all mass of your simulation compressed into a sphere, radius of that sphere
constexpr f1_t pradius = compressesd_radius * gcem::pow(particle_count,-1.0_ft/3.0_ft); // "radius" of a particle
constexpr f1_t H = pradius*2.5_ft; // the smoothing length H of a particle

// --------------------
// Material settings

// parameters of the equation of state
constexpr f1_t rho0 = tmass /particle_count / (4.0_ft/3.0_ft * pradius * pradius * pradius * M_PI); // the materials rest density
constexpr f1_t BULK = 92; // the materials bulk modulus
constexpr f1_t dBULKdP = 16; // the materials change of the bulk modulus with pressure
constexpr f1_t SOUNDSPEED = gcem::sqrt(BULK / rho0); // speed of sound in material

// parameters for solid bodys
constexpr f1_t shear = 128; // the materials shear modulus

// choose one of the plasticity models, if you disable both the material is purely elastic

// von miese plasticity using a yield stress Y
//#define PLASTICITY_MIESE
constexpr f1_t Y =0.5;

// mohr-coulomb plasticity, using friction angle and cohesion
#define PLASTICITY_MC
constexpr f1_t friction_angle = mpu::rad(45.0_ft); // the materials internal friction angle in radians
constexpr f1_t tanfr = gcem::tan(friction_angle); // tangents of the friction angle
constexpr f1_t cohesion = 0.0_ft; // the materials cohesion


// --------------------
// Boundary / Environment settings

// Use the Clohessy-Wiltshire model to put the entire simulation into a circular orbit around a central body
// and simulate the resulting tidal forces. Keep in mind this is a simple model / approximation.
// Also momentum is no longer conserved.
// The central body is along negative x axis, y axis points along the movement direction of the cloud along the orbit.
// The strength of the tidal forces is controlled by the parameter n = sqrt( M / (a*a*a)) (for G=1).
// where M is the mass of the central body and a the semi-major axis of the orbit
// You can also define it in terms of the hill radius as n = sqrt( m / (3*r_hill^3)) with m beeing the mass contained within r_hill.
//#define CLOHESSY_WILTSHIRE
constexpr f1_t cw_n = gcem::sqrt( 1.0_ft / 3.0_ft);

//--------------------
// artificial correction

// use artificial viscosity
#define ARTIFICIAL_VISCOSITY
constexpr f1_t alpha = 1.0_ft; // strength of artificial viscosity

// enable / disable the balsara switch
#define BALSARA_SWITCH

// artificial stress to prevent particle clumps
// not needed for most simulations
//#define ARTIFICIAL_STRESS
constexpr f1_t mateps = 0.4_ft;
constexpr f1_t matexp = 4.0_ft;
constexpr f1_t normalsep = H*0.3_ft;

// enable XSPH, a technique to smooth the velocity field
// currently broken (can be used but not recommended)
//#define XSPH
constexpr f1_t xsph_factor = 0.5_ft;


//--------------------
// advanced options
// (only change this if you know what you are doing)

// cuda tuning params
constexpr size_t BLOCK_SIZE = 256;
constexpr size_t INTEG_BS = BLOCK_SIZE; // integration blocksize
constexpr size_t INTEG_PPT = 1; // particles handled per integration thread

// types for particle buffer. you can remove things you don't need to save memory
using DeviceParticlesType = DeviceParticleBuffer<DEV_POSM,DEV_VEL,DEV_ACC,DEV_BALSARA,DEV_XVEL,DEV_DENSITY,DEV_DENSITY_DT,DEV_DSTRESS,DEV_DSTRESS_DT>;
using HostParticlesType = HostParticleBuffer<HOST_POSM,HOST_VEL,HOST_ACC,HOST_BALSARA,HOST_XVEL,HOST_DENSITY,HOST_DENSITY_DT,HOST_DSTRESS,HOST_DSTRESS_DT>;


// DO NOT MODIFY BELOW HERE
//-------------------------------------------------------------------------
// DO NOT MODIFY BELOW HERE
// check if options are valid...

#if defined(FIXED_TIMESTEP_LEAPFROG) && defined(VARIABLE_TIMESTEP_LEAPFROG)
    #error "Choose only one integration method!"
#elif !defined(FIXED_TIMESTEP_LEAPFROG) && !defined(VARIABLE_TIMESTEP_LEAPFROG)
    #error "Choose an integration method!"
#endif

#if (defined(READ_FROM_FILE) && defined(ROTATING_UNIFORM_SPHERE)) || (defined(READ_FROM_FILE) && defined(ROTATING_PLUMMER_SPHERE)) || (defined(ROTATING_UNIFORM_SPHERE) && defined(ROTATING_PLUMMER_SPHERE))
    #error "Choose only one input method!"
#endif

#if defined(PLASTICITY_MIESE) && defined(PLASTICITY_MC)
    #error "Choose up to one plasticity model, not both!"
#endif

#endif //GRASPH2_SETTINGS_H
