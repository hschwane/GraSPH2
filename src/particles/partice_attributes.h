/*
 * GraSPH2
 * partice_attributes.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */
#ifndef GRASPH2_PARTICE_ATTRIBUTES_H
#define GRASPH2_PARTICE_ATTRIBUTES_H

// includes
//--------------------
#include "types.h"
#include "particle_base.h"
//--------------------

//-------------------------------------------------------------------
// create bases for particles and particle buffers

MAKE_PARTICLE_ATTRIB(POS,pos,f3_t, deriv_of_nothing);
MAKE_PARTICLE_ATTRIB(MASS,mass,f1_t, deriv_of_nothing);
MAKE_PARTICLE_ATTRIB(VEL,vel,f3_t, POS);
MAKE_PARTICLE_ATTRIB(ACC,acc,f3_t, VEL);
MAKE_PARTICLE_ATTRIB(XVEL,xvel,f3_t, POS);
MAKE_PARTICLE_ATTRIB(DENSITY,density,f1_t, deriv_of_nothing);
MAKE_PARTICLE_ATTRIB(DENSITY_DT,density_dt,f1_t, DENSITY);
MAKE_PARTICLE_ATTRIB(DSTRESS,dstress,m3_t, deriv_of_nothing);
MAKE_PARTICLE_ATTRIB(DSTRESS_DT,dstress_dt,m3_t, DSTRESS);

using particle_base_order = std::tuple<POS,MASS,VEL,ACC,XVEL,DENSITY,DENSITY_DT,DSTRESS,DSTRESS_DT>;

#endif //GRASPH2_PARTICE_ATTRIBUTES_H
