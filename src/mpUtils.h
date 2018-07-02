/*
 * mpUtils
 * mpUtils.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */
#ifndef MPUTILS_MPUTILS_H
#define MPUTILS_MPUTILS_H

// general stuff
#include "stringUtils.h"
#include "timeUtils.h"

// include configuration util
#include "Cfg/CfgFile.h"

// include the logger
#include "Log/ConsoleSink.h"
#include "Log/FileSink.h"
#include "Log/Log.h"

#ifdef __linux__
    #include "Log/SyslogSink.h"
#endif

// include timer
#include "Timer/AsyncTimer.h"
#include "Timer/DeltaTimer.h"
#include "Timer/Stopwatch.h"
#include "Timer/Timer.h"

// include graphics
#ifdef USE_OPENGL
    #include "Graphics/Graphics.h"
#endif

// include cuda
#ifdef USE_CUDA
    #include "Cuda/cudaUtils.h"
#endif


#endif //MPUTILS_MPUTILS_H
