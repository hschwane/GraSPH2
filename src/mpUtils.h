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
#include "type_traitUtils.h"
#include "templateUtils.h"
#include "Range.h"
#if !defined(MPU_NO_PREPROCESSOR_UTILS)
    #include "preprocessorUtils.h"
#endif

// configuration util
#include "Cfg/CfgFile.h"

// the logger
#include "Log/ConsoleSink.h"
#include "Log/FileSink.h"
#include "Log/Log.h"
#ifdef __linux__
    #include "Log/SyslogSink.h"
#endif

// timer
#include "Timer/AsyncTimer.h"
#include "Timer/DeltaTimer.h"
#include "Timer/Stopwatch.h"
#include "Timer/Timer.h"

#endif //MPUTILS_MPUTILS_H
