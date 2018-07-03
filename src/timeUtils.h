/*
 * mpUtils
 * timeUtils.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */
#ifndef MPUTILS_TIMEUTILS_H
#define MPUTILS_TIMEUTILS_H

// includes
//--------------------
#include <algorithm>
#include <sstream>
#include <thread>
#include <chrono>
#include <iomanip>
//--------------------

// namespace
//--------------------
namespace mpu {
//--------------------

// typedefs
//--------------------
// make using timer, usw easier
typedef std::chrono::duration<int, std::ratio<60 * 60 * 24 * 365>> years;
typedef std::chrono::duration<int, std::ratio<60 * 60 * 24 * 7>> weeks;
typedef std::chrono::duration<int, std::ratio<60 * 60 * 24>> days;
typedef std::chrono::hours hours;
typedef std::chrono::minutes minutes;
typedef std::chrono::seconds seconds;
typedef std::chrono::milliseconds milliseconds;
typedef std::chrono::microseconds microseconds;
typedef std::chrono::nanoseconds nanoseconds;
//--------------------

// aliases for std::this_thread functions
//--------------------
inline void yield() { std::this_thread::yield(); }
template <typename T>
inline void sleep_d(T duration) { std::this_thread::sleep_for(duration); }
inline void sleep(int sec) { sleep_d(seconds(sec)); }
inline void sleep_ms(int ms) { sleep_d(milliseconds(ms)); }
inline void sleep_us(int us) { sleep_d(microseconds(us)); }
template <typename T>
inline void sleep_until(T tp) { std::this_thread::sleep_until(tp); }

}

#endif //MPUTILS_TIMEUTILS_H
