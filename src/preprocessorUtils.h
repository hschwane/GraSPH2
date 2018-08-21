/*
 * mpUtils
 * preprocessorUtils.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */
#ifndef MPUTILS_PREPROCESSORUTILS_H
#define MPUTILS_PREPROCESSORUTILS_H

//! stringsize
#define STRINGIZE(arg)  STRINGIZE1(arg)
#define STRINGIZE1(arg) STRINGIZE2(arg)
#define STRINGIZE2(arg) #arg

//! concatination
#define CONCATENATE(arg1, arg2)   CONCATENATE1(arg1, arg2)
#define CONCATENATE1(arg1, arg2)  CONCATENATE2(arg1, arg2)
#define CONCATENATE2(arg1, arg2)  arg1##arg2

//! a comma for use with the preprocessor
#define MPU_COMMA ,

// a comma seperated list
#define MPU_COMMA_LIST_1(x, ...) x
#define MPU_COMMA_LIST_2(x, ...) x MPU_COMMA MPU_COMMA_LIST_1(__VA_ARGS__)
#define MPU_COMMA_LIST_3(x, ...) x MPU_COMMA MPU_COMMA_LIST_2(__VA_ARGS__)
#define MPU_COMMA_LIST_4(x, ...) x MPU_COMMA MPU_COMMA_LIST_3(__VA_ARGS__)
#define MPU_COMMA_LIST_5(x, ...) x MPU_COMMA MPU_COMMA_LIST_4(__VA_ARGS__)
#define MPU_COMMA_LIST_6(x, ...) x MPU_COMMA MPU_COMMA_LIST_5(__VA_ARGS__)
#define MPU_COMMA_LIST_7(x, ...) x MPU_COMMA MPU_COMMA_LIST_6(__VA_ARGS__)
#define MPU_COMMA_LIST_8(x, ...) x MPU_COMMA MPU_COMMA_LIST_7(__VA_ARGS__)

#define MPU_COMMA_LIST_NARG(...) MPU_COMMA_LIST_NARG_(__VA_ARGS__, MPU_COMMA_LIST_RSEQ_N())
#define MPU_COMMA_LIST_NARG_(...) MPU_COMMA_LIST_ARG_N(__VA_ARGS__)
#define MPU_COMMA_LIST_ARG_N(_1, _2, _3, _4, _5, _6, _7, _8, N, ...) N
#define MPU_COMMA_LIST_RSEQ_N() 8, 7, 6, 5, 4, 3, 2, 1, 0

#define MPU_COMMA_LIST_(N, x, ...) CONCATENATE(MPU_COMMA_LIST_, N)(x, __VA_ARGS__)
#define MPU_COMMA_LIST(x, ...) MPU_COMMA_LIST_(MPU_COMMA_LIST_NARG(x, __VA_ARGS__), x, __VA_ARGS__)


#endif //MPUTILS_PREPROCESSORUTILS_H
