/*
 * mpUtils
 * misc.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */
#ifndef MPUTILS_MISC_H
#define MPUTILS_MISC_H

// includes
//--------------------
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "Log/Log.h"
//--------------------

// namespace
//--------------------
namespace mpu {
namespace gph {
//--------------------

//-------------------------------------------------------------------
// some global functions for the graphics framework

/**
 * Print some info about the supported openGL version to the log
 */
void inline logGlIinfo()
{
    logINFO("Graphics") << "Printing openGL version information:"
                        << "\n\t\tOpenGL version: " << glGetString(GL_VERSION)
                        << "\n\t\tGLSL version: " << glGetString(GL_SHADING_LANGUAGE_VERSION)
                        << "\n\t\tVendor: " << glGetString(GL_VENDOR)
                        << "\n\t\tRenderer: " << glGetString(GL_RENDERER)
                        << "\n\t\tGLFW. Version: " << glfwGetVersionString();
}

/**
 * pass "true" to enable or "false" to disable Vsync
 */
void inline enableVsync(bool enabled)
{
    if(enabled)
        glfwSwapInterval(1);
    else
        glfwSwapInterval(0);
}

/** Calculates the byte offset of a given member.
 * usage:
 * auto off = offset_of(&MyStruct::my_member);
 */
template<typename T, typename TMember>
GLuint offset_of(TMember T::* field) noexcept
{
    // Use 0 instead of nullptr to prohibit a reinterpret_cast of nullptr_t
    // which throws a compiler error on some compilers.
    return static_cast<GLuint>(reinterpret_cast<size_t>(&(reinterpret_cast<T*>(0)->*field)));
}

}}

#endif //MPUTILS_MISC_H
