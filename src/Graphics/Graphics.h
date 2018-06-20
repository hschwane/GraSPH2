/*
 * mpUtils
 * Graphics.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2017 Hendrik Schwanekamp
 *
 */
#ifndef MPUTILS_GRAPHICS_H
#define MPUTILS_GRAPHICS_H

// includes
//--------------------
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/ext.hpp>
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

// include everything useful from the graphics part of the framework
//____________________
#include "Window.h"
#include "Utils/Transform.h"
#include "Utils/ModelViewProjection.h"
#include "Opengl/Buffer.h"
#include "Opengl/VertexArray.h"
#include "Opengl/Shader.h"
#include "Opengl/Texture.h"
#include "Opengl/Framebuffer.h"
#include "Rendering/Camera.h"
#include "Rendering/screenFillingTri.h"
//--------------------

#endif //MPUTILS_GRAPHICS_H
