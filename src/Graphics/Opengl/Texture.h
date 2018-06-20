/*
 * gpulic
 * Texture.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the Texture class
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */

#ifndef GPULIC_TEXTURE_H
#define GPULIC_TEXTURE_H

// includes
//--------------------
#include <GL/glew.h>
#include "Handle.h"
#include <experimental/filesystem>
#include <cinttypes>
#include <glm/glm.hpp>
//--------------------

// namespace
//--------------------
namespace mpu {
namespace gph {
namespace fs = std::experimental::filesystem;
//--------------------


//-------------------------------------------------------------------
/**
 * enum class CubeMapFace
 *
 * Identify faces of a cube map.
 *
 */
enum class CubeMapFace
{
    ePosX = 0,
    eNegX,
    ePosY,
    eNegY,
    ePosZ,
    eNegZ
};

//-------------------------------------------------------------------
/**
 * enum class TextureFileType
 * Supported image formats to be loaded from files.
 */
enum class TextureFileType
{
    eU8,
    eF32
};

//-------------------------------------------------------------------
// some global functions
std::unique_ptr<float[], void(*)(void*)> loadImageF32(const fs::path& path, int& width, int& height); //!< use stb image to load a float image file
std::unique_ptr<unsigned char[], void(*)(void*)> loadImageU8(const fs::path& path, int& width, int& height); //!< use stb image to load a 8bit image file
uint32_t maxMipmaps(const uint32_t width, const uint32_t height, const uint32_t depth); //!< maximum number of mipmaps for given image dimensions

//-------------------------------------------------------------------
/**
 * class Texture
 *
 * usage: Creates a texture with immutable storage parameters.
 * Use together with sampler to specifiy sampling settings.
 *
 */
class Texture : public Handle<uint32_t, decltype(&glCreateTextures), &glCreateTextures, decltype(&glDeleteTextures), &glDeleteTextures, GLenum>
{
public:
    Texture() = default; //!< create empty texture
    Texture(nullptr_t);//!< create no texture
    explicit Texture(GLenum target); //!< create a texture for a specific target, eg GL_TEXTURE_2D

    /**
     * @brief created a texture with content from the speified file
     * @param type eU8 or eF32
     * @param file the file to read image data from
     * @param has_mipmaps should mipmaps be generated
     */
    explicit Texture(TextureFileType type, const fs::path& file, bool has_mipmaps = true);

    /**
     * @brief allocates a 1D texture to a specific formate
     * @param internal_format the sized internal format (see openGL doc)
     * @param size the size of the texture
     * @param levels the number of texture levels to be created
     */
    void allocate1D(GLenum internal_format, int size, int levels = 1) const;

    /**
     * @brief allocates a 2D texture to a specific formate
     * @param internal_format the sized internal format (see openGL doc)
     * @param size the size of the texture
     * @param levels the number of texture levels to be created
     */
    void allocate2D(GLenum internal_format, glm::ivec2 size, int levels = 1) const;

    /**
     * @brief allocates a 3D texture to a specific formate
     * @param internal_format the sized internal format (see openGL doc)
     * @param size the size of the texture
     * @param levels the number of texture levels to be created
     */
    void allocate3D(GLenum internal_format, glm::ivec3 size, int levels = 1) const;

    /**
     * @brief upload texture data for a 1D texture
     * @param level the level to which the data is loaded
     * @param offset the offset of the new data inside the texture
     * @param size the size of the new teture data
     * @param format the pixel format: GL_RED, GL_RG, GL_RGB, GL_BGR, GL_RGBA, GL_DEPTH_COMPONENT, or GL_STENCIL_INDEX
     * @param type the type of the pixel data, eg  GL_UNSIGNED_BYTE, GL_INT or GL_FLOAT
     * @param pixels the actual pixel data
     */
    void upload1D(int level, int offset, int size, GLenum format, GLenum type, const void* pixels) const;

    /**
     * @brief upload texture data for a 2D texture
     * @param level the level to which the data is loaded
     * @param offset the offset of the new data inside the texture
     * @param size the size of the new teture data
     * @param format the pixel format: GL_RED, GL_RG, GL_RGB, GL_BGR, GL_RGBA, GL_DEPTH_COMPONENT, or GL_STENCIL_INDEX
     * @param type the type of the pixel data, eg  GL_UNSIGNED_BYTE, GL_INT or GL_FLOAT
     * @param pixels the actual pixel data
     */
    void upload2D(int level, glm::ivec2 offset, glm::ivec2 size, GLenum format, GLenum type, const void* pixels) const;

    /**
     * @brief upload texture data for a 3D texture
     * @param level the level to which the data is loaded
     * @param offset the offset of the new data inside the texture
     * @param size the size of the new teture data
     * @param format the pixel format: GL_RED, GL_RG, GL_RGB, GL_BGR, GL_RGBA, GL_DEPTH_COMPONENT, or GL_STENCIL_INDEX
     * @param type the type of the pixel data, eg  GL_UNSIGNED_BYTE, GL_INT or GL_FLOAT
     * @param pixels the actual pixel data
     */
    void upload3D(int level, glm::ivec3 offset, glm::ivec3 size, GLenum format, GLenum type, const void* pixels) const;

    /**
     * @brief upload image data to a 1D texture
     * @param format the pixel format: GL_RED, GL_RG, GL_RGB, GL_BGR, GL_RGBA, GL_DEPTH_COMPONENT, or GL_STENCIL_INDEX
     * @param type the type of the pixel data, eg  GL_UNSIGNED_BYTE, GL_INT or GL_FLOAT
     * @param pixels the actual image data
     */
    void upload1D(GLenum format, GLenum type, const void* pixels) const;

    /**
     * @brief upload image data to a 2D texture
     * @param format the pixel format: GL_RED, GL_RG, GL_RGB, GL_BGR, GL_RGBA, GL_DEPTH_COMPONENT, or GL_STENCIL_INDEX
     * @param type the type of the pixel data, eg  GL_UNSIGNED_BYTE, GL_INT or GL_FLOAT
     * @param pixels the actual image data
     */
    void upload2D(GLenum format, GLenum type, const void* pixels) const;

    /**
     * @brief upload image data to a 3D texture
     * @param format the pixel format: GL_RED, GL_RG, GL_RGB, GL_BGR, GL_RGBA, GL_DEPTH_COMPONENT, or GL_STENCIL_INDEX
     * @param type the type of the pixel data, eg  GL_UNSIGNED_BYTE, GL_INT or GL_FLOAT
     * @param pixels the actual image data
     */
    void upload3D(GLenum format, GLenum type, const void* pixels) const;

    /**
     * @brief upload data to a cube map face
     * @param face the face to upload the data to
     * @param format the pixel format: GL_RED, GL_RG, GL_RGB, GL_BGR, GL_RGBA, GL_DEPTH_COMPONENT, or GL_STENCIL_INDEX
     * @param type the type of the pixel data, eg  GL_UNSIGNED_BYTE, GL_INT or GL_FLOAT
     * @param pixels the actual image data
     */
    void uploadCube(CubeMapFace face, GLenum format, GLenum type, const void* pixels) const;

    void generateMipmaps() const; //!< generate all needed mipmaps

    /**
     * @brief clear the texture data
     * @param clearData data for one pixel to be used as a clear value
     * @param format the pixel format: GL_RED, GL_RG, GL_RGB, GL_BGR, GL_RGBA, GL_DEPTH_COMPONENT, or GL_STENCIL_INDEX
     * @param type the type of the pixel data, eg  GL_UNSIGNED_BYTE, GL_INT or GL_FLOAT
     * @param level the mipmap level to clear
     */
    void clear(const void* clearData, GLenum format, GLenum type = GL_FLOAT, GLint level=0);

    void bind(GLenum textureUnit, GLenum target) const; //!< binds the texture to a specific target and texture unit
    void bindImage(GLuint bindingIndex, GLenum access, GLenum format, GLint level=0, bool layered=false, GLint layer=0) const; //!< binds an image from the texture to an image binding

    int32_t width() const; //!< get the width of the texture
    int32_t height() const; //!< get the height of the texture
    int32_t depth() const; //!< get the depth of the texture (if it is 3D)
    int32_t levels() const; //!< get the amount of mipmap levels
    GLenum target() const; //!<
    GLenum internalFormat() const; //!< get the internal format

};

/**
 * class Sampler
 *
 * usage:
 * With a sampler, you can set several texture-independant parameters, which can be
 * GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_TEXTURE_WRAP_R,
 * GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_TEXTURE_BORDER_COLOR,
 * GL_TEXTURE_MIN_LOD, GL_TEXTURE_MAX_LOD, GL_TEXTURE_LOD_BIAS, GL_TEXTURE_CUBE_MAP_SEAMLESS,
 * GL_TEXTURE_COMPARE_MODE, or GL_TEXTURE_COMPARE_FUNC
 * Therefore, you can use one sampler for several different textures for uniform sampling.
 * use bind, to bind the sampler to a texture unit
 */
class Sampler : public Handle<uint32_t, decltype(&glCreateSamplers), &glCreateSamplers, decltype(&glDeleteSamplers), &glDeleteSamplers>
{
public:
    Sampler() = default;
    Sampler(nullptr_t) : Handle(nullptr) {};

    void set(GLenum parameter, int value) const;
    void set(GLenum parameter, const float* values) const;
    void bind(GLuint unit) const; //!< bind samler to a texture unit
};



}}

#endif //GPULIC_TEXTURE_H
