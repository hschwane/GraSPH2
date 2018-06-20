/*
 * gpulic
 * Texture.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the Texture class
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */

// includes
//--------------------
#include "Texture.h"
#define STB_IMAGE_IMPLEMENTATION
#include <../external/stb_image.h>
#include <assert.h>
#include <Log/Log.h>

//--------------------

// namespace
//--------------------
namespace mpu {
namespace gph {
//--------------------

// global function definitions
//-------------------------------------------------------------------
std::unique_ptr<float[], void(*)(void*)> loadImageF32(const fs::path& path, int& width, int& height)
{
    int channels;
    return std::unique_ptr<float[], decltype(&stbi_image_free)>(stbi_loadf(path.string().c_str(), &width, &height, &channels, STBI_rgb_alpha), &stbi_image_free);
}

std::unique_ptr<unsigned char[], void(*)(void*)> loadImageU8(const fs::path& path, int& width, int& height)
{
    int channels;
    return std::unique_ptr<unsigned char[], decltype(&stbi_image_free)>(stbi_load(path.string().c_str(), &width, &height, &channels, STBI_rgb_alpha), &stbi_image_free);
}

inline uint32_t maxMipmaps(const uint32_t width, const uint32_t height, const uint32_t depth)
{
    return static_cast<uint32_t>(1 + glm::floor(glm::log2(glm::max(static_cast<float>(width), glm::max(static_cast<float>(height), static_cast<float>(depth))))));
}


// function definitions of the Texture class
//-------------------------------------------------------------------
Texture::Texture(nullptr_t)
        : Handle(nullptr)
{}

Texture::Texture(const GLenum target)
        : Handle(target)
{
    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
}

Texture::Texture(const TextureFileType type, const fs::path& file, const bool has_mipmaps)
        : Texture(GL_TEXTURE_2D)
{
    const auto internal_format = type == TextureFileType::eU8 ? GL_RGBA8 : GL_RGBA32F;

    int width;
    int height;

    const auto swap_image = [this, &width, &height](auto&& pixels) {
        const static auto index = [](int x, int width, int y) {
            return x + y * width * 4;
        };

        for (auto y = 0; y < static_cast<int>(height + 1) / 2; ++y)
            for (auto x = 0; x < static_cast<int>(width * 4); ++x)
                std::swap(pixels[index(x, width, y)],
                          pixels[index(x, width, height - 1 - y)]);
    };

    if (int channels; type == TextureFileType::eU8)
    {
        const auto pixels = std::unique_ptr<unsigned char[], decltype(&stbi_image_free)>(stbi_load(file.string().c_str(), &width, &height, &channels, STBI_rgb_alpha), &stbi_image_free);

        swap_image(pixels);

        allocate2D(internal_format, { width, height }, has_mipmaps ? maxMipmaps(width, height, 1) : 1);
        upload2D(0, { 0, 0 }, { width, height }, GL_RGBA, GL_UNSIGNED_BYTE, pixels.get());
    }
    else
    {
        const auto pixels = std::unique_ptr<float[], decltype(&stbi_image_free)>(stbi_loadf(file.string().c_str(), &width, &height, &channels, STBI_rgb_alpha), &stbi_image_free);

        swap_image(pixels);

        allocate2D(internal_format, { width, height }, has_mipmaps ? maxMipmaps(width, height, 1) : 1);
        upload2D(0, { 0, 0 }, { width, height }, GL_RGBA, GL_FLOAT, pixels.get());
    }

    if (has_mipmaps)
    {
        generateMipmaps();
    }
}

void Texture::allocate1D(const GLenum internal_format, const  int size, const int levels) const
{
    assert_true(width() == 0,"TEXTURE", "This texture's storage is immutable. Create a new texture to reallocate memory.");
    glTextureStorage1D(*this, levels, internal_format, size);
}

void Texture::allocate2D(const GLenum internal_format, const glm::ivec2 size, const int levels) const
{
    assert_true(width() == 0,"TEXTURE",  "This texture's storage is immutable. Create a new texture to reallocate memory.");
    glTextureStorage2D(*this, levels, internal_format, size.x, size.y);
}

void Texture::allocate3D(const GLenum internal_format, const glm::ivec3 size, const int levels) const
{
    assert_true(width() == 0,"TEXTURE",  "This texture's storage is immutable. Create a new texture to reallocate memory.");
    glTextureStorage3D(*this, levels, internal_format, size.x, size.y, size.z);
}

void Texture::upload1D(const int level, const int offset, const int size, const GLenum format, const GLenum type, const void* pixels) const
{
    assert_true(width() >= 1,"TEXTURE",  "Trying to upload to a Texture without memory. Please allocate som with allocateStorage1D.");
    glTextureSubImage1D(*this, level, offset, size, format, type, pixels);
}

void Texture::upload2D(const int level, const glm::ivec2 offset, const glm::ivec2 size, const GLenum format, const GLenum type,
                       const void* pixels) const
{
    assert_true(width() >= 1,"TEXTURE",  "Trying to upload to a Texture without memory. Please allocate som with allocateStorage2D.");
    glTextureSubImage2D(*this, level, offset.x, offset.y, size.x, size.y, format, type, pixels);
}

void Texture::upload3D(const int level, const glm::ivec3 offset, const glm::ivec3 size, const GLenum format, const GLenum type,
                       const void* pixels) const
{
    assert_true(width() >= 1,"TEXTURE",  "Trying to upload to a Texture without memory. Please allocate som with allocateStorage3D.");
    glTextureSubImage3D(*this, level, offset.x, offset.y, offset.z, size.x, size.y, size.z, format, type, pixels);
}

void Texture::upload1D(const GLenum format, const GLenum type, const void* pixels) const
{
    assert_true(width() >= 1,"TEXTURE",  "Trying to upload to a Texture without memory. Please allocate som with allocateStorage1D.");
    upload1D(0, 0, width(), format, type, pixels);
}

void Texture::upload2D(const GLenum format, const GLenum type, const void* pixels) const
{
    assert_true(width() >= 1,"TEXTURE",  "Trying to upload to a Texture without memory. Please allocate som with allocateStorage2D.");
    upload2D(0, { 0, 0 }, { width(), height() }, format, type, pixels);
}

void Texture::upload3D(const GLenum format, const GLenum type, const void* pixels) const
{
    assert_true(width() >= 1,"TEXTURE",  "Trying to upload to a Texture without memory. Please allocate som with allocateStorage3D.");
    upload3D(0, { 0, 0, 0 }, { width(), height(), depth() }, format, type, pixels);
}

void Texture::uploadCube(CubeMapFace face, const GLenum format, const GLenum type, const void* pixels) const
{
    assert_true(width() >= 1,"TEXTURE",  "Trying to upload to a Texture without memory. Please allocate som with allocateStorage2D.");
    glTextureSubImage3D(*this, 0, 0, 0, static_cast<int>(face), width(), height(), 1, format, type, pixels);
}

void Texture::generateMipmaps() const
{
    glGenerateTextureMipmap(*this);
}

void Texture::bind(GLenum textureUnit, GLenum target) const
{
    glActiveTexture(textureUnit);
    glBindTexture(target,*this);
}

int32_t Texture::width() const
{
    int value;
    glGetTextureLevelParameteriv(*this, 0, GL_TEXTURE_WIDTH, &value);
    return value;
}

int32_t Texture::height() const
{
    int value;
    glGetTextureLevelParameteriv(*this, 0, GL_TEXTURE_HEIGHT, &value);
    return value;
}

int32_t Texture::depth() const
{
    int value;
    glGetTextureLevelParameteriv(*this, 0, GL_TEXTURE_DEPTH, &value);
    return value;
}

int32_t Texture::levels() const
{
    int value;
    glGetTextureParameteriv(*this, GL_TEXTURE_MAX_LEVEL, &value);
    return value;
}

GLenum Texture::target() const
{
    GLenum value;
    glGetTextureParameterIuiv(*this, GL_TEXTURE_TARGET, &value);
    return value;
}

GLenum Texture::internalFormat() const
{
    GLenum value;
    glGetTextureParameterIuiv(*this, GL_TEXTURE_INTERNAL_FORMAT, &value);
    return value;
}

void Texture::bindImage(GLuint bindingIndex, GLenum access, GLenum format, GLint level, bool layered, GLint layer) const
{
    glBindImageTexture( bindingIndex,*this,level, layered,layer, access, format);
}

void Texture::clear(const void *clearData, GLenum format, GLenum type, GLint level)
{
    glClearTexImage(*this,level,format,type,clearData);
}

void Sampler::set(const GLenum parameter, const int value) const
{
    glSamplerParameteri(*this, parameter, value);
}

void Sampler::set(const GLenum parameter, const float* values) const
{
    glSamplerParameterfv(*this, parameter, values);
}

void Sampler::bind(GLuint unit) const
{
    glBindSampler(unit,*this);
}

}}