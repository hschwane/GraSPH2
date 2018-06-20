/*
 * gpulic
 * Framebuffer.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the Framebuffer class
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */

#ifndef GPULIC_FRAMEBUFFER_H
#define GPULIC_FRAMEBUFFER_H

// includes
//--------------------
#include <GL/glew.h>
#include "Handle.h"
#include "Texture.h"
#include <vector>
#include <map>
//--------------------

// namespace
//--------------------
namespace mpu {
namespace gph {
//--------------------

//-------------------------------------------------------------------
/**
 * class Framebuffer
 *
 * usage: Use a Frame Buffer Object to render.
 * You can attach self managed textures, or use a Renderbuffer which will be
 * managed automatically by the Framebuffer class.
 *
 */
class Framebuffer
        : public Handle<uint32_t, decltype(&glCreateFramebuffers), &glCreateFramebuffers, decltype(&glDeleteFramebuffers), &glDeleteFramebuffers>
{
public:

    Framebuffer() = default;

    Framebuffer(nullptr_t) : Handle(nullptr) {}

    /**
     * @brief creates a renderbuffer and attaches it to the fbo
     * @param attachment  the attachment point to add the attachment, eg GL_COLOR_ATTACHMENT0 or GL_DEPTH_STENCIL_ATTACHMENT
     * @param internal_format the internal format of the buffer, eg GL_RGB or GL_DEPTH24_STENCIL8
     * @param size the dsired size of the renderbuffer
     */
    void useRenderbuffer(GLenum attachment, GLenum internal_format, glm::ivec2 size);

    /**
     * @brief Attach a texture to the fbo. You remain in full responsibility for the textures livetime
     * @param attachment he attachment point to add the attachment, eg GL_COLOR_ATTACHMENT0 or GL_DEPTH_STENCIL_ATTACHMENT
     * @param texture the texture to attach to the fbo
     * @param level the texture mipmap level to attach
     */
    void attach(GLenum attachment, const Texture& texture, int level = 0);
    void use(); //!< start using the fbo
    void disable() const; //!< stop using the fbo

private:
    using Renderbuffer = Handle<uint32_t, decltype(&glCreateRenderbuffers), &glCreateRenderbuffers, decltype(&glDeleteRenderbuffers), &glDeleteRenderbuffers>;

    std::map<GLenum, Renderbuffer> m_color_renderbuffers;
    Renderbuffer m_depth_stencil_renderbuffer{nullptr};

    std::vector<GLenum> m_draw_buffers;
};

}}

#endif //GPULIC_FRAMEBUFFER_H
