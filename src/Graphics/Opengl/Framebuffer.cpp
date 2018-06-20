/*
 * gpulic
 * Framebuffer.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the Framebuffer class
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */

// includes
//--------------------
#include <Log/Log.h>
#include "Framebuffer.h"
//--------------------

// namespace
//--------------------
namespace mpu {
namespace gph {
//--------------------

// function definitions of the Framebuffer class
//-------------------------------------------------------------------
void Framebuffer::useRenderbuffer(const GLenum attachment, const GLenum internal_format, const glm::ivec2 size)
{
    switch (attachment)
    {
        case GL_DEPTH_STENCIL_ATTACHMENT:
            m_depth_stencil_renderbuffer.recreate();
            glNamedRenderbufferStorage(m_depth_stencil_renderbuffer, internal_format, size.x, size.y);
            glNamedFramebufferRenderbuffer(*this, attachment, GL_RENDERBUFFER, m_depth_stencil_renderbuffer);
            break;
        case GL_DEPTH_ATTACHMENT:
        case GL_STENCIL_ATTACHMENT:
            assert_true(false, "Framebuffer", "Invalid Attachments: Depth-only or Stencil-only attachments are not supported. Use a full GL_DEPTH_STENCIL_ATTACHMENT instead.");
            break;
        default:
            auto&& buffer = m_color_renderbuffers[attachment];
            buffer.recreate();
            glNamedRenderbufferStorage(buffer, internal_format, size.x, size.y);
            glNamedFramebufferRenderbuffer(*this, attachment, GL_RENDERBUFFER, buffer);
            if (std::find(m_draw_buffers.begin(), m_draw_buffers.end(), attachment) == m_draw_buffers.end())
            {
                m_draw_buffers.push_back(attachment);
                glNamedFramebufferDrawBuffers(*this, static_cast<int>(m_draw_buffers.size()), m_draw_buffers.data());
            }
            break;
    }

    assert_true(glCheckNamedFramebufferStatus(*this, GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE,"Framebuffer", "Framebuffer incomplete.");
}

void Framebuffer::attach(const GLenum attachment, const Texture& texture, const int level)
{
    glNamedFramebufferTexture(*this, attachment, texture, level);
    switch(attachment)
    {
        case GL_DEPTH_STENCIL_ATTACHMENT:
            m_depth_stencil_renderbuffer = Renderbuffer(nullptr);
            break;
        case GL_DEPTH_ATTACHMENT:
        case GL_STENCIL_ATTACHMENT:
            assert_true(false,"Framebuffer", "Invalid Attachments: Depth-only or Stencil-only attachments are not supported. Use a full GL_DEPTH_STENCIL_ATTACHMENT instead.");
            break;
        default:
            m_color_renderbuffers.erase(attachment);
            if (std::find(m_draw_buffers.begin(), m_draw_buffers.end(), attachment) == m_draw_buffers.end())
            {
                m_draw_buffers.push_back(attachment);
                glNamedFramebufferDrawBuffers(*this, static_cast<int>(m_draw_buffers.size()), m_draw_buffers.data());
            }
            break;
    }

    assert_true(glCheckNamedFramebufferStatus(*this, GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE, "Framebuffer", "Framebuffer incomplete.");
}

void Framebuffer::use()
{
    glBindFramebuffer(GL_FRAMEBUFFER, *this);
}

void Framebuffer::disable() const
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}


}}