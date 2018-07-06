/*
 * mpUtils
 * oglFrontend.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */

// includes
//--------------------
#include "frontendInterface.h"
#include <mpUtils.h>
#include <Graphics/Graphics.h>
//--------------------

// hide all local stuff in this namespace
namespace fnd {
namespace oglFronted {


// settings
//--------------------
glm::uvec2 SIZE = {800,800};
constexpr char TITLE[] = "Planetform";

const glm::vec4 BG_COLOR = {0,0,0,1};

constexpr float particleRenderSize = 0.01f;
constexpr float particleBrightness = 1.0f;

constexpr char FRAG_SHADER_PATH[] = PROJECT_SHADER_PATH"particleRenderer.frag";
constexpr char VERT_SHADER_PATH[] = PROJECT_SHADER_PATH"particleRenderer.vert";

constexpr int POS_BUFFER_BINDING = 0;
constexpr int VEL_BUFFER_BINDING = 1;
//--------------------

// internal global variables
//--------------------
mpu::gph::Window &window()
{
    static mpu::gph::Window _interalWindow(SIZE.x, SIZE.y, TITLE);
    return _interalWindow;
}

std::function<void(bool)> pauseHandler; //!< function to be calles when the simulation needs to be paused

// opengl buffer
size_t particleCount{0};
mpu::gph::Buffer positionBuffer(nullptr);
mpu::gph::Buffer velocityBuffer(nullptr);
mpu::gph::VertexArray vao(nullptr);
mpu::gph::ShaderProgram shader(nullptr);
//--------------------

}

//-------------------------------------------------------------------
// define the interface functions
void initializeFrontend()
{
    using namespace oglFronted;

    window();

    mpu::gph::addShaderIncludePath(LIB_SHADER_PATH);
    mpu::gph::addShaderIncludePath(PROJECT_SHADER_PATH);

    glClearColor(BG_COLOR.x,BG_COLOR.y,BG_COLOR.z,BG_COLOR.w);
    glEnable(GL_PROGRAM_POINT_SIZE);

    vao.recreate();

    shader.recreate();
    shader.rebuild({{FRAG_SHADER_PATH},{VERT_SHADER_PATH}},{{"PARTICLES_PERSPECTIVE"},{"PARTICLES_ROUND"}});

    shader.uniform2f("viewport_size", glm::vec2(SIZE));
    shader.uniform1f("render_size", particleRenderSize);
    shader.uniform1f("brightness", particleBrightness);
    shader.uniformMat4("model_view_projection", glm::mat4(1.0f));
    shader.uniformMat4("projection", glm::mat4(1.0f));

    logINFO("openGL Frontend") << "Initialization of openGL frontend successful. Have fun with real time visualization!";
}

uint32_t getPositionBuffer(size_t n)
{
    using namespace oglFronted;

    if(particleCount != 0)
    {
        assert_critical(particleCount == n, "openGL Frontend",
                    "You can not initialize position and velocity buffer with different particle numbers.");
    }
    else
        particleCount = n;


    positionBuffer.recreate();
    positionBuffer.allocate<glm::vec4>(n);
    vao.addAttributeBufferArray(POS_BUFFER_BINDING,positionBuffer,0, sizeof(glm::vec4),3,0);

    return positionBuffer;
}

uint32_t getVelocityBuffer(size_t n)
{
    using namespace oglFronted;

    if(particleCount != 0)
    {
        assert_critical(particleCount == n, "openGL Frontend",
                    "You can not initialize position and velocity buffer with different particle numbers.");
    }
    else
        particleCount = n;


    velocityBuffer.recreate();
    velocityBuffer.allocate<glm::vec4>(particleCount);
    vao.addAttributeBufferArray(VEL_BUFFER_BINDING,velocityBuffer,0, sizeof(glm::vec4),3,0);

    return velocityBuffer;
}

void setPauseHandler(std::function<void(bool)> f)
{
    using namespace oglFronted;
    pauseHandler = f;
}

bool handleFrontend(double dt)
{
    using namespace oglFronted;

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    vao.bind();
    shader.use();
    glDrawArrays(GL_POINTS, 0, particleCount);

    return window().update();
}

}