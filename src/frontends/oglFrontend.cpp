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
#include <mpUtils/mpUtils.h>
#include <mpUtils/mpGraphics.h>
//--------------------

// hide all local stuff in this namespace
namespace fnd {
namespace oglFronted {

enum class Falloff
{
    NONE,
    LINEAR,
    SQUARED,
    CUBED,
    ROOT
};

// settings
//--------------------
glm::uvec2 SIZE = {800,800};
constexpr char TITLE[] = "sph";

const glm::vec4 BG_COLOR = {0.3f,0.3f,0.3f,1};

const bool enableVsync    = false;

float particleRenderSize    = 0.004f;
float particleBrightness    = 1.0f;
Falloff falloffStyle        = Falloff::LINEAR;
bool perspectiveSize        = true;
bool roundParticles         = true;
bool additiveBlending       = false;
bool depthTest              = true;
bool colorcodeVelocity      = false;
glm::vec4 particleColor     = {1.0,1.0,1.0,1.0};
double printIntervall       = 2.0;

using vecType=glm::vec4;

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

// camera
mpu::gph::ModelViewProjection mvp;
mpu::gph::Camera &camera()
{
    static mpu::gph::Camera _internalCamera(std::make_shared<mpu::gph::SimpleWASDController>(&window(),40,2));
    return _internalCamera;
}

// timing
double delta{0};
double time{0};
int frames{0};

//--------------------

void recompileShader()
{
    std::vector<mpu::gph::glsl::Definition> definitions;
    if(perspectiveSize)
        definitions.push_back({"PARTICLES_PERSPECTIVE"});
    if(roundParticles)
        definitions.push_back({"PARTICLES_ROUND"});
    if(colorcodeVelocity)
        definitions.push_back({"COLORCODE_VELOCITY"});

    switch(falloffStyle)
    {
        case Falloff::LINEAR:
            definitions.push_back({"PARTICLE_FALLOFF",{"color=color*(1-distFromCenter)"}});
            break;
        case Falloff::SQUARED:
            definitions.push_back({"PARTICLE_FALLOFF",{"color=color*(1-distFromCenter*distFromCenter)"}});
            break;
        case Falloff::CUBED:
            definitions.push_back({"PARTICLE_FALLOFF",{"color=color*(1-distFromCenter*distFromCenter*distFromCenter)"}});
            break;
        case Falloff::ROOT:
            definitions.push_back({"PARTICLE_FALLOFF",{"color=color*(1-sqrt(distFromCenter))"}});
            break;
        case Falloff::NONE:
        default:
            definitions.push_back({"PARTICLE_FALLOFF",{""}});
            break;
    }

    shader.rebuild({{FRAG_SHADER_PATH},{VERT_SHADER_PATH}},definitions);
    shader.uniform2f("viewport_size", glm::vec2(SIZE));
    shader.uniform1f("render_size", particleRenderSize);
    shader.uniform1f("brightness", particleBrightness);
    shader.uniformMat4("model_view_projection", glm::mat4(1.0f));
    shader.uniformMat4("projection", glm::mat4(1.0f));
    shader.uniform4f("defaultColor",particleColor);
}

}

//-------------------------------------------------------------------
// define the interface functions
void initializeFrontend()
{
    using namespace oglFronted;

    window();

    mpu::gph::addShaderIncludePath(MPU_LIB_SHADER_PATH);
    mpu::gph::addShaderIncludePath(PROJECT_SHADER_PATH);

    mpu::gph::enableVsync(enableVsync);
    glClearColor(BG_COLOR.x,BG_COLOR.y,BG_COLOR.z,BG_COLOR.w);
    glEnable(GL_PROGRAM_POINT_SIZE);
    if(additiveBlending)
    {
        glBlendFunc(GL_ONE, GL_ONE);
        glEnable(GL_BLEND);
    }
    if(depthTest)
    {
        glEnable(GL_DEPTH_TEST);
    }

    vao.recreate();

    shader.recreate();
    recompileShader();


    camera().setMVP(&mvp);
    camera().setClip(0.001,10);

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
    positionBuffer.allocate<vecType>(n);
    vao.addAttributeBufferArray(POS_BUFFER_BINDING,positionBuffer,0, sizeof(vecType),4,0);

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
    velocityBuffer.allocate<vecType>(particleCount);
    vao.addAttributeBufferArray(VEL_BUFFER_BINDING,velocityBuffer,0, sizeof(vecType),4,0);

    return velocityBuffer;
}

void setPauseHandler(std::function<void(bool)> f)
{
    using namespace oglFronted;
    pauseHandler = f;
}

bool handleFrontend(double t)
{
    using namespace oglFronted;
    static mpu::DeltaTimer timer;
    delta = timer.getDeltaTime();

    time+=delta;
    frames++;
    if(time > printIntervall)
    {
        logINFO("openGL Frontend") << "Simulated Time: " << t << " fps: " << frames / time << " ms/f: " << time/frames * 1000;
        frames=0;
        time=0;
    }

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    camera().update(delta);

    vao.bind();
    shader.use();
    shader.uniformMat4("model_view_projection", mvp.getModelViewProjection());
    shader.uniformMat4("projection", mvp.getProj());
    glDrawArrays(GL_POINTS, 0, particleCount);

    if(window().getKey(GLFW_KEY_1))
        pauseHandler(false);
    if(window().getKey(GLFW_KEY_2))
        pauseHandler(true);

    return window().update();
}

}