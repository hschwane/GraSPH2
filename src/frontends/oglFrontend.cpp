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

//!< differnt falloff types
enum class Falloff
{
    NONE,
    LINEAR,
    SQUARED,
    CUBED,
    ROOT
};

//!< different coloring modes
enum class ColorMode
{
    CONSTANT = 0,
    VELOCITY = 1,
    SPEED = 2,
    DENSITY = 3
};
int numColorModes=4;
std::string colorModeToString[] = {"constant","velocity","speed","density"};

// settings
//--------------------
glm::uvec2 SIZE = {800,800}; //!< initial and current size of the window
constexpr char TITLE[] = "sph";

const glm::vec4 BG_COLOR = {0.3f,0.3f,0.3f,1};

const bool enableVsync    = false;

float particleRenderSize    = 0.0004f;
float particleBrightness    = 1.0f;
Falloff falloffStyle        = Falloff::LINEAR;
bool perspectiveSize        = true;
bool roundParticles         = true;
bool additiveBlending       = false;
ColorMode colorMode      = ColorMode::CONSTANT;
float upperBound = 1;   // upper bound of density / velocity transfer function
float lowerBound = 0.001;   // lower bound of density / velocity transfer function
glm::vec4 particleColor     = {1.0,1.0,1.0,1.0};
double printIntervall       = 4.0;

using vecType=glm::vec4;

constexpr char FRAG_SHADER_PATH[] = PROJECT_SHADER_PATH"particleRenderer.frag";
constexpr char VERT_SHADER_PATH[] = PROJECT_SHADER_PATH"particleRenderer.vert";

constexpr int POS_BUFFER_BINDING = 0;
constexpr int VEL_BUFFER_BINDING = 1;
constexpr int DENSITY_BUFFER_BINDING = 2;
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
mpu::gph::Buffer densityBuffer(nullptr);
mpu::gph::VertexArray vao(nullptr);
mpu::gph::ShaderProgram shader(nullptr);

// camera
mpu::gph::ModelViewProjection mvp;
mpu::gph::Camera &camera()
{
    static mpu::gph::Camera _internalCamera(std::make_shared<mpu::gph::SimpleWASDController>(&window(),10,2));
    return _internalCamera;
}

// timing
double delta{0};
double time{0};
int frames{0};

// input
bool wasCpressed=false;
bool wasZpressed=false;
bool needInfoPrintingUpper=false;
bool needInfoPrintingLower=false;
bool needInfoPrintingSize=false;
bool needInfoPrintingBrightness=false;

//--------------------

void recompileShader()
{
    std::vector<mpu::gph::glsl::Definition> definitions;
    if(perspectiveSize)
        definitions.push_back({"PARTICLES_PERSPECTIVE"});
    if(roundParticles)
        definitions.push_back({"PARTICLES_ROUND"});

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
    shader.uniform1ui("colorMode",static_cast<unsigned int>(colorMode));
    shader.uniform1f("upperBound",upperBound);
    shader.uniform1f("lowerBound",lowerBound);
}

void window_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0,0,width,height);
    camera().setAspect( float(width) / height);
    SIZE={width,height};
    shader.uniform2f("viewport_size", glm::vec2(SIZE));
}

void setBlending(bool additive)
{
    if(additive)
    {
        glBlendFunc(GL_ONE, GL_ONE);
        glEnable(GL_BLEND);
        glDisable(GL_DEPTH_TEST);
    }
    else
    {
        glDisable(GL_BLEND);
        glEnable(GL_DEPTH_TEST);
    }
}

}

//-------------------------------------------------------------------
// define the interface functions
void initializeFrontend()
{
    using namespace oglFronted;

    window().setSizeCallback(window_size_callback);

    mpu::gph::addShaderIncludePath(MPU_LIB_SHADER_PATH);
    mpu::gph::addShaderIncludePath(PROJECT_SHADER_PATH);

    mpu::gph::enableVsync(enableVsync);
    glClearColor(BG_COLOR.x,BG_COLOR.y,BG_COLOR.z,BG_COLOR.w);
    glEnable(GL_PROGRAM_POINT_SIZE);
    setBlending(additiveBlending);

    vao.recreate();

    shader.recreate();
    recompileShader();


    camera().setMVP(&mvp);
    camera().setClip(0.001,20);

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

uint32_t getDensityBuffer(size_t n)
{
    using namespace oglFronted;

    if(particleCount != 0)
    {
        assert_critical(particleCount == n, "openGL Frontend",
                        "You can not initialize position and density buffer with different particle numbers.");
    }
    else
        particleCount = n;


    densityBuffer.recreate();
    densityBuffer.allocate<float>(particleCount);
    vao.addAttributeBufferArray(DENSITY_BUFFER_BINDING,densityBuffer,0, sizeof(float),4,0);

    return densityBuffer;
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

    // handle change of coloring mode
    bool key_c = window().getKey(GLFW_KEY_C);
    if( key_c && !wasCpressed)
    {
        colorMode = static_cast<ColorMode>( (static_cast<int>(colorMode) + 1) % numColorModes);
        shader.uniform1ui("colorMode",static_cast<unsigned int>(colorMode));
        logINFO("openGL Frontend") << "Color Mode: " << colorModeToString[static_cast<unsigned int>(colorMode)];
        wasCpressed=true;
    }
    else if( !key_c && wasCpressed)
        wasCpressed = false;

    if(window().getKey(GLFW_KEY_V))
    {
        upperBound += (upperBound+0.1)*0.5f*delta;
        shader.uniform1f("upperBound",upperBound);
        needInfoPrintingUpper = true;
    }
    else if(window().getKey(GLFW_KEY_X))
    {
        upperBound -= (upperBound+0.1) * 0.5f*delta;
        upperBound = (upperBound < 0) ? 0 : upperBound;
        shader.uniform1f("upperBound",upperBound);
        needInfoPrintingUpper = true;
    }
    else if(needInfoPrintingUpper)
    {
        logINFO("openGL Frontend") << "Transfer function lower bound: " << lowerBound << " upper bound: " << upperBound;
        needInfoPrintingUpper = false;
    }

    if(window().getKey(GLFW_KEY_B))
    {
        lowerBound += (lowerBound+std::numeric_limits<float>::min())*0.5f*delta;
        shader.uniform1f("lowerBound",lowerBound);
        needInfoPrintingLower = true;

    } else if(window().getKey(GLFW_KEY_Z))
    {
        lowerBound -= (lowerBound+std::numeric_limits<float>::min()) * 0.5f*delta;
        lowerBound = (lowerBound < 0) ? 0 : lowerBound;
        shader.uniform1f("lowerBound",lowerBound);
        needInfoPrintingLower = true;
    }
    else if(needInfoPrintingLower)
    {
        logINFO("openGL Frontend") << "Transfer function lower bound: " << lowerBound << " upper bound: " << upperBound;
        needInfoPrintingLower = false;
    }

    // handle changes of particle size
    if(window().getKey(GLFW_KEY_R))
    {
        particleRenderSize += (particleRenderSize+std::numeric_limits<float>::min())*0.5f*delta;
        shader.uniform1f("render_size",particleRenderSize);
        needInfoPrintingSize=true;

    } else if(window().getKey(GLFW_KEY_F))
    {
        particleRenderSize -= (particleRenderSize+std::numeric_limits<float>::min()) * 0.5f*delta;
        particleRenderSize = (particleRenderSize < 0) ? 0 : particleRenderSize;
        shader.uniform1f("render_size",particleRenderSize);
        needInfoPrintingSize=true;
    }
    else if(needInfoPrintingSize)
    {
        logINFO("openGL Frontend") << "Rendered Particle Size: " << particleRenderSize;
        needInfoPrintingSize = false;
    }

    // handle changes of particle brightness
    if(window().getKey(GLFW_KEY_T))
    {
        particleBrightness += (particleBrightness+std::numeric_limits<float>::min())*0.5f*delta;
        shader.uniform1f("brightness",particleBrightness);
        needInfoPrintingBrightness=true;

    } else if(window().getKey(GLFW_KEY_G))
    {
        particleBrightness -= (particleBrightness+std::numeric_limits<float>::min()) * 0.5f*delta;
        particleBrightness = (particleBrightness < 0) ? 0 : particleBrightness;
        shader.uniform1f("brightness",particleBrightness);
        needInfoPrintingBrightness=true;
    }
    else if(needInfoPrintingBrightness)
    {
        logINFO("openGL Frontend") << "Rendered Particle Brightness: " << particleBrightness;
        needInfoPrintingBrightness = false;
    }

    bool key_y=window().getKey(GLFW_KEY_Y);
    if( key_y && !wasZpressed)
    {
        additiveBlending = !additiveBlending;
        setBlending(additiveBlending);
        wasZpressed=true;
    }
    else if(!key_y && wasZpressed)
        wasZpressed = false;


    return window().update();
}

void setParticleSize(float pradius)
{
    using namespace oglFronted;
    particleRenderSize = pradius;
    shader.uniform1f("render_size",particleRenderSize);
}

}