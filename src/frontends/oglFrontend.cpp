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
#include "../settings/precisionSettings.h"
#include <mpUtils/mpUtils.h>
#include <mpUtils/mpGraphics.h>
//--------------------

// hide all local stuff in this namespace
namespace fnd {
namespace oglFronted {

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

const bool enableVsync    = false;
const glm::vec4 BG_COLOR = {0.3f,0.3f,0.3f,1};

float particleRadius = 0.0004f;
bool additiveBlending = false;
ColorMode colorMode = ColorMode::CONSTANT;
float upperBound = 1;   // upper bound of density / velocity transfer function
float lowerBound = 0.001;   // lower bound of density / velocity transfer function
glm::vec3 particleColor     = {1.0,1.0,1.0}; // color used when color mode is set to constant
float particleAlpha = 1.0f;
float materialShininess = 4.0f;
double printIntervall = 4.0;
bool linkLightToCamera = true;
glm::vec3 lightPosition = {500,500,1000};
glm::vec3 lightDiffuse = {0.4,0.4,0.4};
glm::vec3 lightSpecular = {0.3,0.3,0.3};
glm::vec3 lightAmbient = {0.1,0.1,0.1};
bool renderFlatDisks = false;
bool flatFalloff = false;
bool enableEdgeHighlights = false;

//--------------------

// internal global variables
//--------------------

#if defined(DOUBLE_PRECISION)
using vecType=glm::dvec4;
    using fType=double;
    GLenum glType=GL_DOUBLE;
#elif defined(SINGLE_PRECISION)
using vecType=glm::vec4;
using fType=float;
GLenum glType=GL_FLOAT;
#endif

constexpr char FRAG_SHADER_PATH[] = PROJECT_SHADER_PATH"particleRenderer.frag";
constexpr char VERT_SHADER_PATH[] = PROJECT_SHADER_PATH"particleRenderer.vert";
constexpr char GEOM_SHADER_PATH[] = PROJECT_SHADER_PATH"particleRenderer.geom";

constexpr int POS_BUFFER_BINDING = 0;
constexpr int VEL_BUFFER_BINDING = 1;
constexpr int DENSITY_BUFFER_BINDING = 2;

mpu::gph::Window &window()
{
    static mpu::gph::Window _interalWindow(SIZE.x, SIZE.y, TITLE);
    return _interalWindow;
}

std::function<void(bool)> pauseHandler; //!< function to be called when the simulation needs to be paused
std::function<void(const std::string&)> dropHandler; //!< function is called when a file is dropped onto the window

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
double lastSimTime{0};
int frames{0};

// input
bool wasCpressed=false;
bool wasZpressed=false;
bool wasHpressed=false;
bool wasUpressed=false;
bool wasJpressed=false;
bool wasTpressed=false;
bool needInfoPrintingUpper=false;
bool needInfoPrintingLower=false;
bool needInfoPrintingSize=false;

//--------------------

void recompileShader()
{
    std::vector<mpu::gph::glsl::Definition> definitions;

    shader.rebuild({{FRAG_SHADER_PATH},{VERT_SHADER_PATH},{GEOM_SHADER_PATH}},definitions);
    shader.uniform1f("sphereRadius", particleRadius);
    shader.uniformMat4("view", glm::mat4(1.0f));
    shader.uniformMat4("model", glm::mat4(1.0f));
    shader.uniformMat4("projection", glm::mat4(1.0f));
    shader.uniform3f("defaultColor",particleColor);
    shader.uniform1f("materialAlpha",particleAlpha);
    shader.uniform1f("materialShininess",materialShininess);
    shader.uniform1ui("colorMode",static_cast<unsigned int>(colorMode));
    shader.uniform1f("upperBound",upperBound);
    shader.uniform1f("lowerBound",lowerBound);
    shader.uniform3f("light.position",lightPosition);
    shader.uniform3f("light.diffuse",lightDiffuse);
    shader.uniform3f("light.specular",lightSpecular);
    shader.uniform3f("ambientLight",lightAmbient);
    shader.uniform1b("lightInViewSpace",linkLightToCamera);
    shader.uniform1b("renderFlatDisks",renderFlatDisks);
    shader.uniform1b("flatFalloff",flatFalloff);
    shader.uniform1b("enableEdgeHighlights",enableEdgeHighlights);
}

void window_drop_callback(GLFWwindow * w, int count, const char ** c)
{
    pauseHandler(true);
    dropHandler(std::string(c[0]));
}

void window_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0,0,width,height);
    camera().setAspect( float(width) / height);
    SIZE={width,height};
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
    camera().setFOV(45);

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
    vao.addAttributeBufferArray(POS_BUFFER_BINDING,positionBuffer,0, sizeof(vecType),4,0,glType);

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
    vao.addAttributeBufferArray(VEL_BUFFER_BINDING,velocityBuffer,0, sizeof(vecType),4,0,glType);

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
    densityBuffer.allocate<fType>(particleCount);
    vao.addAttributeBufferArray(DENSITY_BUFFER_BINDING,densityBuffer,0, sizeof(fType),4,0,glType);

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
        logINFO("openGL Frontend") << "SimTime: " << t << " | f/s: " << frames / time << " | ms/f: " << time/frames * 1000 << " | timestep: " << (t-lastSimTime)/frames << " | SimTime/s: " << (t-lastSimTime)/time;
        lastSimTime = t;
        frames=0;
        time=0;
    }

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    camera().update(delta);

    vao.bind();
    shader.use();
    shader.uniformMat4("view", mvp.getView());
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
        particleRadius += (particleRadius + std::numeric_limits<float>::min()) * 0.5f * delta;
        shader.uniform1f("sphereRadius", particleRadius);
        needInfoPrintingSize=true;

    } else if(window().getKey(GLFW_KEY_F))
    {
        particleRadius -= (particleRadius + std::numeric_limits<float>::min()) * 0.5f * delta;
        particleRadius = (particleRadius < 0) ? 0 : particleRadius;
        shader.uniform1f("sphereRadius", particleRadius);
        needInfoPrintingSize=true;
    }
    else if(needInfoPrintingSize)
    {
        logINFO("openGL Frontend") << "Rendered Particle Size: " << particleRadius;
        needInfoPrintingSize = false;
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

    bool key_h=window().getKey(GLFW_KEY_H);
    if( key_h && !wasHpressed)
    {
        linkLightToCamera = !linkLightToCamera;
        shader.uniform1b("lightInViewSpace",linkLightToCamera);
        wasHpressed=true;
    }
    else if(!key_h && wasHpressed)
        wasHpressed = false;

    bool key_u=window().getKey(GLFW_KEY_U);
    if( key_u && !wasUpressed)
    {
        renderFlatDisks = !renderFlatDisks;
        shader.uniform1b("renderFlatDisks",renderFlatDisks);
        wasUpressed=true;
    }
    else if(!key_u && wasUpressed)
        wasUpressed = false;


    bool key_j=window().getKey(GLFW_KEY_J);
    if( key_j && !wasJpressed)
    {
        flatFalloff = !flatFalloff;
        shader.uniform1b("flatFalloff",flatFalloff);
        wasJpressed=true;
    }
    else if(!key_j && wasJpressed)
        wasJpressed = false;

    bool key_t=window().getKey(GLFW_KEY_T);
    if( key_t && !wasTpressed)
    {
        flatFalloff = !flatFalloff;
        shader.uniform1b("enableEdgeHighlights",enableEdgeHighlights);
        wasTpressed=true;
    }
    else if(!key_t && wasTpressed)
        wasTpressed = false;


    return window().update();
}

void setParticleSize(float pradius)
{
    using namespace oglFronted;
    particleRadius = pradius;
    shader.uniform1f("sphereRadius", particleRadius);
}

void setDropHandler(std::function<void(std::string)> f)
{
    using namespace oglFronted;
    dropHandler = f;
    window().setDropCallbac(window_drop_callback);
}

}