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
float brightness = 1.0f; // additional brightness for the particles, gets multiplied with color
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
float fieldOfFiew = 45.0f; // field of view in degrees
float near = 0.001f;
float far = 20;
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

// graphics helper objects
mpu::gph::Window &window()
{
    static mpu::gph::Window _internalWindow(SIZE.x, SIZE.y, TITLE);
    return _internalWindow;
}
mpu::gph::Camera &camera()
{
    static mpu::gph::Camera _internalCamera;
    return _internalCamera;
}
glm::mat4 projection;

// opengl objects
size_t particleCount{0};
std::unique_ptr<mpu::gph::Buffer<vecType>> positionBuffer;
std::unique_ptr<mpu::gph::Buffer<vecType>> velocityBuffer;
std::unique_ptr<mpu::gph::Buffer<fType>> densityBuffer;
std::unique_ptr<mpu::gph::VertexArray> vao;
std::unique_ptr<mpu::gph::ShaderProgram> shader;

// other
std::function<void(bool)> pauseHandler; //!< function to be called when the simulation needs to be paused

// timing
double time{0};
double lastSimTime{0};
int frames{0};

// UI
bool showRenderingWindow{false};
bool showCameraDebugWindow{false};

//--------------------
// some helping functions

void compileShader()
{
    shader = std::unique_ptr<mpu::gph::ShaderProgram>(new mpu::gph::ShaderProgram({{FRAG_SHADER_PATH},{VERT_SHADER_PATH},{GEOM_SHADER_PATH}}));
    shader->uniform1f("sphereRadius", particleRadius);
    shader->uniformMat4("view", glm::mat4(1.0f));
    shader->uniformMat4("model", glm::mat4(1.0f));
    shader->uniformMat4("projection", glm::mat4(1.0f));
    shader->uniform3f("defaultColor",particleColor);
    shader->uniform1f("brightness",brightness);
    shader->uniform1f("materialAlpha",particleAlpha);
    shader->uniform1f("materialShininess",materialShininess);
    shader->uniform1ui("colorMode",static_cast<unsigned int>(colorMode));
    shader->uniform1f("upperBound",upperBound);
    shader->uniform1f("lowerBound",lowerBound);
    shader->uniform3f("light.position",lightPosition);
    shader->uniform3f("light.diffuse",lightDiffuse);
    shader->uniform3f("light.specular",lightSpecular);
    shader->uniform3f("ambientLight",lightAmbient);
    shader->uniform1b("lightInViewSpace",linkLightToCamera);
    shader->uniform1b("renderFlatDisks",renderFlatDisks);
    shader->uniform1b("flatFalloff",flatFalloff);
    shader->uniform1b("enableEdgeHighlights",enableEdgeHighlights);
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

void addKeybindings()
{
    using namespace mpu::gph;
    // camera
    camera().addInputs();
    Input::mapKeyToInput("CameraMoveSideways",GLFW_KEY_D,Input::ButtonBehavior::whenDown,Input::AxisBehavior::positive);
    Input::mapKeyToInput("CameraMoveSideways",GLFW_KEY_A,Input::ButtonBehavior::whenDown,Input::AxisBehavior::negative);
    Input::mapKeyToInput("CameraMoveForwardBackward",GLFW_KEY_W,Input::ButtonBehavior::whenDown,Input::AxisBehavior::positive);
    Input::mapKeyToInput("CameraMoveForwardBackward",GLFW_KEY_S,Input::ButtonBehavior::whenDown,Input::AxisBehavior::negative);
    Input::mapKeyToInput("CameraMoveUpDown",GLFW_KEY_Q,Input::ButtonBehavior::whenDown,Input::AxisBehavior::negative);
    Input::mapKeyToInput("CameraMoveUpDown",GLFW_KEY_E,Input::ButtonBehavior::whenDown,Input::AxisBehavior::positive);

    Input::mapCourserToInput("CameraPanHorizontal", Input::AxisOrientation::horizontal,Input::AxisBehavior::negative,0, "EnablePan");
    Input::mapCourserToInput("CameraPanVertical", Input::AxisOrientation::vertical,Input::AxisBehavior::positive,0, "EnablePan");
    Input::mapScrollToInput("CameraZoom");

    Input::mapMouseButtonToInput("EnablePan", GLFW_MOUSE_BUTTON_MIDDLE);
    Input::mapKeyToInput("EnablePan", GLFW_KEY_LEFT_ALT);

    Input::mapCourserToInput("CameraRotateHorizontal", Input::AxisOrientation::horizontal,Input::AxisBehavior::negative,0, "EnableRotation");
    Input::mapCourserToInput("CameraRotateVertical", Input::AxisOrientation::vertical,Input::AxisBehavior::negative,0, "EnableRotation");

    Input::mapMouseButtonToInput("EnableRotation", GLFW_MOUSE_BUTTON_LEFT);
    Input::mapKeyToInput("EnableRotation", GLFW_KEY_LEFT_CONTROL);

    Input::mapKeyToInput("CameraMovementSpeed",GLFW_KEY_RIGHT_BRACKET,Input::ButtonBehavior::whenDown,Input::AxisBehavior::positive);
    Input::mapKeyToInput("CameraMovementSpeed",GLFW_KEY_SLASH,Input::ButtonBehavior::whenDown,Input::AxisBehavior::negative);
//    Input::mapKeyToInput("CameraToggleMode",GLFW_KEY_R);
    Input::mapKeyToInput("CameraSlowMode",GLFW_KEY_LEFT_SHIFT,Input::ButtonBehavior::whenDown);
    Input::mapKeyToInput("CameraFastMode",GLFW_KEY_SPACE,Input::ButtonBehavior::whenDown);

    Input::addButton("CameraToggleModeAndReset","Changes the camera mode and resets target when mode is switched to trackball.",[](mpu::gph::Window&)
    {
        if(camera().getMode()==mpu::gph::Camera::fps)
        {
            camera().setMode(mpu::gph::Camera::trackball);
            camera().setTarget({0, 0, 0});
        } else
            camera().setMode(mpu::gph::Camera::fps);
    });
    Input::mapKeyToInput("CameraToggleModeAndReset",GLFW_KEY_R);

    // pause / run simulation
    Input::addButton("PauseSim","Pauses the simulation.",[](mpu::gph::Window&){pauseHandler(true); mpu::gph::enableVsync(true);});
    Input::addButton("ResumeSim","Resume the simulation.",[](mpu::gph::Window&){pauseHandler(false); mpu::gph::enableVsync(enableVsync);});
    Input::mapKeyToInput("PauseSim",GLFW_KEY_2);
    Input::mapKeyToInput("ResumeSim",GLFW_KEY_1);

    // change coloring mode
    Input::addButton("ChangeColorMode","Toggles between all coloring modes.",[](mpu::gph::Window&)
    {
        colorMode = static_cast<ColorMode>( (static_cast<int>(colorMode) + 1) % numColorModes);
        shader->uniform1ui("colorMode",static_cast<unsigned int>(colorMode));
        logINFO("openGL Frontend") << "Color Mode: " << colorModeToString[static_cast<unsigned int>(colorMode)];
    });
    Input::mapKeyToInput("ChangeColorMode",GLFW_KEY_F);

    // particle size
    Input::addAxis("ParticleSize","Changes the drawn size of the particles.",[](mpu::gph::Window&,double v)
    {
        particleRadius += (particleRadius + std::numeric_limits<float>::min()) * 0.025f *  float(v);
        shader->uniform1f("sphereRadius", particleRadius);
    });
    Input::mapKeyToInput("ParticleSize",GLFW_KEY_T,Input::ButtonBehavior::whenDown,Input::AxisBehavior::positive);
    Input::mapKeyToInput("ParticleSize",GLFW_KEY_G,Input::ButtonBehavior::whenDown,Input::AxisBehavior::negative);
    Input::addButton("PrintParticleSize","Prints the current particle size to the console.",[](mpu::gph::Window&)
    {
        logINFO("openGL Frontend") << "Rendered Particle Size: " << particleRadius;
    });
    Input::mapKeyToInput("PrintParticleSize",GLFW_KEY_T,Input::ButtonBehavior::onRelease);
    Input::mapKeyToInput("PrintParticleSize",GLFW_KEY_G,Input::ButtonBehavior::onRelease);

    // UI
    Input::addButton("ToggleRendererWindow","Toggle visibility of the renderer settings window.",[](mpu::gph::Window&)
    {
        showRenderingWindow = ! showRenderingWindow;
    });
    Input::mapKeyToInput("ToggleRendererWindow",GLFW_KEY_V);

    Input::addButton("ToggleCamDebugWindow","Toggle visibility of the camera debug window.",[](mpu::gph::Window&)
    {
        showCameraDebugWindow = ! showCameraDebugWindow;
    });
    Input::mapKeyToInput("ToggleCamDebugWindow",GLFW_KEY_C);
}

void drawRendererSettingsWindow(bool* show = nullptr)
{
    using namespace mpu::gph;
    if(ImGui::Begin("Rendering",show))
    {
        ImGui::PushID("RenderingMode");
        if(ImGui::CollapsingHeader("Rendering Mode"))
        {
            if(ImGui::Checkbox("Additive Blending", &additiveBlending))
                setBlending(additiveBlending);

            if(ImGui::RadioButton("Sphere", !renderFlatDisks))
            {
                renderFlatDisks = false;
                shader->uniform1b("renderFlatDisks", renderFlatDisks);
            }
            ImGui::SameLine();
            if(ImGui::RadioButton("Disk", renderFlatDisks))
            {
                renderFlatDisks = true;
                shader->uniform1b("renderFlatDisks", renderFlatDisks);
            }

            if(renderFlatDisks)
            {
                if(ImGui::Checkbox("Falloff", &flatFalloff))
                    shader->uniform1b("flatFalloff", flatFalloff);
            }

            if(ImGui::Checkbox("Edge Highlights",&enableEdgeHighlights))
                shader->uniform1b("enableEdgeHighlights",enableEdgeHighlights);
        }

        ImGui::PopID();
        ImGui::PushID("ParticleProperties");

        if(ImGui::CollapsingHeader("Particle properties"))
        {
            if(ImGui::SliderFloat("Size", &particleRadius, 0.00001, 1, "%.5f",2.0f))
                shader->uniform1f("sphereRadius", particleRadius);

            const char* modes[] = {"constant","velocity","speed","density"};
            int selected = static_cast<int>(colorMode);
            if(ImGui::Combo("Color Mode", &selected, modes,numColorModes))
            {
                colorMode = static_cast<ColorMode>(selected);
                shader->uniform1ui("colorMode",static_cast<unsigned int>(colorMode));
            }

            if(colorMode == ColorMode::CONSTANT)
            {

                if(ImGui::ColorEdit3("Color", glm::value_ptr(particleColor)))
                    shader->uniform3f("defaultColor", particleColor);
            } else if(colorMode != ColorMode::VELOCITY)
            {
                if(ImGui::DragFloat("Upper Bound",&upperBound,0.01,0.0f,0.0f,"%.5f"))
                    shader->uniform1f("upperBound",upperBound);
                if(ImGui::DragFloat("Lower Bound",&lowerBound,0.01,0.0f,0.0f,"%.5f"))
                    shader->uniform1f("lowerBound",lowerBound);
            }

            if(ImGui::SliderFloat("Brightness", &brightness, 0, 1, "%.4f", 4.0f))
                shader->uniform1f("brightness", brightness);

            if(ImGui::SliderFloat("Shininess", &materialShininess, 0, 50))
                shader->uniform1f("materialShininess", materialShininess);
        }

        ImGui::PopID();
        ImGui::PushID("LightProperties");

        if(ImGui::CollapsingHeader("Light properties"))
        {
            if(ImGui::Checkbox("Move Light with Camera", &linkLightToCamera))
                shader->uniform1b("lightInViewSpace", linkLightToCamera);

            if(ImGui::DragFloat3("Position", glm::value_ptr(lightPosition)))
                shader->uniform3f("light.position", lightPosition);

            if(ImGui::ColorEdit3("Diffuse", glm::value_ptr(lightDiffuse)))
                shader->uniform3f("light.diffuse", lightDiffuse);

            if(ImGui::ColorEdit3("Specular", glm::value_ptr(lightSpecular)))
                shader->uniform3f("light.specular", lightSpecular);

            if(ImGui::ColorEdit3("Ambient", glm::value_ptr(lightAmbient)))
                shader->uniform3f("ambientLight", lightAmbient);
        }
        ImGui::PopID();
    }
    ImGui::End();
}

}

//-------------------------------------------------------------------
// define the interface functions
void initializeFrontend()
{
    using namespace oglFronted;

    window().addFBSizeCallback([](int width, int height)
    {
        glViewport(0,0,width,height);
        projection = glm::perspective( glm::radians(fieldOfFiew), float(width) / float(height), near, far);
        shader->uniformMat4("projection", projection);
        SIZE={width,height};
    });

    ImGui::create(window());

    mpu::gph::enableVsync(true);
    glClearColor(BG_COLOR.x,BG_COLOR.y,BG_COLOR.z,BG_COLOR.w);
    setBlending(additiveBlending);

    vao = std::make_unique<mpu::gph::VertexArray>();
    mpu::gph::addShaderIncludePath(MPU_LIB_SHADER_PATH"/include");
    mpu::gph::addShaderIncludePath(PROJECT_SHADER_PATH);
    compileShader();

    projection = glm::perspective( glm::radians(fieldOfFiew), float(SIZE.x) / float(SIZE.y), near, far);
    shader->uniformMat4("projection", projection);

    addKeybindings();

    camera().setPosition({0,0,4});
    camera().setTarget({0,0,0});
    camera().setRotationSpeedFPS(0.0038);
    camera().setRotationSpeedTB(0.012);

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

    positionBuffer = std::make_unique<mpu::gph::Buffer<vecType>>(n);
    vao->addAttributeBufferArray(POS_BUFFER_BINDING,POS_BUFFER_BINDING,*positionBuffer,0, sizeof(vecType),4,0,glType);

    return *positionBuffer;
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


    velocityBuffer = std::make_unique<mpu::gph::Buffer<vecType>>(n);
    vao->addAttributeBufferArray(VEL_BUFFER_BINDING,VEL_BUFFER_BINDING,*velocityBuffer,0, sizeof(vecType),4,0,glType);

    return *velocityBuffer;
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


    densityBuffer = std::make_unique<mpu::gph::Buffer<fType>>(n);
    vao->addAttributeBufferArray(DENSITY_BUFFER_BINDING,DENSITY_BUFFER_BINDING,*densityBuffer,0, sizeof(fType),4,0,glType);

    return *densityBuffer;
}

void setPauseHandler(std::function<void(bool)> f)
{
    using namespace oglFronted;
    pauseHandler = std::move(f);
}

bool handleFrontend(double t)
{
    using namespace oglFronted;

    if(!window().frameBegin())
        return false;
    mpu::gph::Input::update();

    time += mpu::gph::Input::deltaTime();
    frames++;
    if(time > printIntervall)
    {
        logINFO("openGL Frontend") << "SimTime: " << t << " | f/s: " << frames / time << " | ms/f: " << time/frames * 1000 << " | timestep: " << (t-lastSimTime)/frames << " | SimTime/s: " << (t-lastSimTime)/time;
        lastSimTime = t;
        frames=0;
        time=0;
    }

    // UI
    if(showRenderingWindow)
        drawRendererSettingsWindow(&showRenderingWindow);
    if(showCameraDebugWindow)
        camera().showDebugWindow(&showCameraDebugWindow);

    camera().update();

    // render
    vao->bind();
    shader->use();
    shader->uniformMat4("view", camera().viewMatrix());
    glDrawArrays(GL_POINTS, 0, particleCount);

    window().frameEnd();
    return true;
}

void setParticleSize(float pradius)
{
    using namespace oglFronted;
    particleRadius = pradius;
    shader->uniform1f("sphereRadius", particleRadius);
}

void setDropHandler(std::function<void(std::string)> f)
{
    using namespace oglFronted;
    mpu::gph::Input::addDropCallback([f](mpu::gph::Window& wnd,const std::vector<std::string>& files)
    {
       if(&window() == &wnd)
       {
           f(files[0]);
       }
    });
}

}