cmake_minimum_required(VERSION 3.8)

# -----------------
# general settings
# -----------------

# Appends the cmake/ path to MAKE_MODULE_PATH variable.
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

# include some useful stuff
include(GNUInstallDirs)
include(GetVersionFromGit)

# possibility to turn of features with special dependencies
option(USE_CUDA "Compile with cuda features" OFF)
option(USE_OPENGL "Compile with openGL features" ON)

# enable languages
enable_language(CXX)
if (USE_CUDA)
    enable_language(CUDA)
endif()

# create project
project(mpUtils VERSION "${VERSION_SHORT}")

# find directories
set(LIB_SHADER_PATH ${CMAKE_SOURCE_DIR}/shader CACHE PATH "Library Shader Path. Set manually if it was not found.")
set(LIB_RESOURCE_PATH ${CMAKE_SOURCE_DIR}/resources CACHE PATH "Library Resource path. Set manually if it was not found.")
if (USE_CUDA)
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
endif()

# set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
if (USE_CUDA)
    set(CUDA_CODE_GEN "-gencode arch=compute_50,code=sm_50" CACHE STRING "Cuda arch for which code should be generated.")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_CODE_GEN} -std=c++14")
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")
endif ()

# set defines
add_definitions(-DLIB_SHADER_PATH="${LIB_SHADER_PATH}/")
add_definitions(-DLIB_RESOURCE_PATH="${LIB_RESOURCE_PATH}/")
add_definitions(-DMPU_LIB_VERSION="${VERSION_SHORT}")
if (USE_OPENGL)
    add_definitions(-DGLFW_INCLUDE_GLCOREARB)
    add_definitions(-DGLEW_NO_GLU)
    add_definitions(-DUSE_OPENGL)
endif ()
if (USE_CUDA)
    add_definitions(-DUSE_CUDA)
endif ()

# -------------------------------------------------------------
# find packages, load libraris include pathes and source files
# -------------------------------------------------------------

# general
list(APPEND LIBRARIES "stdc++fs")
list(APPEND SOURCE_FILES
        "src/stringUtils.cpp"
        "src/Log/LogStream.cpp"
        "src/Log/FileSink.cpp"
        "src/Log/ConsoleSink.cpp"
        "src/Log/SyslogSink.cpp"
        "src/Log/Log.cpp"
        "src/Cfg/CfgFile.cpp")

# mutithreading
find_package( Threads ) # the systems thread lib
list(APPEND LIBRARIES ${CMAKE_THREAD_LIBS_INIT})

# openGL and window managing
if (USE_OPENGL)
    find_package(OpenGL REQUIRED)
    find_package(GLEW REQUIRED)
    find_package(GLFW3 REQUIRED)
    find_package(GLM REQUIRED)

    list(APPEND LIBRARIES   "${GLFW3_LIBRARIES}"
                            "${GLEW_LIBRARIES}"
                            "${OPENGL_LIBRARY}")
    list(APPEND INCLUDE_PATHES
            "${OPENGL_INCLUDE_DIRS}"
            "${GLEW_INCLUDE_PATH}"
            "${GLFW3_INCLUDE_PATH}"
            "${GLM_INCLUDE_PATH}")

    list(APPEND SOURCE_FILES
            "src/Graphics/Rendering/Camera.cpp"
            "src/Graphics/Opengl/Framebuffer.cpp"
            "src/Graphics/Opengl/Shader.cpp"
            "src/Graphics/Opengl/Texture.cpp"
            "src/Graphics/Opengl/glsl/Preprocessor.cpp"
            "src/Graphics/Utils/Transform.cpp"
            "src/Graphics/Utils/ModelViewProjection.cpp"
            "src/Graphics/Window.cpp")
endif ()

# cuda
if (USE_CUDA)
#    list(APPEND SOURCE_FILES
#            "src/Cuda/...")
    if ($ENV{CLION_IDE})
        list(APPEND INCLUDE_PATHES "/usr/local/cuda/include")
    endif ()
endif ()

# create target
add_library(mpUtils SHARED ${SOURCE_FILES})

# link libraries
target_link_libraries( mpUtils  ${LIBRARIES})

# set properties
set_target_properties(mpUtils PROPERTIES VERSION ${VERSION_SHORT} SOVERSION ${VERSION_MAJOR})

# set include dir (in this case also the src)
target_include_directories(mpUtils PUBLIC src ${INCLUDE_PATHES})
        #$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/>
        #$<INSTALL_INTERFACE:src/>)


# ---------------------
# set install options
# ---------------------
install(TARGETS mpUtils
        ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})

install(
        DIRECTORY ${CMAKE_SOURCE_DIR}/src/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mpUtils
        FILES_MATCHING PATTERN "*.h*")

# --------------------------------------------------------
# see if there are executables and add the subdirectories
# --------------------------------------------------------
file(GLOB children RELATIVE ${CMAKE_SOURCE_DIR}/exec ${CMAKE_SOURCE_DIR}/exec/*)
set(subdirs "")
foreach(child ${children})
    if(IS_DIRECTORY ${CMAKE_SOURCE_DIR}/exec/${child})
        if (NOT ${child} MATCHES "\\..*")
            if(EXISTS ${CMAKE_SOURCE_DIR}/exec/${child}/CMakeLists.txt)
                string(REPLACE " " "_" child ${child})
                set(subdirs ${subdirs} ${child})
                message("Found Executable in folder '${child}'.")
            endif()
        endif()
    endif()
endforeach()
foreach(n ${subdirs})
    add_subdirectory(${CMAKE_SOURCE_DIR}/exec/${n})
endforeach()
