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
option(USE_CUDA "Compile with cuda features." ON)
option(USE_OPENGL "Compile with openGL features." ON)
option(USE_GLM "Include and link GLM. This is required for openGL." ON)
option(USE_OPENMP "Activate openMP support." ON)
option(ADD_ADDITIONAL_X11_LIBS "On some linux systems additional Librarys are needed for linking of opengl." OFF)

# dependencies between features
if(USE_OPENGL)
    set(USE_GLM ON)
endif()
if(USE_OPENGL AND NOT USE_GLM)
    message(SEND_ERROR "GLM is required for openGL support!")
endif()

# enable languages
enable_language(CXX)
if (USE_CUDA)
    enable_language(CUDA)
#    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
endif()

# create project
project(mpUtils VERSION "${VERSION_SHORT}")

# set some defines for library directories
set(LIB_SHADER_PATH ${CMAKE_SOURCE_DIR}/shader CACHE PATH "Library Shader Path. Set manually if it was not found.")
set(LIB_RESOURCE_PATH ${CMAKE_SOURCE_DIR}/resources CACHE PATH "Library Resource path. Set manually if it was not found.")

# set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
if (USE_CUDA)
    set(CUDA_CODE_GEN "-gencode arch=compute_50,code=sm_50" CACHE STRING "Cuda arch for which code should be generated.")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_CODE_GEN} -std=c++14")
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP")
endif ()

# set defines
add_definitions(-DLIB_SHADER_PATH="${LIB_SHADER_PATH}/")
add_definitions(-DLIB_RESOURCE_PATH="${LIB_RESOURCE_PATH}/")
add_definitions(-DMPU_LIB_VERSION="${VERSION_SHORT}")
if(USE_GLM)
    add_definitions(-DGLM_ENABLE_EXPERIMENTAL)
    add_definitions(-DUSE_GLM)
endif()
if(USE_OPENMP)
    add_definitions(-DUSE_OPENMP)
endif()
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
find_package(Threads REQUIRED) # the systems thread lib
if(Threads_FOUND)
    message("SYSTEM LIBRARY 'Thread' FOUND.")
    list(APPEND LIBRARIES ${CMAKE_THREAD_LIBS_INIT})
else()
    message(SEND_ERROR "System Thread library was not found.")
endif()

# glm
if(USE_GLM)
    find_package(GLM REQUIRED)
    list(APPEND INCLUDE_PATHES
            "${GLM_INCLUDE_PATH}")
endif()

# openGL and window managing
if (USE_OPENGL)
    find_package(OpenGL REQUIRED)
    find_package(GLEW REQUIRED)
    find_package(GLFW3 REQUIRED)

    list(APPEND LIBRARIES   "${GLFW3_LIBRARIES}"
                            "${GLEW_LIBRARIES}"
                            "${OPENGL_LIBRARY}")

    list(APPEND INCLUDE_PATHES
            "${OPENGL_INCLUDE_DIRS}"
            "${GLEW_INCLUDE_PATH}"
            "${GLFW3_INCLUDE_PATH}")

    list(APPEND SOURCE_FILES
            "src/Graphics/Rendering/Camera.cpp"
            "src/Graphics/Opengl/Framebuffer.cpp"
            "src/Graphics/Opengl/Shader.cpp"
            "src/Graphics/Opengl/Texture.cpp"
            "src/Graphics/Opengl/glsl/Preprocessor.cpp"
            "src/Graphics/Utils/Transform.cpp"
            "src/Graphics/Utils/ModelViewProjection.cpp"
            "src/Graphics/Window.cpp")

    # x11 is required on some linux systems
    find_package(X11)
    if(X11_FOUND)
        message("EXTERNAL LIBRARY 'X11' FOUND.")
        list(APPEND LIBRARIES   "${X11_LIBRARIES}")
        list(APPEND INCLUDE_PATHES ${X11_INCLUDE_DIR})
    else()
        message("X11 was not found, but it might not be needed.")
    endif()

    if(ADD_ADDITIONAL_X11_LIBS)
        message("Additional X11 libraries added.")
        list(APPEND LIBRARIES "Xrandr" "Xinerama" "Xcursor" "Xxf86vm")
    else()
        message("Additional X11 libraries were NOT added. If you get crazy linker errors try to enable ADD_ADDITIONAL_X11_LIBS.")
    endif()

endif ()

# openMP
if(USE_OPENMP)
    find_package(OpenMP REQUIRED) # openmp
    if(OPENMP_FOUND)
        message("OpenMP support enabled.")
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    else()
        message(SEND_ERROR "OpenMP support could not be enabled.")
    endif()
endif()

# cuda
if (USE_CUDA)
#    list(APPEND SOURCE_FILES
#            "src/Cuda/...")
    if ($ENV{CLION_IDE})
        list(APPEND INCLUDE_PATHES CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
    endif ()
    # enable openmp for the cuda compiler
    if(USE_OPENMP)
        set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp")
    endif()
endif ()

# create target
add_library(mpUtils SHARED ${SOURCE_FILES})

# link libraries
target_link_libraries( mpUtils  ${LIBRARIES})

# set properties
set_target_properties(mpUtils PROPERTIES VERSION ${VERSION_SHORT} SOVERSION ${VERSION_MAJOR})
if (USE_CUDA)
    set_target_properties( mpUtils PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif ()

# set include dir (in this case also the src)
target_include_directories(mpUtils PUBLIC src ${INCLUDE_PATHES})
        #$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/>
        #$<INSTALL_INTERFACE:src/>)


# ---------------------
# set install options
# ---------------------
# currently not supportet
# install(TARGETS mpUtils
#        ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
#        LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
#        RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})
#
#install(
#        DIRECTORY ${CMAKE_SOURCE_DIR}/src/
#        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mpUtils
#        FILES_MATCHING PATTERN "*.h*")

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
