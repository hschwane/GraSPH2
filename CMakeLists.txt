# -------------------------------------------------------------
# general setup
# -------------------------------------------------------------
cmake_minimum_required(VERSION 3.8...3.12)

if(${CMAKE_VERSION} VERSION_LESS 3.12)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()

# Require out-of-source builds
file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
if(EXISTS "${LOC_PATH}")
    message(FATAL_ERROR "You cannot build in a source directory (or any directory with a CMakeLists.txt file). Please make a build subdirectory. Feel free to remove CMakeCache.txt and CMakeFiles.")
endif()


# -------------------------------------------------------------
# find dependencies
# -------------------------------------------------------------
enable_language(CXX)
enable_language(CUDA)
enable_language(C)
find_package(mpUtils 0.10.2 REQUIRED)
find_package(HDF5)

# -------------------------------------------------------------
# set up project
# -------------------------------------------------------------

# Appends the cmake/ path to MAKE_MODULE_PATH variable.
set(CMAKE_MODULE_PATH ${mpUtils_CMAKE_SCRIPTS_PATH} ${CMAKE_MODULE_PATH})

# create project
include(GetVersionFromGit)
project(GraSPH2 VERSION "${VERSION_SHORT}" LANGUAGES CXX CUDA)

# default build configuration
include(setDefaultTypeRelease)


# -------------------------------------------------------------
# user settings and optional features
# -------------------------------------------------------------
include(CMakeDependentOption)

# switch between different frontends
cmake_dependent_option(OPENGL_FRONTEND "Use the opengl frontend" ON "mpUtils_OPENGL_AVAILIBLE" OFF)
option(USE_FAST_MATH "Use fast math settings for nvcc. Faster simulation, but lower accuracy." OFF)
option(USE_DEBUG_LOGGING "Use debug logging even in release builds." OFF)
set(CUSTOM_SETTING_PATH "" CACHE STRING "Specify a path to a folder that contains a 'setttigns' subfolder. Configuration filles are then used from that folder instead.")

# check if that worked as we want it
if( OPENGL_FRONTEND AND (NOT mpUtils_OPENGL_AVAILIBLE))
    message(SEND_ERROR "You want to use opengl frontend but openGL is not supported by your build of mpUtils")
endif()

if(NOT mpUtils_CUDA_AVAILIBLE)
    message(SEND_ERROR "The mpUtils instalation was compiled without cuda support, however cuda support is needed.")
endif()

# -------------------------------------------------------------
# create target
# -------------------------------------------------------------

# create target
add_executable(GraSPH2 "src/dummy.cpp")

# add general source files
target_sources(GraSPH2 PRIVATE
                "src/main.cu"
                "src/ResultStorageManager.cu"
                "src/initialConditions/particleSources/UniformSphere.cu"
                "src/initialConditions/particleSources/PlummerSphere.cu"
                ${settingFiles}
            )

# add frontend dependend source files
if (OPENGL_FRONTEND)
    target_sources(GraSPH2 PRIVATE "src/frontends/oglFrontend.cpp")
    target_compile_definitions(GraSPH2 PRIVATE FRONTEND_OPENGL)
else()
    target_sources(GraSPH2 PRIVATE "src/frontends/consoleFrontend.cpp")
    target_compile_definitions(GraSPH2 PRIVATE FRONTEND_CONSOLE)
endif ()

# add definitions
set(PFSPH_PROJECT_SHADER_PATH "${CMAKE_CURRENT_LIST_DIR}/shader")
target_compile_definitions(GraSPH2 PRIVATE PROJECT_SHADER_PATH="${PFSPH_PROJECT_SHADER_PATH}/")

if (USE_DEBUG_LOGGING)
    target_compile_definitions(GraSPH2 PRIVATE MPU_ENABLE_DEBUG_LOGGING)
endif()

# link libraries
target_link_libraries(GraSPH2 mpUtils::mpUtils)

# set cuda code generation flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${mpUtils_CUDA_ARCH_FLAGS}")


# -------------------------------------------------------------
# set includes
# -------------------------------------------------------------

target_include_directories(GraSPH2
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        )

if(CUSTOM_SETTING_PATH)
    target_include_directories(GraSPH2
            PRIVATE
                ${CUSTOM_SETTING_PATH}
            )
else()
    target_include_directories(GraSPH2
            PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/settings
        )
endif()


# -------------------------------------------------------------
# add dependencies from "external" folder
# -------------------------------------------------------------
if(HDF5_FOUND)
    message(STATUS "libhdf5 found, hdf5 support enabled")
    add_subdirectory(external/HighFive)
    target_link_libraries(GraSPH2 HighFive)
else()
    message(WARNING "libhdf5 was not found. hdf5 support will be disabled")
endif()


# -------------------------------------------------------------
# set properties
# -------------------------------------------------------------

# set required language standard
set_target_properties(GraSPH2 PROPERTIES
        CXX_STANDARD 14
        CXX_STANDARD_REQUIRED YES
        CUDA_STANDARD 14
        CUDA_STANDARD_REQUIRED YES
        )

target_compile_definitions(GraSPH2 PRIVATE "GRASPH_VERSION=\"${VERSION_SHORT}\"")
target_compile_definitions(GraSPH2 PRIVATE "GRASPH_VERSION_SHA=\"${VERSION_SHA1}\"")

target_compile_options(GraSPH2 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--default-stream per-thread>)
target_compile_options(GraSPH2 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
#target_compile_options(GraSPH2 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-v --keep>)
if (USE_FAST_MATH)
    target_compile_options(GraSPH2 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
    target_compile_definitions(GraSPH2 PRIVATE "USING_CUDA_FAST_MATH")
endif ()

# force c++17 if using opengl
if(OPENGL_FRONTEND)
    set_target_properties(GraSPH2 PROPERTIES CXX_STANDARD 17)
endif()

if(HDF5_FOUND)
    target_compile_definitions(GraSPH2 PRIVATE "HDF5_AVAILABLE")
endif()

# set -Wa,-I for resources search path
if(CUSTOM_SETTING_PATH)
    target_compile_options(GraSPH2 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wa,-I${CMAKE_SOURCE_DIR},-I${CUSTOM_SETTING_PATH}>)
else()
    target_compile_options(GraSPH2 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wa,-I${CMAKE_SOURCE_DIR},-I${CMAKE_SOURCE_DIR}/settings>)
endif()


set_target_properties( GraSPH2 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)