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
find_package(mpUtils 0.9.2 REQUIRED)


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
add_executable(GraSPH2)

# add general source files
target_sources(GraSPH2 PRIVATE
                "src/main.cu"
                "src/ResultStorageManager.cu"
                "src/sph/eos.cu"
                "src/sph/models.cu"
                "src/initialConditions/particleSources/UniformSphere.cu"
                "src/initialConditions/particleSources/TextFile.cu"
                "src/dummy.cpp"
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

# force c++17 if using opengl
if(OPENGL_FRONTEND)
    set_target_properties(GraSPH2 PROPERTIES CXX_STANDARD 17)
endif()

set_target_properties( GraSPH2 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)