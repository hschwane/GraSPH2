cmake_minimum_required(VERSION 3.8)

# set src files
set(SOURCE_FILES
        main.cu Particles.cu dummy.cpp)

# have different frontends
option(OPENGL_FRONTEND "Use the opengl frontend" ON)

if(NOT USE_OPENGL)
    set(OPENGL_FRONTEND OFF)
endif()

if (OPENGL_FRONTEND)
    set(SOURCE_FILES ${SOURCE_FILES} frontends/oglFrontend.cpp)
    add_definitions(-DFRONTEND_OPENGL)
else()
    set(SOURCE_FILES ${SOURCE_FILES} frontends/consoleFrontend.cpp)
    add_definitions(-DFRONTEND_CONSOLE)
endif ()

# find directories
set(PFSPH_PROJECT_SHADER_PATH "${CMAKE_CURRENT_LIST_DIR}/shader" CACHE PATH "Project specific path. Set manually if it was not found.")
set(PFSPH_PROJECT_RESOURCE_PATH "${CMAKE_CURRENT_LIST_DIR}/resources" CACHE PATH "Project specific path. Set manually if it was not found.")

# set defines
add_definitions(-DPROJECT_SHADER_PATH="${PFSPH_PROJECT_SHADER_PATH}/")
add_definitions(-DROJECT_RESOURCE_PATH="${PFSPH_PROJECT_RESOURCE_PATH}/")

# create target
add_executable(pfsph ${SOURCE_FILES})

# make thrust to use openMP on the host
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP")

# link libraries
target_link_libraries(pfsph mpUtils)

