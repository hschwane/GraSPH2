cmake_minimum_required(VERSION 3.8)

# set src files
set(SOURCE_FILES
        main.cu Particles.cu Particles.h dummy.cpp)

# create target
add_executable(pfsph ${SOURCE_FILES})

# make thrust to use openMP on the host
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP")

# link libraries
target_link_libraries(pfsph mpUtils)

