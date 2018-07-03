cmake_minimum_required(VERSION 3.8)

# set src files
set(SOURCE_FILES
        main.cpp
        cudaTest.cu)

# create target
add_executable(devTest ${SOURCE_FILES})

# link libraries
target_link_libraries(devTest mpUtils)

