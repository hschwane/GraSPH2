cmake_minimum_required(VERSION 3.8)

# set src files
set(SOURCE_FILES
        main.cpp
        cudaTest.cu)

# create target
add_executable(devTest ${SOURCE_FILES})
#set_target_properties( devTest PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# link libraries
target_link_libraries(devTest mpUtils)

