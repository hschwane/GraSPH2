cmake_minimum_required(VERSION 3.8)

# set src files
set(SOURCE_FILES
        main.cpp)

# create target
add_executable(pfsph ${SOURCE_FILES})

# link libraries
target_link_libraries(pfsph mpUtils)

