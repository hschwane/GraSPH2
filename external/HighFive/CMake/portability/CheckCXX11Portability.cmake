#
# Basic check for C++11 compiler support
#

include(CheckCXXCompilerFlag)

if(NOT DEFINED CXX11_INCOMPATIBLE_COMPILER)

if(CMAKE_CXX_COMPILER_ID STREQUAL "XL")
	set(CXX11_INCOMPATIBLE_COMPILER TRUE)
else()
	set(CXX11_INCOMPATIBLE_COMPILER FALSE)
endif()

endif()

if(NOT CXX11_INCOMPATIBLE_COMPILER)
	set(CMAKE_REQUIRED_QUIET ON)
	CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
	set(CMAKE_REQUIRED_QUIET OFF)
endif()

