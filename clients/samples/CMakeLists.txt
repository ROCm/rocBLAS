# ########################################################################
# Copyright 2016 Advanced Micro Devices, Inc.
# ########################################################################
# We require 2.8.12 for linking libraries we specify PRIVATE, INTERFACE and
# PUBLIC keywords
cmake_minimum_required( VERSION 2.8.12 )

if( CMAKE_GENERATOR MATCHES "NMake" )
	option( NMAKE_COMPILE_VERBOSE "Print VERBOSE compile/link msgs to the console" OFF )
	if( NMAKE_COMPILE_VERBOSE )
		set( CMAKE_START_TEMP_FILE "" )
		set( CMAKE_END_TEMP_FILE "" )
		set( CMAKE_VERBOSE_MAKEFILE 1 )
	endif( )
endif( )

# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if( NOT MSVC_IDE AND NOT CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." )
endif()

include( build-version )
project_version( NAME rocblas-samples LANGUAGES CXX )

# Modify the global find property to help us find libraries like Boost in the correct paths for 64-bit
# Essentially, find_library calls will look for /lib64 instead of /lib; works for windows and linux
if( BUILD_64 )
	set_property( GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS TRUE )
	message( STATUS "FIND_LIBRARY_USE_LIB64_PATHS{TRUE} - 64-bit builds" )
else( )
	set_property( GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS FALSE )
	message( STATUS "FIND_LIBRARY_USE_LIB64_PATHS{FALSE} - 32-bit builds" )
endif( )

# Set common compile and link options
if( MSVC )
	# Following options for nMake
	message( STATUS "Detected MSVS Ver: " ${MSVC_VERSION} )

	# CMake uses huge stack frames for windows, we would like to remove.
	string( REGEX REPLACE "/STACK:[0-9]+" "" CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}" )
	string( REGEX REPLACE "/STACK:[0-9]+" "" CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}" )
	string( REGEX REPLACE "/STACK:[0-9]+" "" CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS}" )
endif( )

# This option only works for make/nmake and the ninja generators, but no reason it shouldn't be on all the time
# This tells cmake to create a compile_commands.json file that can be used with clang tooling or vim
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

find_package( rocblas REQUIRED CONFIG )

set( rocblas_samples_common
      ../common/utility.cpp
    )

#add_executable( example-openmp example_openmp.cpp ${rocblas_samples_common} )
add_executable( example-sscal example_sscal.cpp ${rocblas_samples_common} )

# Try to test for specific compiler features if cmake version is recent enough
if( CMAKE_VERSION VERSION_GREATER "3.0" )
  #target_compile_features( example-openmp PRIVATE cxx_static_assert cxx_nullptr cxx_lambdas cxx_auto_type )
  target_compile_features( example-sscal PRIVATE cxx_static_assert cxx_nullptr cxx_lambdas cxx_auto_type )
else( )
  # Otherwise, just try to compile the library with a standards flag
  if( CMAKE_COMPILER_IS_GNUCXX OR ( CMAKE_CXX_COMPILER_ID MATCHES "Clang" ) )
    # ::gets removed from std++14, but g++4.8 still uses it
    # http://clang-developers.42468.n3.nabble.com/libc-gets-removed-from-C11-td4032945.html
    # target_compile_options( rocblas-test PRIVATE -std=c++14 )
    #target_compile_options( example-openmp PRIVATE -std=c++11 -fopenmp -pthread )
    target_compile_options( example-sscal PRIVATE -std=c++11 -pthread )
  endif( )
endif( )


target_include_directories( example-sscal
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
)

#target_link_libraries( example-openmp rocblas )
target_link_libraries( example-sscal rocblas )

# Ubuntu systems need to explicitely link to pthreads lib because of --as-needed
# https://github.com/google/googletest/issues/391#issuecomment-125645879
# No easy way to filter to Ubuntu
if( UNIX )
#  target_link_libraries( example-openmp pthread )
  target_link_libraries( example-sscal pthread )
endif( )

#set_target_properties( example-openmp PROPERTIES DEBUG_POSTFIX "-d" )
# set_target_properties( example-openmp PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )

set_target_properties( example-sscal PROPERTIES DEBUG_POSTFIX "-d" )
# set_target_properties( example-sscal PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )
