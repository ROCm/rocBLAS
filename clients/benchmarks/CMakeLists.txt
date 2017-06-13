# ########################################################################
# Copyright 2016 Advanced Micro Devices, Inc.
# ########################################################################
# We require 2.8.12 for linking libraries we specify PRIVATE, INTERFACE and
# PUBLIC keywords
cmake_minimum_required( VERSION 2.8.12 )

include( build-version )
project_version( NAME rocblas-client LANGUAGES CXX )

#==================================NEW============================
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

find_package( rocblas REQUIRED CONFIG PATHS /opt/rocm/rocblas )

# set( Boost_DEBUG ON )
set( Boost_USE_MULTITHREADED ON )
set( Boost_DETAILED_FAILURE_MSG ON )
set( Boost_ADDITIONAL_VERSIONS 1.60.0 1.60 )
set( Boost_USE_STATIC_LIBS ON )
find_package( Boost REQUIRED COMPONENTS program_options )

message( STATUS "Boost_FOUND: " ${Boost_FOUND} )

if( NOT Boost_FOUND )
	set( Boost_USE_STATIC_LIBS OFF )
	find_package( Boost REQUIRED COMPONENTS program_options )
endif( )



set( rocblas_benchmark_common
      ../common/utility.cpp
      ../common/cblas_interface.cpp
      ../common/flops.cpp
      ../common/norm.cpp
      ../common/unit.cpp
      ../common/rocblas_template_specialization.cpp
    )

add_executable( client client.cpp ${rocblas_benchmark_common} )


add_subdirectory ( ./perf_script )



if( BUILD_WITH_TENSILE )
    target_compile_definitions( client PRIVATE BUILD_WITH_TENSILE=1 )
    message(STATUS, "Build Tensil equal 1")
else()
    target_compile_definitions( client PRIVATE BUILD_WITH_TENSILE=0 )
    message(STATUS, "Build Tensil equal 0")
endif()


# Try to test for specific compiler features if cmake version is recent enough
if( CMAKE_VERSION VERSION_GREATER "3.0" )
  target_compile_features( client PRIVATE cxx_static_assert cxx_nullptr cxx_lambdas cxx_auto_type )
else( )
  # Otherwise, just try to compile the library with a standards flag
  if( CMAKE_COMPILER_IS_GNUCXX OR ( CMAKE_CXX_COMPILER_ID MATCHES "Clang" ) )
    # ::gets removed from std++14, but g++4.8 still uses it
    # http://clang-developers.42468.n3.nabble.com/libc-gets-removed-from-C11-td4032945.html
    # target_compile_options( rocblas-test PRIVATE -std=c++14 )
    #target_compile_options( example-openmp PRIVATE -std=c++11 -fopenmp -pthread )
    target_compile_options( client PRIVATE -std=c++11 -pthread )
  endif( )
endif( )


target_include_directories( client
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>
)

target_link_libraries( client rocblas ${Boost_LIBRARIES} blas lapack)

# Ubuntu systems need to explicitely link to pthreads lib because of --as-needed
# https://github.com/google/googletest/issues/391#issuecomment-125645879
# No easy way to filter to Ubuntu
if( UNIX )
  target_link_libraries( client pthread )
endif( )

set_target_properties( client PROPERTIES DEBUG_POSTFIX "-d" )
# set_target_properties( client PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )

# Depending on whether we are building for 64 or 32 bit, construct common paths and names that subdirectories can reference for their use
if( rocBLAS_BUILD64 )
	set( CPACK_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}-${rocBLAS-clients_VERSION}-${CMAKE_HOST_SYSTEM_NAME}-x64" )
	set( INCLUDE_DIR include )
else( )
	set( CPACK_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}-${rocBLAS-clients_VERSION}-${CMAKE_HOST_SYSTEM_NAME}-x32" )
	set( INCLUDE_DIR include )
endif( )

# The following code is setting variables to control the behavior of CPack to generate our
if( WIN32 )
	set( CPACK_SOURCE_GENERATOR "ZIP" )
	set( CPACK_GENERATOR "ZIP" )
else( )
	set( CPACK_SOURCE_GENERATOR "TGZ" )
	set( CPACK_GENERATOR "TGZ" )
endif( )

set( CPACK_SOURCE_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}-${rocBLAS-clients_VERSION}-${CMAKE_HOST_SYSTEM_NAME}-Source")
# set( CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.txt")
set( CPACK_PACKAGE_VERSION_MAJOR ${rocBLAS-clients_VERSION_MAJOR} )
set( CPACK_PACKAGE_VERSION_MINOR ${rocBLAS-clients_VERSION_MINOR} )
set( CPACK_PACKAGE_VERSION_PATCH ${rocBLAS-clients_VERSION_PATCH} )
set( CPACK_PACKAGE_VERSION_TWEAK ${rocBLAS-clients_VERSION_TWEAK} )
set( CPACK_PACKAGE_DESCRIPTION_SUMMARY "Radeon Open Compute BLAS library package")
set( CPACK_PACKAGE_VENDOR "AMD")
set( CPACK_SOURCE_IGNORE_FILES "/\\\\.git/;/\\\\.hg/;/\\\\.svn/;" )

# Define all variables that influence CPack before including CPack, such as install targets
include( CPack )
