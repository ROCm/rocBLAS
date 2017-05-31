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
project_version( NAME rocblas-test LANGUAGES CXX )

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

elseif( CMAKE_COMPILER_IS_GNUCXX OR ( CMAKE_CXX_COMPILER_ID MATCHES "Clang" ) )
  message( STATUS "Detected GNU or Clang compiler collection" )

  if( BUILD_64 )
    set( CMAKE_CXX_FLAGS "-m64 ${CMAKE_CXX_FLAGS}" )
    set( CMAKE_C_FLAGS "-m64 ${CMAKE_C_FLAGS}" )
  else( )
    set( CMAKE_CXX_FLAGS "-m32 ${CMAKE_CXX_FLAGS}" )
    set( CMAKE_C_FLAGS "-m32 ${CMAKE_C_FLAGS}" )
  endif( )
else( )
  message( AUTHOR_WARNING "Compiler not recognized.  Using default flags." )
endif( )

# This option only works for make/nmake and the ninja generators, but no reason it shouldn't be on all the time
# This tells cmake to create a compile_commands.json file that can be used with clang tooling or vim
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

# set( Boost_DEBUG ON )
set( Boost_USE_MULTITHREADED ON )
set( Boost_DETAILED_FAILURE_MSG ON )
set( Boost_ADDITIONAL_VERSIONS 1.61.0 1.61 )
set( Boost_USE_STATIC_LIBS ON )
find_package( Boost REQUIRED COMPONENTS program_options )

if( NOT Boost_FOUND )
  set( Boost_USE_STATIC_LIBS OFF )
  find_package( Boost REQUIRED COMPONENTS program_options )
endif( )

# message( STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}" )
# message( STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}" )

find_package( GTest REQUIRED )
find_package( cblas REQUIRED CONFIG  )
find_package( rocblas REQUIRED CONFIG )
find_package( HIP REQUIRED )

if( BUILD_WITH_TENSILE )
set(Tensile_TEST_SRC
    gemm_gtest.cpp
    gemm_strided_batched_gtest.cpp
    trsm_gtest.cpp
    )
endif()

set(rocblas_test_source
    rocblas_gtest_main.cpp
    set_get_vector_gtest.cpp
    set_get_matrix_gtest.cpp
    blas1_gtest.cpp
    gemv_gtest.cpp
    ger_gtest.cpp
    ${Tensile_TEST_SRC}
    )


set( rocblas_benchmark_common
      ../common/utility.cpp
      ../common/cblas_interface.cpp
      ../common/flops.cpp
      ../common/norm.cpp
      ../common/unit.cpp
      ../common/rocblas_template_specialization.cpp
    )

# link_directories( /opt/rocm/lib )
add_executable( rocblas-test ${rocblas_test_source} ${rocblas_benchmark_common} )

if( BUILD_WITH_TENSILE )
    target_compile_definitions( rocblas-test PRIVATE BUILD_WITH_TENSILE=1 )
    message(STATUS, "Build Tensil equal 1")
else()
    target_compile_definitions( rocblas-test PRIVATE BUILD_WITH_TENSILE=0 )
    message(STATUS, "Build Tensil equal 0")
endif()


# Try to test for specific compiler features if cmake version is recent enough
if( CMAKE_VERSION VERSION_GREATER "3.0" )
    target_compile_features( rocblas-test PRIVATE cxx_static_assert cxx_nullptr cxx_lambdas cxx_auto_type)
else( )
  # Otherwise, just try to compile the library with a standards flag
  if( CMAKE_COMPILER_IS_GNUCXX OR ( CMAKE_CXX_COMPILER_ID MATCHES "Clang" ) )
    # ::gets removed from std++14, but g++4.8 still uses it
    # http://clang-developers.42468.n3.nabble.com/libc-gets-removed-from-C11-td4032945.html
    # target_compile_options( rocblas-test PRIVATE -std=c++14 )
    target_compile_options( rocblas-test PRIVATE -std=c++11 -pthread )
  endif( )
endif( )

# target_compile_definitions( rocblas-test PRIVATE GTEST_USE_OWN_TR1_TUPLE=1 -DGOOGLE_TEST -D__HIP_PLATFORM_HCC__)
target_compile_definitions( rocblas-test PRIVATE GTEST_USE_OWN_TR1_TUPLE=1 -DGOOGLE_TEST )
target_include_directories( rocblas-test
  PRIVATE
#    $<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${GTEST_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${CBLAS_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<BUILD_INTERFACE:${HIP_ROOT_DIR}/include>
)

# target_link_libraries( rocblas-test rocblas ${Boost_LIBRARIES} ${CBLAS_LIBRARIES} ${GTEST_LIBRARIES} libmcwamp.a libhip_hcc.a CXLActivityLogger hsa-runtime64 hc_am hsakmt dl lapack c++ gfortran)
target_link_libraries( rocblas-test rocblas ${CBLAS_LIBRARIES} ${GTEST_LIBRARIES} lapack gfortran)

# Ubuntu systems need to explicitely link to pthreads lib because of --as-needed
# https://github.com/google/googletest/issues/391#issuecomment-125645879
# No easy way to filter to Ubuntu
if( UNIX )
  target_link_libraries( rocblas-test pthread )
endif( )

set_target_properties( rocblas-test PROPERTIES DEBUG_POSTFIX "-d" )
set_target_properties( rocblas-test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )
