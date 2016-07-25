# ########################################################################
# Copyright 2016 Advanced Micro Devices, Inc.
# ########################################################################

message( STATUS "Configuring hip external dependency" )
include( ExternalProject )

if( WIN32 )
  message( AUTHOR_WARNING "It is not known if HiP works in a windows environment" )
endif( )

set( hip_git_repository "https://github.com/GPUOpen-ProfessionalCompute-Tools/HIP.git" CACHE STRING "URL to download hip from" )
set( hip_git_tag "master" CACHE STRING "URL to download hip from" )

set( HOST_TOOLCHAIN_FILE "${PROJECT_SOURCE_DIR}/cmake/${HOST_TOOLCHAIN_NAME}-toolchain.cmake" )

set( hip_cmake_args -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>/package -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=${HOST_TOOLCHAIN_FILE} )

if( ${BUILD_SHARED_LIBS} )
  message( STATUS "Compiling HIP as SHARED library")
  # list( APPEND hip_cmake_args -DHIP_USE_SHARED_LIBRARY=1 )
  list( APPEND hip_cmake_args -DCMAKE_CXX_FLAGS=-fPIC )
endif()

  ExternalProject_Add(
    HIP
    PREFIX ${CMAKE_BINARY_DIR}/extern/hip
  GIT_REPOSITORY ${hip_git_repository}
  GIT_TAG ${hip_git_tag}
    CMAKE_ARGS ${hip_cmake_args}
    LOG_BUILD 1
    LOG_INSTALL 1
  )

set_property( TARGET HIP PROPERTY FOLDER "extern")
ExternalProject_Get_Property( HIP install_dir )

# For use by the user of external-hip.cmake
set( HIP_ROOT ${install_dir}/package )
