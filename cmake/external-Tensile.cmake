# ########################################################################
# Copyright 2016 Advanced Micro Devices, Inc.
# ########################################################################

# Downloads and builds Tensile. Defines:
# Tensile_INCLUDE_DIRS
# TensileLib_LIBRARIES
# TensileLogger_LIBRARIES

include( ExternalProject )

set( Tensile_REPO "https://github.com/guacamoleo/Tensile.git"
    CACHE STRING "URL to download Tensile from" )
set( Tensile_TAG "v2" CACHE STRING "Tensile branch to download" )


#include( ProcessorCount )
#ProcessorCount( Cores )
#if( NOT Cores EQUAL 0 )
#  # Travis can fail to build Boost sporadically; uses 32 cores, reduce stress on VM
#  if( DEFINED ENV{TRAVIS} )
#    if( Cores GREATER 8 )
#      set( Cores 8 )
#    endif( )
#  endif( )

  # Add build thread in addition to the number of cores that we have
  #  math( EXPR Cores "${Cores} + 1 " )
  #else( )
  # If we could not detect # of cores, assume 1 core and add an additional build thread
  #  set( Cores "2" )
  #endif( )


#set( tensile_cmake_args -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>/package -DTensile_BUILD_CLIENTS=OFF )

#if( DEFINED CMAKE_CXX_COMPILER )
#  list( APPEND tensile_cmake_args -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} )
#endif( )

ExternalProject_Add(
  Tensile
  GIT_REPOSITORY ${Tensile_REPO}
  GIT_TAG ${Tensile_TAG}
  PREFIX ${CMAKE_BINARY_DIR}/extern/Tensile
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
)

#set_property( TARGET Tensile PROPERTY FOLDER "extern")
#ExternalProject_Get_Property( Tensile install_dir )

# For use by the user of external-Tensile.cmake
set( Tensile_ROOT ${CMAKE_BINARY_DIR}/extern/Tensile/src/Tensile)
message( STATUS "Downloading Tensile to Tensile_ROOT=${Tensile_ROOT}" )
