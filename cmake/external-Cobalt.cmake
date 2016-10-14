# ########################################################################
# Copyright 2016 Advanced Micro Devices, Inc.
# ########################################################################

# Downloads and builds Cobalt. Defines:
# Cobalt_INCLUDE_DIRS
# CobaltLib_LIBRARIES
# CobaltLogger_LIBRARIES

message( STATUS "Configuring Cobalt external dependency" )
include( ExternalProject )

set( Cobalt_REPO "https://github.com/kknox/Cobalt.git"
    CACHE STRING "URL to download Cobalt from" )
set( Cobalt_TAG "develop" CACHE STRING "Cobalt branch to download" )

option( Cobalt_ENABLE_LOGGER "Enable logger in Cobalt?" OFF )

include( ProcessorCount )
ProcessorCount( Cores )
if( NOT Cores EQUAL 0 )
  # Travis can fail to build Boost sporadically; uses 32 cores, reduce stress on VM
  if( DEFINED ENV{TRAVIS} )
    if( Cores GREATER 8 )
      set( Cores 8 )
    endif( )
  endif( )

  # Add build thread in addition to the number of cores that we have
  math( EXPR Cores "${Cores} + 1 " )
else( )
  # If we could not detect # of cores, assume 1 core and add an additional build thread
  set( Cores "2" )
endif( )

message( "Building Cobalt with ${Cores} cores" )

# TODO rocBLAS clients to use logger and write XMLs to build dir, not src
ExternalProject_Add(
  Cobalt
  GIT_REPOSITORY ${Cobalt_REPO}
  GIT_TAG ${Cobalt_TAG}
  PREFIX ${CMAKE_BINARY_DIR}/extern/Cobalt
  CMAKE_ARGS
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>/package
    -DCobalt_BACKEND=HIP
    -DCobalt_ENABLE_LOGGER=${Cobalt_ENABLE_LOGGER}
    -DCobalt_OPTIMIZE_ALPHA=OFF
    -DCobalt_OPTIMIZE_BETA=OFF
    -DCobalt_DIR_PROBLEMS=${CMAKE_SOURCE_DIR}/library/src/blas3/Cobalt/XML_Problems
    -DCobalt_DIR_SOLUTIONS=${CMAKE_SOURCE_DIR}/library/src/blas3/Cobalt/XML_SolutionTimes
  BUILD_COMMAND COMMAND make -j ${Cores} CobaltLib_HCC COMMAND make -j ${Cores} CobaltLib_HCC
  LOG_BUILD 1
)

ExternalProject_Get_Property( Cobalt SOURCE_DIR )
ExternalProject_Get_Property( Cobalt BINARY_DIR )
set( Cobalt_INCLUDE_DIRS ${SOURCE_DIR}/CobaltLib/include )
set( CobaltLib_LIBRARIES ${BINARY_DIR}/CobaltLib/libCobaltLib.a )
set( CobaltLogger_LIBRARIES ${BINARY_DIR}/CobaltLib/libCobaltLogger.a )
#MESSAGE( "Cobalt_INCLUDE_DIRS: ${Cobalt_INCLUDE_DIRS}" )
#MESSAGE( "CobaltLib_LIBRARIES: ${CobaltLib_LIBRARIES}" )
#MESSAGE( "CobaltLogger_LIBRARIES: ${CobaltLogger_LIBRARIES}" )

set_property( TARGET Cobalt PROPERTY FOLDER "extern")
ExternalProject_Get_Property( Cobalt install_dir )

# For use by the user of external-Cobalt.cmake
set( Cobalt_ROOT ${install_dir}/package )
