# ########################################################################
# Copyright 2016 Advanced Micro Devices, Inc.
# ########################################################################

# Downloads and builds Cobalt. Defines:
# Cobalt_INCLUDE_DIRS
# CobaltLib_LIBRARIES
# CobaltLogger_LIBRARIES

message( STATUS "Configuring Cobalt external dependency" )
include( ExternalProject )

set( Cobalt_REPO "https://github.com/clMathLibraries/Cobalt.git"
    CACHE STRING "URL to download Cobalt from" )
set( Cobalt_TAG "develop" CACHE STRING "Cobalt branch to download" )

option( Cobalt_ENABLE_LOGGER "Enable logger in Cobalt?" )

# TODO rocBLAS clients to use logger and write XMLs to build dir, not src

ExternalProject_Add(
  Cobalt
  GIT_REPOSITORY ${Cobalt_REPO}
  GIT_TAG ${Cobalt_TAG}
  PREFIX ${CMAKE_BINARY_DIR}/extern/Cobalt
  CMAKE_ARGS
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCobalt_BACKEND=HIP
    -DCobalt_ENABLE_LOGGER=${Cobalt_ENABLE_LOGGER}
    -DCobalt_OPTIMIZE_ALPHA=OFF
    -DCobalt_OPTIMIZE_BETA=OFF
    -DCobalt_DIR_PROBLEMS=${CMAKE_SOURCE_DIR}/library/src/blas3/Cobalt/XML_Problems
    -DCobalt_DIR_SOLUTIONS=${CMAKE_SOURCE_DIR}/library/src/blas3/Cobalt/XML_SolutionTimes
  BUILD_COMMAND make && make
  INSTALL_COMMAND ""
  #UPDATE_COMMAND ""
)

ExternalProject_Get_Property( Cobalt SOURCE_DIR )
ExternalProject_Get_Property( Cobalt BINARY_DIR )
set( Cobalt_INCLUDE_DIRS ${SOURCE_DIR}/CobaltLib/include )
set( CobaltLib_LIBRARIES ${BINARY_DIR}/CobaltLib/libCobaltLib.a )
set( CobaltLogger_LIBRARIES ${BINARY_DIR}/CobaltLib/libCobaltLogger.a )
#MESSAGE( "Cobalt_INCLUDE_DIRS: ${Cobalt_INCLUDE_DIRS}" )
#MESSAGE( "CobaltLib_LIBRARIES: ${CobaltLib_LIBRARIES}" )
#MESSAGE( "CobaltLogger_LIBRARIES: ${CobaltLogger_LIBRARIES}" )

#set_property( TARGET Cobalt PROPERTY FOLDER "extern")
#ExternalProject_Get_Property( Cobalt source_dir )

# For use by the user of external-Cobalt.cmake
# set( Cobalt_ROOT ${source_dir} )
