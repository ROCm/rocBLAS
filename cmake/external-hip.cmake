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

# Master branch has a new structure that combines googletest with googlemock
ExternalProject_Add(
  hip
  GIT_REPOSITORY ${hip_git_repository}
  GIT_TAG ${hip_git_tag}
  PREFIX ${CMAKE_BINARY_DIR}/extern/hip
  CONFIGURE_COMMAND ""
  BUILD_COMMAND HIPCC_FLAGS=-fPIC make
  BUILD_IN_SOURCE 1
  INSTALL_COMMAND ""
)

ExternalProject_Get_Property( hip source_dir )

set_property( TARGET hip PROPERTY FOLDER "extern")
ExternalProject_Get_Property( hip source_dir )

# For use by the user of external-hip.cmake
set( HIP_ROOT ${source_dir} )
