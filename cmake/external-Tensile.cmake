# ########################################################################
# Copyright 2016 Advanced Micro Devices, Inc.
# ########################################################################

# Downloads and builds Tensile. Defines:
# Tensile_INCLUDE_DIRS
# TensileLib_LIBRARIES
# TensileLogger_LIBRARIES

include(ExternalProject)
include(FindPythonInterp)

set( Tensile_REPO "https://github.com/guacamoleo/Tensile.git"
    CACHE STRING "URL to download Tensile from" )
set( Tensile_TAG "v2" CACHE STRING "Tensile branch to download" )



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
#message( STATUS "Downloaded and installed Tensile" )
