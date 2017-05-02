# ########################################################################
# Copyright 2016 Advanced Micro Devices, Inc.
# ########################################################################

include(ExternalProject)
include(FindPythonInterp)

set( Tensile_REPO "https://github.com/RadeonOpenCompute/Tensile.git"
    CACHE STRING "Tensile URL to download" )
set( Tensile_TAG "v2.4.5" CACHE STRING "Tensile tag to download" )

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

# For use by the user of external-Tensile.cmake
set( Tensile_ROOT ${CMAKE_BINARY_DIR}/extern/Tensile/src/Tensile)
