# ########################################################################
# Copyright 2016 Advanced Micro Devices, Inc.
# ########################################################################
# Convenience function to hide complexity of POLICY for project() and
# locates version logic into a single file
include( CMakeParseArguments )

# Check if cmake supports the new VERSION tag for project() commands
# rocblas becomes the name of the project with a particular version
macro( project_version )
  set( options )
  set( oneValueArgs NAME )
  set( multiValueArgs LANGUAGES )

  cmake_parse_arguments( PV "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

  if( NOT DEFINED PV_NAME )
    message( FATAL_ERROR "project_version(): PV_NAME argument missing")
  endif( )
  if( POLICY CMP0048 )
    cmake_policy( SET CMP0048 NEW )
    project( ${PV_NAME} VERSION 0.4.2.0 LANGUAGES ${PV_LANGUAGES} )

  else( )
    project( ${PV_NAME} ${PV_LANGUAGES} )
    # Define a version for the code
    if( NOT DEFINED ${PV_NAME}_VERSION_MAJOR )
      set( ${PV_NAME}_VERSION_MAJOR 0 )
    endif( )

    if( NOT DEFINED ${PV_NAME}_VERSION_MINOR )
      set( ${PV_NAME}_VERSION_MINOR 4 )
    endif( )

    if( NOT DEFINED ${PV_NAME}_VERSION_PATCH )
      set( ${PV_NAME}_VERSION_PATCH 2 )
    endif( )

    if( NOT DEFINED ${PV_NAME}_VERSION_TWEAK )
      set( ${PV_NAME}_VERSION_TWEAK 0 )
    endif( )

    set( ${PV_NAME}_VERSION "${${PV_NAME}_VERSION_MAJOR}.${${PV_NAME}_VERSION_MINOR}.${${PV_NAME}_VERSION_PATCH}.${${PV_NAME}_VERSION_TWEAK}" )
  endif( )
endmacro( )
