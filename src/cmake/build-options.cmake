# ########################################################################
# Copyright 2016 Advanced Micro Devices, Inc.
# ########################################################################
# This file is intended to be used in two ways; independently in a stand alone PROJECT
# and as part of a superbuild.  If the file is included in a stand alone project, the
# variables are not expected to be preset, and this will produce options() in the GUI
# for the user to examine.  If this file is included in a superbuild, the options will be
# presented in the superbuild GUI, but then passed into the ExternalProject as -D
# parameters, which would already define them.

# Determine whether to build 64-bit (default) or 32-bit
if( NOT BUILD_LIBRARY_64 )
	if( MSVC_IDE )
		set( BUILD_LIBRARY_64 ${CMAKE_CL_64} )
		set_property( GLOBAL PROPERTY USE_FOLDERS TRUE )
	else()
		option( BUILD_LIBRARY_64 "Build a 64-bit product" ON )
	endif()
endif()

# Enable UNICODE support in the library
if( NOT BUILD_LIBRARY_UNICODE )
	option( BUILD_LIBRARY_UNICODE "Create a solution that compiles rocblas with Unicode Support" ON )

	if( BUILD_LIBRARY_UNICODE )
		message( STATUS "UNICODE build" )
	endif( )
endif()

# If the user wants unicode, WIN32 needs extra definitions
if( BUILD_LIBRARY_UNICODE AND WIN32 )
	add_definitions( "/DUNICODE /D_UNICODE" )
endif( )

if( NOT BUILD_LIBRARY_TYPE )
	set( BUILD_LIBRARY_TYPE "SHARED" CACHE STRING "Build the rocblas library as SHARED or STATIC build types" )
	set_property( CACHE BUILD_LIBRARY_TYPE PROPERTY STRINGS SHARED STATIC )
endif( )

# option( BUILD_LIBRARY_DEPENDENCY_COBALT "Build cobalt kernel library" ON )
