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

# Check if cmake supports the new VERSION tag for project() commands
# rocblas becomes the name of the project with a particular version
if( POLICY CMP0048 )
	cmake_policy( SET CMP0048 NEW )
	project( rocblas-samples VERSION 0.0.0.1 LANGUAGES C CXX )
else( )
	project( rocblas-samples CXX )
	# Define a version for the code
	if( NOT DEFINED rocblas_VERSION_MAJOR )
		set( rocblas-samples_VERSION_MAJOR 0 )
	endif( )

	if( NOT DEFINED rocblas_VERSION_MINOR )
		set( rocblas-samples_VERSION_MINOR 0 )
	endif( )

	if( NOT DEFINED rocblas_VERSION_PATCH )
		set( rocblas-samples_VERSION_PATCH 0 )
	endif( )

	if( NOT DEFINED rocblas_VERSION_TWEAK )
		set( rocblas-samples_VERSION_TWEAK 1 )
	endif( )

	set( rocblas-samples_VERSION "${rocblas-samples_VERSION_MAJOR}.${rocblas-samples_VERSION_MINOR}.${rocblas-samples_VERSION_PATCH}.${rocblas-samples_VERSION_TWEAK}")
endif( )

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

find_package( rocblas REQUIRED CONFIG )

# Build into subdirectories
add_subdirectory( gemm-expert )
