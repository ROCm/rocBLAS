# ########################################################################
# Copyright 2016 Advanced Micro Devices, Inc.
# ########################################################################

# We require 2.8.12 for linking libraries we specify PRIVATE, INTERFACE and
# PUBLIC keywords
cmake_minimum_required( VERSION 2.8.12 )

# CMP0063: Honor visibility properties for all target types [cmake 3.3]
if( POLICY CMP0063 )
  cmake_policy( SET CMP0063 NEW )
endif( )

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
if( NOT CMAKE_CONFIGURATION_TYPES )
  set( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." )
endif()

if( DEFINED HOST_TOOLCHAIN_FILE )
endif( )

if( DEFINED DEVICE_TOOLCHAIN_FILE )
	set( CMAKE_TOOLCHAIN_FILE "${DEVICE_TOOLCHAIN_FILE}" )
endif( )

# Check if cmake supports the new VERSION tag for project() commands
# rocblas becomes the name of the project with a particular version
if( POLICY CMP0048 )
	cmake_policy( SET CMP0048 NEW )
	project( rocblas VERSION 0.0.2.0 LANGUAGES CXX C )
else( )
	project( rocblas CXX C )
	# Define a version for the code
	if( NOT DEFINED rocblas_VERSION_MAJOR )
		set( rocblas_VERSION_MAJOR 0 )
	endif( )

	if( NOT DEFINED rocblas_VERSION_MINOR )
		set( rocblas_VERSION_MINOR 0 )
	endif( )

	if( NOT DEFINED rocblas_VERSION_PATCH )
		set( rocblas_VERSION_PATCH 2 )
	endif( )

	if( NOT DEFINED rocblas_VERSION_TWEAK )
		set( rocblas_VERSION_TWEAK 0 )
	endif( )

	set( rocblas_VERSION "${rocblas_VERSION_MAJOR}.${rocblas_VERSION_MINOR}.${rocblas_VERSION_PATCH}.${rocblas_VERSION_TWEAK}")
endif( )

# MACOSX_RPATH is enabled by default policy
if( POLICY CMP0042 )
	cmake_policy( SET CMP0042 NEW )
endif( )

message( STATUS "rocblas_VERSION= ${rocblas_VERSION}" )
message( STATUS "CMAKE_BUILD_TYPE= ${CMAKE_BUILD_TYPE}" )

# This is incremented when the ABI to the library changes
set( rocblas_SOVERSION 1 )

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake )

# This option only works for make/nmake and the ninja generators, but no reason it shouldn't be on all the time
# This tells cmake to create a compile_commands.json file that can be used with clang tooling or vim
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

# On windows, it's convenient to change the default install prefix such that it does NOT point to 'program files'
# Need to check out CMAKE_RUNTIME_OUTPUT_DIRECTORY variable, and see if that eliminates the need to modify install path
if( WIN32 AND CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT )
	set( CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/package" CACHE PATH "Install path prefix, prepended onto install directories" FORCE )
endif( )

# Set common compile and link options
if( MSVC )
	# Following options for nMake
	message( STATUS "Detected MSVS Ver: " ${MSVC_VERSION} )

	# CMake uses huge stack frames for windows, we would like to remove.
	string( REGEX REPLACE "/STACK:[0-9]+" "" CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}" )
	string( REGEX REPLACE "/STACK:[0-9]+" "" CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}" )
	string( REGEX REPLACE "/STACK:[0-9]+" "" CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS}" )
endif( )

include ( build-bitness )

# Print out compiler flags for viewing/debug
message( STATUS "CMAKE_CXX_COMPILER flags: " ${CMAKE_CXX_FLAGS} )
message( STATUS "CMAKE_CXX_COMPILER debug flags: " ${CMAKE_CXX_FLAGS_DEBUG} )
message( STATUS "CMAKE_CXX_COMPILER release flags: " ${CMAKE_CXX_FLAGS_RELEASE} )
message( STATUS "CMAKE_CXX_COMPILER relwithdebinfo flags: " ${CMAKE_CXX_FLAGS_RELWITHDEBINFO} )
message( STATUS "CMAKE_EXE_LINKER link flags: " ${CMAKE_EXE_LINKER_FLAGS} )

include( GNUInstallDirs )

# configure a header file to pass the CMake version settings to the source, and package the header files in the output archive
configure_file( "${PROJECT_SOURCE_DIR}/include/rocblas-version.h.in" "${PROJECT_BINARY_DIR}/include/rocblas-version.h" )

set( rocblas_headers_public
  include/rocblas.h
  include/rocblas_types.h
  include/rocblas_auxiliary.h
  include/rocblas_template_functions.h
  include/rocblas_functions.h
  ${PROJECT_BINARY_DIR}/include/rocblas-version.h
)

source_group( "Header Files\\Public" FILES ${rocblas_headers_public} )

# Build into subdirectories
add_subdirectory( src )

install( FILES
		${rocblas_headers_public}
	DESTINATION
		${CMAKE_INSTALL_INCLUDEDIR} )

# Depending on whether we are building for 64 or 32 bit, construct common paths and names that subdirectories can reference for their use
if( BUILD_64 )
	set( CPACK_PACKAGE_NAME "${CMAKE_PROJECT_NAME}-amd64" )
else( )
	set( CPACK_PACKAGE_NAME "${CMAKE_PROJECT_NAME}-i686" )
endif( )

# The following code is setting variables to control the behavior of CPack to generate our
if( WIN32 )
	set( CPACK_SOURCE_GENERATOR "ZIP" )
	set( CPACK_GENERATOR "ZIP" )
else( )
	set( CPACK_SOURCE_GENERATOR "TGZ" )

	set( CPACK_GENERATOR "DEB" )
	set( CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64" )
	set( CPACK_DEBIAN_COMPRESSION_TYPE "xz" )
	# set( CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON )
endif( )

set( CPACK_PACKAGE_CONTACT "Kent Knox <kent.knox@amd.com>" )
set( CPACK_PACKAGE_DESCRIPTION_SUMMARY "Radeon Open Compute BLAS library")
set( CPACK_PACKAGE_VERSION_MAJOR ${rocblas_VERSION_MAJOR} )
set( CPACK_PACKAGE_VERSION_MINOR ${rocblas_VERSION_MINOR} )
set( CPACK_PACKAGE_VERSION_PATCH ${rocblas_VERSION_PATCH} )
set( CPACK_PACKAGE_VERSION_TWEAK ${rocblas_VERSION_TWEAK} )
set( CPACK_PACKAGE_VERSION ${rocblas_VERSION} )
set( CPACK_PACKAGE_VENDOR "AMD")
set( CPACK_SOURCE_IGNORE_FILES "/\\\\.git/" )

# Define all variables that influence CPack before including CPack, such as install targets
include( CPack )
