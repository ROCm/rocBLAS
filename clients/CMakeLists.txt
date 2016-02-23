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
  set( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." )
endif()

# Check if cmake supports the new VERSION tag for project() commands
# rocBLAS becomes the name of the project with a particular version
if( POLICY CMP0048 )
	cmake_policy( SET CMP0048 NEW )
	project( rocBLAS-clients VERSION 0.0.0.1 LANGUAGES C CXX )
else( )
	project( rocBLAS-clients C CXX )
	# Define a version for the code
	if( NOT DEFINED rocBLAS-clients_VERSION_MAJOR )
		set( rocBLAS-clients_VERSION_MAJOR 0 )
	endif( )

	if( NOT DEFINED rocBLAS-clients_VERSION_MINOR )
		set( rocBLAS-clients_VERSION_MINOR 0 )
	endif( )

	if( NOT DEFINED rocBLAS-clients_VERSION_PATCH )
		set( rocBLAS-clients_VERSION_PATCH 0 )
	endif( )

	if( NOT DEFINED rocBLAS-clients_VERSION_TWEAK )
		set( rocBLAS-clients_VERSION_TWEAK 1 )
	endif( )

	set( rocBLAS-clients_VERSION "${rocBLAS-clients_VERSION_MAJOR}.${rocBLAS-clients_VERSION_MINOR}.${rocBLAS-clients_VERSION_PATCH}.${rocBLAS-clients_VERSION_TWEAK}")
endif( )

message( STATUS "rocBLAS-clients_VERSION= ${rocBLAS-clients_VERSION}" )
message( STATUS "CMAKE_BUILD_TYPE= ${CMAKE_BUILD_TYPE}" )

# This is incremented when the ABI to the library changes
set( rocBLAS-clients_SOVERSION 1 )

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake )

# This option only works for make/nmake and the ninja generators, but no reason it shouldn't be on all the time
# This tells cmake to create a compile_commands.json file that can be used with clang tooling or vim
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

# On windows, it's convenient to change the default install prefix such that it does NOT point to 'program files'
# Need to check out CMAKE_RUNTIME_OUTPUT_DIRECTORY variable, and see if that eliminates the need to modify install path
if( WIN32 AND CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT )
	set( CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/package" CACHE PATH "Install path prefix, prepended onto install directories" FORCE )
endif( )

if( MSVC_IDE )
	set( rocBLAS_BUILD64 ${CMAKE_CL_64} )
	set_property( GLOBAL PROPERTY USE_FOLDERS TRUE )
else()
	option( rocBLAS_BUILD64 "Build a 64-bit product" ON )
endif()

# These variables are meant to contain string which should be appended to the installation paths
# of library and executable binaries, respectively.  They are meant to be user configurable/overridable.
set( SUFFIX_LIB_DEFAULT "" )
set( SUFFIX_BIN_DEFAULT "" )

# Modify the global find property to help us find libraries like Boost in the correct paths for 64-bit
# Essentially, find_library calls will look for /lib64 instead of /lib; works for windows and linux
if( rocBLAS_BUILD64 )
	set_property( GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS TRUE )
	message( STATUS "64bit build - FIND_LIBRARY_USE_LIB64_PATHS TRUE" )

	if( NOT APPLE )
		set( SUFFIX_LIB_DEFAULT "64" )
	endif( )
else( )
	set_property( GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS FALSE )
	message( STATUS "32bit build - FIND_LIBRARY_USE_LIB64_PATHS FALSE" )
endif( )

set( SUFFIX_LIB ${SUFFIX_LIB_DEFAULT} CACHE STRING "String to append to 'lib' install path" )
set( SUFFIX_BIN ${SUFFIX_BIN_DEFAULT} CACHE STRING "String to append to 'bin' install path" )

# Currently, linux has a problem outputing both narrow and wide characters,
# which happens in our client because openCL only supports narrow characters
if( WIN32 )
	option( BUILD_UNICODE "Create a solution that compiles rocBLAS with Unicode Support" ON )
	if( BUILD_UNICODE )
		message( STATUS "UNICODE build" )
	endif( )
else()
	set( BUILD_UNICODE OFF )
	message( STATUS "UNICODE disabled on linux" )
endif()

# avoid warning on newer cmake versions
if( APPLE )
  set( CMAKE_MACOSX_RPATH 0 )
endif( )

# Set common compile and link options
if( MSVC )
	# Following options for nMake
	message( STATUS "Detected MSVS Ver: " ${MSVC_VERSION} )

	# CMake uses huge stack frames for windows, for some reason.  We remove.
	string( REGEX REPLACE "/STACK:[0-9]+" "" CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}" )
	string( REGEX REPLACE "/STACK:[0-9]+" "" CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}" )
	string( REGEX REPLACE "/STACK:[0-9]+" "" CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS}" )

	if( BUILD_StripSymbols )
		string( REGEX REPLACE "/Zi" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}" )
	endif()
elseif( CMAKE_COMPILER_IS_GNUCC )
	message( STATUS "Detected GNU compiler collection" )

	if( rocBLAS_BUILD64 )
		set( CMAKE_CXX_FLAGS "-m64 ${CMAKE_CXX_FLAGS}" )
		set( CMAKE_C_FLAGS "-m64 ${CMAKE_C_FLAGS}" )
	else( )
		set( CMAKE_CXX_FLAGS "-m32 ${CMAKE_CXX_FLAGS}" )
		set( CMAKE_C_FLAGS "-m32 ${CMAKE_C_FLAGS}" )
	endif( )
else( )
	message( AUTHOR_WARNING "Compiler not recognized.  Using default flags." )
endif( )

# If UNICODE is defined, pass extra definitions into
if( BUILD_UNICODE )
	add_definitions( "/DUNICODE /D_UNICODE" )
endif( )

# Print out compiler flags for viewing/debug
message( STATUS "CMAKE_CXX_COMPILER flags: " ${CMAKE_CXX_FLAGS} )
message( STATUS "CMAKE_CXX_COMPILER debug flags: " ${CMAKE_CXX_FLAGS_DEBUG} )
message( STATUS "CMAKE_CXX_COMPILER release flags: " ${CMAKE_CXX_FLAGS_RELEASE} )
message( STATUS "CMAKE_CXX_COMPILER relwithdebinfo flags: " ${CMAKE_CXX_FLAGS_RELWITHDEBINFO} )
message( STATUS "CMAKE_EXE_LINKER link flags: " ${CMAKE_EXE_LINKER_FLAGS} )

include( build-options )

if( BUILD_BENCHMARKS )
	ExternalProject_Add( benchmarks
		DEPENDS rocBLAS
		SOURCE_DIR ${PROJECT_SOURCE_DIR}/benchmarks
		BINARY_DIR benchmarks-build
		CMAKE_ARGS ...
		INSTALL_COMMAND ""
	)
endif( )

if( BUILD_TESTS )
	ExternalProject_Add( tests
		DEPENDS rocBLAS
		SOURCE_DIR ${PROJECT_SOURCE_DIR}/tests
		BINARY_DIR tests-build
		CMAKE_ARGS ...
		INSTALL_COMMAND ""
	)
endif( )

if( BUILD_SAMPLES )
	ExternalProject_Add( samples
		DEPENDS rocBLAS
		SOURCE_DIR ${PROJECT_SOURCE_DIR}/samples
		BINARY_DIR samples-build
		CMAKE_ARGS ...
		INSTALL_COMMAND ""
	)
endif( )
