# ########################################################################
# Copyright 2016 Advanced Micro Devices, Inc.
# ########################################################################

message( STATUS "Configuring boost external dependency" )
include( ExternalProject )

# TODO:  Options should be added to allow downloading Boost straight from github

# This file is used to add Boost as a library dependency to another project
# This sets up boost to download from sourceforge, and builds it as a cmake
# ExternalProject

# Change this one line to upgrade to newer versions of boost
set( ext.Boost_VERSION "1.60.0" CACHE STRING "Boost version to download/use" )
mark_as_advanced( ext.Boost_VERSION )
string( REPLACE "." "_" ext.Boost_Version_Underscore ${ext.Boost_VERSION} )

message( STATUS "ext.Boost_VERSION: " ${ext.Boost_VERSION} )

if( WIN32 )
  # For newer cmake versions, 7z archives are much smaller to download
  if( CMAKE_VERSION VERSION_LESS "3.1.0" )
    set( Boost_Ext "zip" )
  else( )
    set( Boost_Ext "7z" )
  endif( )
else( )
  set( Boost_Ext "tar.bz2" )
endif( )

if( WIN32 )
  set( Boost.Command b2 --prefix=<INSTALL_DIR>/package )
else( )
  set( Boost.Command ./b2 --prefix=<INSTALL_DIR>/package )
endif( )

if( CMAKE_COMPILER_IS_GNUCXX )
  list( APPEND Boost.Command cxxflags=-fPIC -std=c++11 )
elseif( XCODE_VERSION OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang") )
  list( APPEND Boost.Command cxxflags=-std=c++11 -stdlib=libc++ linkflags=-stdlib=libc++ )
endif( )

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

message( STATUS "ExternalBoost using ( " ${Cores} " ) cores to build with" )
message( STATUS "ExternalBoost building [ program_options, serialization, filesystem, system, regex ] components" )

list( APPEND Boost.Command -j ${Cores} --with-program_options --with-serialization --with-filesystem --with-system --with-regex )

if( BUILD_64 )
  list( APPEND Boost.Command address-model=64 )
else( )
  list( APPEND Boost.Command address-model=32 )
endif( )

if( MSVC10 )
  list( APPEND Boost.Command toolset=msvc-10.0 )
elseif( MSVC11 )
  list( APPEND Boost.Command toolset=msvc-11.0 )
elseif( MSVC12 )
  list( APPEND Boost.Command toolset=msvc-12.0 )
elseif( MSVC14 )
  list( APPEND Boost.Command toolset=msvc-14.0 )
elseif( XCODE_VERSION OR ( CMAKE_CXX_COMPILER_ID MATCHES "Clang" ) )
  list( APPEND Boost.Command toolset=clang )
elseif( NOT "$ENV{CC}" STREQUAL "" )
  # CMake apprarently puts the full path of the compiler into CC
  # The user might specify a non-default gcc compiler through ENV
  message( STATUS "ENV{CC}=$ENV{CC}" )
  get_filename_component( gccToolset $ENV{CC} NAME )

  # see: https://svn.boost.org/trac/boost/ticket/5917
  string( TOLOWER ${gccToolset} gccToolset )
  if( gccToolset STREQUAL "cc")
    set( gccToolset "gcc" )
  endif( )
  list( APPEND Boost.Command toolset=${gccToolset} )
endif( )

if( WIN32 AND (ext.Boost_VERSION VERSION_LESS "1.60.0") )
  list( APPEND Boost.Command define=BOOST_LOG_USE_WINNT6_API )
endif( )

set( ext.Boost_LINK "static" CACHE STRING "Which boost link method?  static | shared | static,shared" )
mark_as_advanced( ext.Boost_LINK )

if( WIN32 )
    # Versioned is the default on windows
    set( ext.Boost_LAYOUT "versioned" CACHE STRING "Which boost layout method?  versioned | tagged | system" )

    # For windows, default to build both variants to support the VS IDE
    set( ext.Boost_VARIANT "debug,release" CACHE STRING "Which boost variant?  debug | release | debug,release" )
else( )
    # Tagged builds provide unique enough names to be able to build both variants
    set( ext.Boost_LAYOUT "tagged" CACHE STRING "Which boost layout method?  versioned | tagged | system" )

   # For Linux, typically a build tree only needs one variant
   if( ${CMAKE_BUILD_TYPE} STREQUAL "Debug")
     set( ext.Boost_VARIANT "debug" CACHE STRING "Which boost variant?  debug | release | debug,release" )
   else( )
     set( ext.Boost_VARIANT "release" CACHE STRING "Which boost variant?  debug | release | debug,release" )
   endif( )
endif( )
mark_as_advanced( ext.Boost_LAYOUT )
mark_as_advanced( ext.Boost_VARIANT )

list( APPEND Boost.Command link=${ext.Boost_LINK} variant=${ext.Boost_VARIANT} --layout=${ext.Boost_LAYOUT} install )

message( STATUS "Boost.Command: ${Boost.Command}" )

# If the user has a cached local copy stored somewhere, they can define the full path to the package in a BOOST_URL environment variable
if( DEFINED ENV{BOOST_URL} )
  set( ext.Boost_URL "$ENV{BOOST_URL}" CACHE STRING "URL to download Boost from" )
else( )
  set( ext.Boost_URL "http://sourceforge.net/projects/boost/files/boost/${ext.Boost_VERSION}/boost_${ext.Boost_Version_Underscore}.${Boost_Ext}/download" CACHE STRING "URL to download Boost from" )
endif( )
mark_as_advanced( ext.Boost_URL )

set( Boost.Bootstrap "" )
set( ext.MD5_HASH "" )
if( WIN32 )
  set( Boost.Bootstrap "bootstrap.bat" )

  if( CMAKE_VERSION VERSION_LESS "3.1.0" )
    # .zip file
    set( ext.MD5_HASH "0cc5b9cf9ccdf26945b225c7338b4288" )
  else( )
    # .7z file
    set( ext.MD5_HASH "7ce7f5a4e396484da8da6b60d4ed7661" )
  endif( )
else( )
  set( Boost.Bootstrap "./bootstrap.sh" )

  # .tar.bz2
  set( ext.MD5_HASH "65a840e1a0b13a558ff19eeb2c4f0cbe" )

  if( XCODE_VERSION OR ( CMAKE_CXX_COMPILER_ID MATCHES "Clang" ) )
    list( APPEND Boost.Bootstrap --with-toolset=clang )
  endif( )
endif( )

# Below is a fancy CMake command to download, build and install Boost on the users computer
ExternalProject_Add(
  boost
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/extern/boost
  URL ${ext.Boost_URL}
  URL_MD5 ${ext.MD5_HASH}
  UPDATE_COMMAND ${Boost.Bootstrap}
  LOG_UPDATE 1
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ${Boost.Command}
  BUILD_IN_SOURCE 1
  LOG_BUILD 1
  INSTALL_COMMAND ""
)

set_property( TARGET boost PROPERTY FOLDER "extern" )
ExternalProject_Get_Property( boost install_dir )

# For use by the user of ExternalBoost.cmake
set( BOOST_ROOT ${install_dir}/package )
