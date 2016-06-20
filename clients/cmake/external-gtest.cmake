# ########################################################################
# Copyright 2016 Advanced Micro Devices, Inc.
# ########################################################################

message( STATUS "Configuring gtest external dependency" )
include( ExternalProject )

# set( gtest_version "1.7.0" CACHE STRING "gtest version to download/use" )
# mark_as_advanced( gtest_version )
#
# message( STATUS "gtest_version: " ${gtest_version} )
#
# if( DEFINED ENV{GMOCK_URL} )
#   set( ext.gtest_URL "$ENV{GMOCK_URL}" CACHE STRING "URL to download gtest from" )
# else( )
#   set( ext.gtest_URL "https://github.com/google/googletest/archive/release-${gtest_version}.zip" CACHE STRING "URL to download gtest from" )
# endif( )
# mark_as_advanced( ext.gtest_URL )

set( gtest_git_repository "https://github.com/google/googletest.git" CACHE STRING "URL to download gtest from" )
set( gtest_git_tag "master" CACHE STRING "URL to download gtest from" )

# -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY:PATH=${LIB_DIR}
set( gtest_cmake_args -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>/package -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE} )

if( MSVC )
  list( APPEND gtest_cmake_args -Dgtest_force_shared_crt=ON )
else( )
  # GTEST_USE_OWN_TR1_TUPLE necessary to compile with hipcc
  set( EXTRA_FLAGS "-DGTEST_USE_OWN_TR1_TUPLE=1" )

  if( BUILD_64 )
    set( EXTRA_FLAGS "${EXTRA_FLAGS} -m64" )
  else( )
    set( EXTRA_FLAGS "${EXTRA_FLAGS} -m32" )
  endif( )

  list( APPEND gtest_cmake_args -DCMAKE_CXX_FLAGS=${EXTRA_FLAGS} )
endif( )

if( CMAKE_CONFIGURATION_TYPES )
  set( gtest_make
        COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --config Release
        COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --config Debug
  )
else( )
  # Add build thread in addition to the number of cores that we have
  include( ProcessorCount )
  ProcessorCount( Cores )

  # If we are not using an IDE, assume nmake with visual studio
  if( MSVC )
    set( gtest_make "nmake" )
  else( )
    set( gtest_make "make" )

    # The -j paramter does not work with nmake
    if( NOT Cores EQUAL 0 )
      math( EXPR Cores "${Cores} + 1 " )
      list( APPEND gtest_make -j ${Cores} )
    else( )
      # If we could not detect # of cores, assume 1 core and add an additional build thread
      list( APPEND gtest_make -j 2 )
    endif( )
  endif( )

  # WARNING: find_package( gtest ) only works if it can find release binaries
  # Even if you want to link against debug binaries, you must build release binaries too
  list( APPEND gtest_cmake_args -DCMAKE_BUILD_TYPE=Release )
  message( STATUS "ExternalGmock using ( " ${Cores} " ) cores to build with" )
endif( )

# message( STATUS "gtest_make ( " ${gtest_make} " ) " )
# message( STATUS "gtest_cmake_args ( " ${gtest_cmake_args} " ) " )

# Master branch has a new structure that combines googletest with googlemock
ExternalProject_Add(
  googletest
  PREFIX ${CMAKE_BINARY_DIR}/extern/gtest
  GIT_REPOSITORY ${gtest_git_repository}
  GIT_TAG ${gtest_git_tag}
  CMAKE_ARGS ${gtest_cmake_args}
  BUILD_COMMAND ${gtest_make}
  LOG_BUILD 1
  LOG_INSTALL 1
)

ExternalProject_Get_Property( googletest source_dir )

if( BUILD_64 )
  set( LIB_DIR lib64 )
else( )
  set( LIB_DIR lib )
endif( )

# For visual studio, the path 'debug' is hardcoded because that is the default VS configuration for a build.
# Doesn't matter if its the gtest or gtestd project above
set( package_dir "<INSTALL_DIR>/package" )
set( gtest_lib_dir "<BINARY_DIR>/${LIB_DIR}" )
if( CMAKE_CONFIGURATION_TYPES )
    # Create a package by bundling libraries and header files
    ExternalProject_Add_Step( googletest createPackage
      COMMAND ${CMAKE_COMMAND} -E copy_directory ${gtest_lib_dir}/Debug ${package_dir}/${LIB_DIR}
      COMMAND ${CMAKE_COMMAND} -E copy_directory ${gtest_lib_dir}/Release ${package_dir}/${LIB_DIR}
      COMMAND ${CMAKE_COMMAND} -E copy_directory ${gtest_lib_dir}/Debug ${package_dir}/${LIB_DIR}
      COMMAND ${CMAKE_COMMAND} -E copy_directory ${gtest_lib_dir}/Release ${package_dir}/${LIB_DIR}
      COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/include ${package_dir}/include
      COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/gtest/include/gtest ${package_dir}/include/gtest
      DEPENDEES install
    )
else( )
  if( BUILD_64 )
    ExternalProject_Add_Step( googletest rename_lib_dir
      COMMAND ${CMAKE_COMMAND} -E remove_directory ${package_dir}/${LIB_DIR}
      COMMAND ${CMAKE_COMMAND} -E rename ${package_dir}/lib ${package_dir}/${LIB_DIR}
      DEPENDEES install
    )
  endif( )
endif( )

set_property( TARGET googletest PROPERTY FOLDER "extern")
ExternalProject_Get_Property( googletest install_dir )

# For use by the user of ExternalGtest.cmake
set( GTEST_ROOT ${install_dir}/package )
