# ########################################################################
# Copyright 2016 Advanced Micro Devices, Inc.
# ########################################################################

# Natively available on including Ubuntu 14.04, OpenSUSE 13.2, CentOS 6.6
cmake_minimum_required( VERSION 2.8.12 )

# On windows, it's convenient to change the default install prefix such that it does NOT point to 'program files'
if( WIN32 AND CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT )
  set( CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/library-package" CACHE PATH "Install path prefix, prepended onto install directories" FORCE )
endif( )

# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if( NOT DEFINED CMAKE_CONFIGURATION_TYPES AND NOT DEFINED CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE "Release" CACHE STRINGS "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel.")
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS None Debug Release RelWithDebInfo MinSizeRel)
endif()

# Append our library helper cmake path and the cmake path for hip (for convenience)
# Users may override HIP path by specifying their own in CMAKE_MODULE_PATH
list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake /opt/rocm/hip/cmake )

# The superbuild does not build anything itself, all compiling is done in external projects
project( rocblas-superbuild NONE )

# Default ON settings so cmake builds library and samples
option( BUILD_LIBRARY "Build rocBLAS library" ON )
option( BUILD_WITH_TENSILE "Building rocBLAS with Tensile or not" ON)

option( BUILD_CLIENTS "Build rocBLAS clients" ON )
option( BUILD_CLIENTS_SAMPLES "Build rocBLAS client samples" ON )

# which benchmark solution schedule
set( Tensile_ROOT "" CACHE STRING "Local path of Tensile.")
if ( BUILD_WITH_TENSILE )
  set( Tensile_LOGIC "FijiROCm15" CACHE STRING "Tensile to use which logic?")
  set_property( CACHE Tensile_LOGIC PROPERTY STRINGS
    FijiROCm15
    )
  option( Tensile_MERGE_FILES "Tensile to merge kernels and solutions files?" ON)
  option( Tensile_SHORT_FILENAMES "Tensile to use short file names? Use if compiler complains they're too long." OFF)
  option( Tensile_PRINT_DEBUG "Tensile to print runtime debug info?" OFF)
endif()

# BUILD_SHARED_LIBS is a cmake built-in; we make it an explicit option such that it shows in cmake-gui
option( BUILD_SHARED_LIBS "Build rocBLAS as a shared library" ON )

set( HIP_ROOT /opt/rocm/hip CACHE PATH "Specify hip installation dir")
set( BOOST_ROOT /opt/boost  CACHE PATH "Specify boost installation dir")

# Default behavior is to NOT install library, but clients may overload
set( rocblas_INSTALL_COMMAND INSTALL_COMMAND ${CMAKE_COMMAND} -E echo_append )

# Clients are programs provided in this repository, that make use of the library as a library client.  This can include
# but is not limited to benchmarks, tests and samples.
if( BUILD_CLIENTS )
  # Clients need to find and link rocblas; we install it locally instead of globally
  unset( rocblas_INSTALL_COMMAND )
endif()

# standard cmake module to manage dependencies
include( ExternalProject )

# This captures all of the dependencies cmake builds itself
set( rocblas_dependencies )

if( NOT DEFINED DEVICE_CXX_COMPILER )
  find_package( HIP REQUIRED )
  set( DEVICE_CXX_COMPILER ${HIP_ROOT_DIR}/bin/hipcc )
endif()

if( BUILD_LIBRARY )
  if( BUILD_WITH_TENSILE )
    # defines
    # Tensile_INCLUDE_DIRS
    # TensileLib_LIBRARIES
    # TensileLogger_LIBRARIES
    if (Tensile_ROOT)
      message(STATUS "Tensile_ROOT=${Tensile_ROOT} specified")
    else()
      find_package(Tensile QUIET)
      if (Tensile_FOUND)
        message(STATUS "Tensile package found.")
      else()
        include(cmake/external-Tensile.cmake )
        list( APPEND rocblas_dependencies Tensile )
        list( APPEND CMAKE_PREFIX_PATH ${Tensile_ROOT} )
        message( STATUS "Tensile not installed; will download to Tensile_ROOT=${Tensile_ROOT}" )
      endif()
    endif()
  endif()

  # Make sure LIBRARY_CXX_FLAGS begins with whitespace so that the flags do not merge with CMAKE_CXX_FLAGS
  set( LIBRARY_CXX_FLAGS " --amdgpu-target=gfx803 --amdgpu-target=gfx900" CACHE STRING "hcc specific compiler flags for rocblas" )

  # WARNING: do not surround CMAKE_PREFIX_PATH with quotes, it breaks
  # Replace all occurances of ; with ^^, which we elect to use a path seperator
  string(REGEX REPLACE ";" "^^" LIBRARY_PREFIX_PATH "${CMAKE_PREFIX_PATH}" )
  string(REGEX REPLACE ";" "^^" LIBRARY_MODULE_PATH "${CMAKE_MODULE_PATH}" )

  set( LIBRARY_CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
    -DCMAKE_PREFIX_PATH=${LIBRARY_PREFIX_PATH}
    -DCMAKE_MODULE_PATH=${LIBRARY_MODULE_PATH}
    -DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}
    -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}${LIBRARY_CXX_FLAGS}
    -DBUILD_WITH_TENSILE=${BUILD_WITH_TENSILE}
    -DTensile_LOGIC=${Tensile_LOGIC}
    -DTensile_MERGE_FILES=${Tensile_MERGE_FILES}
    -DTensile_SHORT_FILENAMES=${Tensile_SHORT_FILENAMES}
    -DTensile_PRINT_DEBUG=${Tensile_PRINT_DEBUG}
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
  )

  # The library uses the c++ compiler to compile device code
  if( DEFINED DEVICE_CXX_COMPILER )
    message( STATUS "DEVICE_CXX_COMPILER: ${DEVICE_CXX_COMPILER} " )
    list( APPEND LIBRARY_CMAKE_ARGS -DCMAKE_CXX_COMPILER=${DEVICE_CXX_COMPILER} )
  endif( )

  # Normalize the different ways of specifying a c compiler through -DCMAKE_C_COMPILER
  if( DEFINED CMAKE_C_COMPILER )
    message( STATUS "CMAKE_C_COMPILER: ${CMAKE_C_COMPILER}" )
    list( APPEND LIBRARY_CMAKE_ARGS -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} )
  elseif( DEFINED ENV{CC} )
    message( STATUS "ENV{CC}: $ENV{CC} " )
    list( APPEND LIBRARY_CMAKE_ARGS -DCMAKE_C_COMPILER=$ENV{CC} )
  endif( )

  if( DEFINED CPACK_PACKAGING_INSTALL_PREFIX )
    list( APPEND LIBRARY_CMAKE_ARGS -DCPACK_PACKAGING_INSTALL_PREFIX=${CPACK_PACKAGING_INSTALL_PREFIX} )
  endif()

  if (Tensile_FOUND)
    set( LIBRARY_CMAKE_ARGS ${LIBRARY_CMAKE_ARGS} -DTensile_FOUND=${Tensile_FOUND})
  else()
    set( LIBRARY_CMAKE_ARGS ${LIBRARY_CMAKE_ARGS} -DTensile_ROOT=${Tensile_ROOT})
  endif()

  if( CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT )
    set( LIBRARY_INSTALL_DIR library-package )
  else( )
    set( LIBRARY_INSTALL_DIR ${CMAKE_INSTALL_PREFIX} )
  endif( )

  # Build the library as an external project
  ExternalProject_Add( rocblas
    DEPENDS ${rocblas_dependencies}
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/library
    BINARY_DIR library-build
    INSTALL_DIR ${LIBRARY_INSTALL_DIR}
    LIST_SEPARATOR ^^
    CMAKE_ARGS ${LIBRARY_CMAKE_ARGS}
    ${rocblas_INSTALL_COMMAND}
  )
endif( )

# Build clients of the library
if( BUILD_CLIENTS )
  include( clients/cmake/build-options.cmake )

  if( BUILD_LIBRARY )
    ExternalProject_Get_Property( rocblas install_dir )
    list( APPEND CMAKE_PREFIX_PATH ${install_dir} )
  endif( )

  # If the c++ compiler used to compile the clients is hipcc/hcc, make sure to specify the desired targets
  if( (CMAKE_CXX_COMPILER MATCHES "hipcc$") OR (CMAKE_CXX_COMPILER MATCHES "hcc$") )
    set( CLIENT_CXX_FLAGS " --amdgpu-target=gfx803 --amdgpu-target=gfx900" CACHE STRING "hcc specific compiler flags for rocblas clients" )
  else( )
    set( CLIENT_CXX_FLAGS "" CACHE STRING "hcc specific compiler flags for rocblas clients" )
  endif( )

  # WARNING: do not surround CMAKE_PREFIX_PATH with quotes, it breaks
  # Replace all occurances of ; with ^^, which we elect to use a path seperator
  string(REGEX REPLACE ";" "^^" CLIENT_PREFIX_PATH "${CMAKE_PREFIX_PATH}" )
  string(REGEX REPLACE ";" "^^" CLIENT_MODULE_PATH "${CMAKE_MODULE_PATH}" )

  # Default arguments that get passed down into all external projects
  set( CLIENTS_CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
    -DDEVICE_CXX_COMPILER=${DEVICE_CXX_COMPILER}
    -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}${CLIENT_CXX_FLAGS}
    -DCMAKE_PREFIX_PATH=${CLIENT_PREFIX_PATH}
    -DCMAKE_MODULE_PATH=${CLIENT_MODULE_PATH}
    -DBUILD_CLIENTS_SAMPLES=${BUILD_CLIENTS_SAMPLES}
    -DBUILD_CLIENTS_BENCHMARKS=${BUILD_CLIENTS_BENCHMARKS}
    -DBUILD_CLIENTS_TESTS=${BUILD_CLIENTS_TESTS}
    -DBUILD_WITH_TENSILE=${BUILD_WITH_TENSILE}
    -DDEVICE_CXX_COMPILER=${DEVICE_CXX_COMPILER}
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
  )

  # Normalize the different ways of specifying a c++ compiler through -DCMAKE_CXX_COMPILER
  if( DEFINED CMAKE_CXX_COMPILER )
    message( STATUS "CMAKE_CXX_COMPILER[clients]: ${CMAKE_CXX_COMPILER} " )
    list( APPEND CLIENTS_CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} )
  elseif( DEFINED ENV{CXX} )
    message( STATUS "ENV{CXX}: $ENV{CXX} " )
    list( APPEND CLIENTS_CMAKE_ARGS -DCMAKE_CXX_COMPILER=$ENV{CXX} )
  endif( )

  # Normalize the different ways of specifying a c compiler through -DCMAKE_C_COMPILER
  if( DEFINED CMAKE_C_COMPILER )
    message( STATUS "CMAKE_C_COMPILER[clients]: ${CMAKE_C_COMPILER}" )
    list( APPEND CLIENTS_CMAKE_ARGS -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} )
  elseif( DEFINED ENV{CC} )
    message( STATUS "ENV{CC}: $ENV{CC} " )
    list( APPEND CLIENTS_CMAKE_ARGS -DCMAKE_C_COMPILER=$ENV{CC} )
  endif( )

  # Clients are set up as an external project to take advantage of specifying toolchain files.
  # We want cmake to go through it's usual discovery process
  ExternalProject_Add( rocblas-clients
    DEPENDS rocblas
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/clients
    BINARY_DIR clients-build
    INSTALL_DIR clients-package
    LIST_SEPARATOR ^^
    CMAKE_ARGS ${CLIENTS_CMAKE_ARGS}
    INSTALL_COMMAND ""
  )
endif( )

# POLICY CMP0037 - "Target names should not be reserved and should match a validity pattern"
# target names like 'install' should be OK at the superbuild level
if( POLICY CMP0037 )
  cmake_policy( SET CMP0037 OLD )
endif( )

add_custom_target( install
  COMMAND ${CMAKE_COMMAND} --build . --target install
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/library-build"
  DEPENDS rocblas
  COMMENT "super-build installing rocblas"
  VERBATIM
)

add_custom_target( package
  COMMAND ${CMAKE_COMMAND} --build . --target package
  COMMAND ${CMAKE_COMMAND} -E copy *.deb *.rpm ..
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/library-build"
  DEPENDS rocblas
  COMMENT "super-build packaging rocblas"
#  VERBATIM  // non-intuitively, VERBATIM breaks the cmake -E copy
)
