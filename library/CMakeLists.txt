# ########################################################################
# Copyright 2016-2019 Advanced Micro Devices, Inc.
# ########################################################################

# The following helper functions wrap common cmake functions.  They are
# used to cope with a few wierdnesses of hipcc/nvcc.
# ########################################################################
# HELPER FUNCTIONS
# ########################################################################

# ########################################################################
# target_compile_features() override
# ########################################################################
function( target_compile_features target_name )
  # With Cmake v3.5, hipcc (with nvcc backend) does not work with target_compile_features
  # Turn on -std=c++14 manually
  if( CMAKE_CXX_COMPILER MATCHES ".*/hipcc$" )
    set_target_properties( ${target_name} PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED ON )
  else( )
    _target_compile_features( ${target_name} ${ARGN} )
  endif( )
endfunction( )

# ########################################################################
# target_link_libraries() override
# ########################################################################
function( target_link_libraries target_name )
  # hipcc takes care of finding hip library dependencies internally; remove
  # explicit mentions of them so cmake doesn't complain on nvcc path
  if( CMAKE_CXX_COMPILER MATCHES ".*/hipcc$" )
    foreach( link_library ${ARGN} )
      if( (link_library MATCHES "^hip::|^hcc::") )
      else( )
        if( TARGET ${link_library} )
          list( APPEND new_list -Xlinker ${link_library} )
        else( )
          list( APPEND new_list ${link_library} )
        endif( )
      endif( )
    endforeach( )
    _target_link_libraries( ${target_name} ${new_list} )
  else( )
    _target_link_libraries( ${target_name} ${ARGN} )
  endif( )
endfunction( )

# ########################################################################
# Main
# ########################################################################

# This is incremented when the ABI to the library changes
set( rocblas_SOVERSION 0.1 )

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake )

# This option only works for make/nmake and the ninja generators, but no reason it shouldn't be on all the time
# This tells cmake to create a compile_commands.json file that can be used with clang tooling or vim
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

# set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-Bsymbolic")

# include( build-bitness )

# Print out compiler flags for viewing/debug
if( BUILD_VERBOSE )
  message( STATUS "rocfft_VERSION: ${rocfft_VERSION}" )
  message( STATUS "\t==>CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}" )
  message( STATUS "\t==>BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS}" )
  message( STATUS "\t==>CMAKE_INSTALL_PREFIX link: " ${CMAKE_INSTALL_PREFIX} )
  message( STATUS "\t==>CMAKE_MODULE_PATH link: " ${CMAKE_MODULE_PATH} )
  message( STATUS "\t==>CMAKE_PREFIX_PATH link: " ${CMAKE_PREFIX_PATH} )
  message( STATUS "==============" )
  message( STATUS "\t==>CMAKE_CXX_COMPILER: " ${CMAKE_CXX_FLAGS} )
  message( STATUS "\t==>CMAKE_CXX_COMPILER debug: " ${CMAKE_CXX_FLAGS_DEBUG} )
  message( STATUS "\t==>CMAKE_CXX_COMPILER release: " ${CMAKE_CXX_FLAGS_RELEASE} )
  message( STATUS "\t==>CMAKE_CXX_COMPILER relwithdebinfo: " ${CMAKE_CXX_FLAGS_RELWITHDEBINFO} )
  message( STATUS "\t==>CMAKE_EXE_LINKER_FLAGS: " ${CMAKE_EXE_LINKER_FLAGS} )
  message( STATUS "\t==>CMAKE_EXE_LINKER_FLAGS_RELEASE: " ${CMAKE_EXE_LINKER_FLAGS_RELEASE} )
  message( STATUS "\t==>CMAKE_SHARED_LINKER_FLAGS: " ${CMAKE_SHARED_LINKER_FLAGS} )
  message( STATUS "\t==>CMAKE_SHARED_LINKER_FLAGS_RELEASE: " ${CMAKE_SHARED_LINKER_FLAGS_RELEASE} )
  message( STATUS "==============" )
  message( STATUS "\t==>CMAKE_SHARED_LIBRARY_C_FLAGS: ${CMAKE_SHARED_LIBRARY_C_FLAGS}" )
  message( STATUS "\t==>CMAKE_SHARED_LIBRARY_CXX_FLAGS: ${CMAKE_SHARED_LIBRARY_CXX_FLAGS}" )
  message( STATUS "\t==>CMAKE_SHARED_LINKER_FLAGS: ${CMAKE_SHARED_LINKER_FLAGS}" )
  message( STATUS "\t==>CMAKE_SHARED_LINKER_FLAGS_DEBUG: ${CMAKE_SHARED_LINKER_FLAGS_DEBUG}" )
  message( STATUS "\t==>CMAKE_SHARED_LINKER_FLAGS_RELEASE: ${CMAKE_SHARED_LINKER_FLAGS_RELEASE}" )
endif( )

# configure a header file to pass the CMake version settings to the source, and package the header files in the output archive
configure_file( "${CMAKE_CURRENT_SOURCE_DIR}/include/rocblas-version.h.in" "${PROJECT_BINARY_DIR}/include/rocblas-version.h" )

set( rocblas_headers_public
  include/rocblas.h
  #  include/rocblas.hpp
  include/rocblas-types.h
  include/rocblas_bfloat16.h
  include/rocblas-auxiliary.h
  include/rocblas-functions.h
  ${PROJECT_BINARY_DIR}/include/rocblas-version.h
)

source_group( "Header Files\\Public" FILES ${rocblas_headers_public} )

include( GNUInstallDirs )

set( BIN_INSTALL_DIR ${CMAKE_INSTALL_BINDIR} )
set( LIB_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR} )
set( INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR} )

# Build into subdirectories
add_subdirectory( src )

# The following code is setting variables to control the behavior of CPack to generate our
# if( WIN32 )
#     set( CPACK_SOURCE_GENERATOR "ZIP" )
#     set( CPACK_GENERATOR "ZIP" )
# else( )
#     set( CPACK_SOURCE_GENERATOR "TGZ" )
#     set( CPACK_GENERATOR "DEB;RPM" CACHE STRING "cpack list: 7Z, DEB, IFW, NSIS, NSIS64, RPM, STGZ, TBZ2, TGZ, TXZ, TZ, ZIP" )
#     # set( CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON )
# endif( )

# Package specific CPACK vars
set( CPACK_DEBIAN_PACKAGE_DEPENDS "hip_hcc (>= 1.3)" )
set( CPACK_RPM_PACKAGE_REQUIRES "hip_hcc >= 1.3" )
set( CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE.md" )

if( NOT CPACK_PACKAGING_INSTALL_PREFIX )
  set( CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}" )
endif( )

set( CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}" "\${CPACK_PACKAGING_INSTALL_PREFIX}/include" "\${CPACK_PACKAGING_INSTALL_PREFIX}/lib" )

# Give rocblas compiled for CUDA backend a different name
if( CMAKE_CXX_COMPILER MATCHES ".*/hcc$" OR CXX_VERSION_STRING MATCHES "clang" )
    set( package_name rocblas )
else( )
    set( package_name rocblas-alt )
endif( )

set( ROCBLAS_CONFIG_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" CACHE PATH "Path placed into ldconfig file" )

rocm_create_package(
    NAME ${package_name}
    DESCRIPTION "Radeon Open Compute BLAS library"
    MAINTAINER "Kent Knox <kent.knox@amd.com>"
    LDCONFIG
    LDCONFIG_DIR ${ROCBLAS_CONFIG_DIR}
)
