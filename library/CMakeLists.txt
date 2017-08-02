# ########################################################################
# Copyright 2016 Advanced Micro Devices, Inc.
# ########################################################################

# This is incremented when the ABI to the library changes
set( rocblas_SOVERSION 0 )

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake )

# This option only works for make/nmake and the ninja generators, but no reason it shouldn't be on all the time
# This tells cmake to create a compile_commands.json file that can be used with clang tooling or vim
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

# set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-Bsymbolic")

# include( build-bitness )

# Print out compiler flags for viewing/debug
if( BUILD_VERBOSE )
  message( STATUS "rocblas_VERSION: ${rocblas_VERSION}" )
  message( STATUS "\t==>CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}" )
  message( STATUS "\t==>BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS}" )
  message( STATUS "\t==>CMAKE_INSTALL_PREFIX link: " ${CMAKE_INSTALL_PREFIX} )
  message( STATUS "\t==>CMAKE_MODULE_PATH link: " ${CMAKE_MODULE_PATH} )
  message( STATUS "\t==>CMAKE_PREFIX_PATH link: " ${CMAKE_PREFIX_PATH} )


  message( STATUS "\t==>CMAKE_CXX_COMPILER flags: " ${CMAKE_CXX_FLAGS} )
  message( STATUS "\t==>CMAKE_CXX_COMPILER debug flags: " ${CMAKE_CXX_FLAGS_DEBUG} )
  message( STATUS "\t==>CMAKE_CXX_COMPILER release flags: " ${CMAKE_CXX_FLAGS_RELEASE} )
  message( STATUS "\t==>CMAKE_CXX_COMPILER relwithdebinfo flags: " ${CMAKE_CXX_FLAGS_RELWITHDEBINFO} )
  message( STATUS "\t==>CMAKE_EXE_LINKER link flags: " ${CMAKE_EXE_LINKER_FLAGS} )
endif( )

find_package( hip REQUIRED CONFIG PATHS /opt/rocm/hip )

# configure a header file to pass the CMake version settings to the source, and package the header files in the output archive
configure_file( "${CMAKE_CURRENT_SOURCE_DIR}/include/rocblas-version.h.in" "${PROJECT_BINARY_DIR}/include/rocblas-version.h" )

set( rocblas_headers_public
  include/rocblas.h
  #  include/rocblas.hpp
  include/rocblas-types.h
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
if( WIN32 )
    set( CPACK_SOURCE_GENERATOR "ZIP" )
    set( CPACK_GENERATOR "ZIP" )
else( )
    set( CPACK_SOURCE_GENERATOR "TGZ" )
    set( CPACK_GENERATOR "DEB;RPM" CACHE STRING "cpack list: 7Z, DEB, IFW, NSIS, NSIS64, RPM, STGZ, TBZ2, TGZ, TXZ, TZ, ZIP" )
    # set( CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON )
endif( )

# Package agnostic CPACK vars
set( CPACK_PACKAGE_CONTACT "Kent Knox <kent.knox@amd.com>" )
set( CPACK_PACKAGE_DESCRIPTION_SUMMARY "Radeon Open Compute BLAS library")
set( CPACK_PACKAGE_VERSION_MAJOR ${rocblas_VERSION_MAJOR} )
set( CPACK_PACKAGE_VERSION_MINOR ${rocblas_VERSION_MINOR} )
set( CPACK_PACKAGE_VERSION_PATCH ${rocblas_VERSION_PATCH} )
set( CPACK_PACKAGE_VERSION_TWEAK ${rocblas_VERSION_TWEAK} )
set( CPACK_PACKAGE_VERSION ${rocblas_VERSION} )
set( CPACK_PACKAGE_VENDOR "AMD")
set( CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE.md" )
set( CPACK_SOURCE_IGNORE_FILES "/\\\\.git/" )

# Package specific CPACK vars
set( CPACK_DEBIAN_PACKAGE_DEPENDS "hip_hcc (>= 1.0.17174)" )
set( CPACK_RPM_PACKAGE_REQUIRES "hip_hcc >= 1.0.17174" )

if( "${CPACK_PACKAGING_INSTALL_PREFIX}" MATCHES "^/opt/rocm.*$")
    include( package-functions )
    write_rocm_package_script_files( ${PROJECT_BINARY_DIR} "rocblas" "librocblas-hcc.so" )

    set( CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA "${PROJECT_BINARY_DIR}/postinst;${PROJECT_BINARY_DIR}/prerm")
    set( CPACK_RPM_POST_INSTALL_SCRIPT_FILE "${PROJECT_BINARY_DIR}/postinst" )
    set( CPACK_RPM_PRE_UNINSTALL_SCRIPT_FILE "${PROJECT_BINARY_DIR}/prerm" )
endif( )

# Define all variables that influence CPack before including CPack, such as install targets
include( CPack )
