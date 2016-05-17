# ########################################################################
# Copyright 2016 Advanced Micro Devices, Inc.
# ########################################################################

# A helper function to prefix a source list of files with a common path into a new list (non-destructive)
function( prepend_path prefix source_list_of_files return_list_of_files )
  foreach( file ${${source_list_of_files}} )
    if(IS_ABSOLUTE ${file} )
      list( APPEND new_list ${file} )
    else( )
      list( APPEND new_list ${prefix}/${file} )
    endif( )
  endforeach( )
  set( ${return_list_of_files} ${new_list} PARENT_SCOPE )
endfunction( )

include( GenerateExportHeader )
add_compiler_export_flags( )

set( rocblas_auxiliary_source
  handle.h
  handle.cpp
  rocblas_auxiliary.cpp
  status.h
  status.cpp
)

set( rocblas_blas3_source
  blas3/gemm.cpp
)

prepend_path( ".." rocblas_headers_public relative_rocblas_headers_public )

add_library( rocblas ${BUILD_LIBRARY_TYPE}
  ${rocblas_blas3_source}
  ${relative_rocblas_headers_public}
  ${rocblas_auxiliary_source}
)

# Try to test for specific compiler features if cmake version is recent enough
if( CMAKE_VERSION VERSION_GREATER "3.0" )
  target_compile_features( rocblas PRIVATE cxx_static_assert cxx_nullptr cxx_lambdas cxx_auto_type )
else( )
  # Otherwise, just try to compile the library with a standards flag
  if( CMAKE_COMPILER_IS_GNUCXX OR ( CMAKE_CXX_COMPILER_ID MATCHES "Clang" ) )
    target_compile_options( rocblas PRIVATE -std=c++11 )
  endif( )
endif( )
set_target_properties( rocblas PROPERTIES COMPILE_FLAGS "-fPIC" )

target_include_directories( rocblas
  PUBLIC  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
          $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
          $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
          $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library>
          $<INSTALL_INTERFACE:include>
)

# Cobalt
MESSAGE( STATUS "Cobalt_INCLUDE_DIRS=${Cobalt_INCLUDE_DIRS}" )
include_directories( ${Cobalt_INCLUDE_DIRS} )
target_link_libraries( rocblas ${CobaltLib_LIBRARIES} )
add_definitions( -DCobalt_BACKEND_HIP=1 )

set_target_properties( rocblas PROPERTIES VERSION ${rocblas_VERSION} SOVERSION ${rocblas_SOVERSION} )
set_target_properties( rocblas PROPERTIES DEBUG_POSTFIX "-d" )
set_target_properties( rocblas PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )

# Following Boost conventions of prefixing 'lib' on static built libraries, across all platforms
if( ${BUILD_LIBRARY_TYPE} STREQUAL "STATIC" )
  set_target_properties( rocblas PROPERTIES PREFIX "lib" )
endif( )

# Package that helps me set visibility for function names exported from shared library
GENERATE_EXPORT_HEADER( rocblas EXPORT_FILE_NAME rocblas-export.h )

# The following is cmake code to generate a config file package for rocblas
# Documentation for how to use the following package:
# https://cmake.org/cmake/help/v3.0/module/CMakePackageConfigHelpers.html
include( CMakePackageConfigHelpers )

set( LIB_INSTALL_DIR lib${SUFFIX_LIB} )
set( INCLUDE_INSTALL_DIR include )
set( config_package_location ${LIB_INSTALL_DIR}/cmake/rocblas )

configure_package_config_file(
  rocblas-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/rocblas-config.cmake
  INSTALL_DESTINATION ${config_package_location}
  PATH_VARS INCLUDE_INSTALL_DIR LIB_INSTALL_DIR
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/rocblas-config-version.cmake
  VERSION ${rocblas_VERSION}
  COMPATIBILITY SameMajorVersion
)

install( TARGETS rocblas
  EXPORT rocblas-targets
  RUNTIME DESTINATION bin${SUFFIX_BIN}
  LIBRARY DESTINATION lib${SUFFIX_LIB}
  ARCHIVE DESTINATION lib${SUFFIX_LIB}
)

install( EXPORT rocblas-targets
  DESTINATION
    ${config_package_location}
)

install( FILES
  ${CMAKE_CURRENT_BINARY_DIR}/rocblas-config.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/rocblas-config-version.cmake
  DESTINATION
    ${config_package_location} )

install( FILES
  ${CMAKE_CURRENT_BINARY_DIR}/rocblas-export.h
  DESTINATION
    include
)
