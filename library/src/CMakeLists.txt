# ########################################################################
# Copyright 2016 Advanced Micro Devices, Inc.
# ########################################################################

# ########################################################################
# A helper function to prefix a source list of files with a common path into a new list (non-destructive)
# ########################################################################
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

# ########################################################################
# Main
# ########################################################################

# Set up Tensile  Dependency 
if( BUILD_WITH_TENSILE )
  find_package(Tensile REQUIRED CONFIG)
  add_tensile_lib( rocblas-tensile
      SOLUTIONS ${CMAKE_CURRENT_SOURCE_DIR}/blas3/Tensile/SolutionXMLs/${Tensile_SCHEDULE}
      BACKEND HIP )

  #rocblas_gemm and rocblas_trsm require tensile 
    set( Tensile_SRC 
      blas3/Tensile/gemm.cpp
      blas3/Tensile/Tensile_status.cpp
      blas3/rocblas_gemm.cpp
#  blas3/rocblas_trmm.cpp
#  blas3/rocblas_trsm.cpp
#  blas3/rocblas_trtri_trsm.cpp
      )

    set(Tensile_INC
        ${CMAKE_CURRENT_SOURCE_DIR}/blas3/Tensile
        )
endif()



set( rocblas_auxiliary_source
  include/handle.h
  include/definitions.h
  include/status.h
  handle.cpp
  rocblas_auxiliary.cpp
  status.cpp
)

set( rocblas_blas3_source
  blas3/rocblas_trtri_batched.cpp
  blas3/rocblas_trtri.cpp
  ${Tensile_SRC}
)

set( rocblas_blas2_source
  blas2/rocblas_gemv.cpp
  blas2/rocblas_ger.cpp
)

set( rocblas_blas1_source
  blas1/fetch_template.cpp
  blas1/rocblas_amax.cpp
  blas1/rocblas_asum.cpp
  blas1/rocblas_axpy.cpp
  blas1/rocblas_copy.cpp
  blas1/rocblas_dot.cpp
  blas1/rocblas_nrm2.cpp
  blas1/rocblas_scal.cpp
  blas1/rocblas_swap.cpp
)

prepend_path( ".." rocblas_headers_public relative_rocblas_headers_public )

if( ${BUILD_SHARED_LIBS} )
  message( STATUS "Library Type: SHARED" )
else( )
  message( STATUS "Library Type: STATIC" )
endif( )


add_library( rocblas
  ${rocblas_blas3_source}
  ${rocblas_blas2_source}
  ${rocblas_blas1_source}
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

target_include_directories( rocblas
  PUBLIC  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
          $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
          $<BUILD_INTERFACE:${Tensile_INC}>
          $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
          $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
          $<INSTALL_INTERFACE:include>
)

if( BUILD_WITH_TENSILE )
    target_link_libraries( rocblas PRIVATE rocblas-tensile )
    target_compile_definitions( rocblas PRIVATE BUILD_WITH_TENSILE=1 )
endif()

set_target_properties( rocblas PROPERTIES VERSION ${rocblas_VERSION} SOVERSION ${rocblas_SOVERSION} )
set_target_properties( rocblas PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )
set_target_properties( rocblas PROPERTIES DEBUG_POSTFIX "-d" OUTPUT_NAME rocblas-hcc )

# Package that helps me set visibility for function names exported from shared library
include( GenerateExportHeader )
if( CMAKE_VERSION VERSION_LESS "3.0" )
  add_compiler_export_flags( )
else()
  set_target_properties( rocblas PROPERTIES CXX_VISIBILITY_PRESET "hidden" VISIBILITY_INLINES_HIDDEN ON )
endif( )
GENERATE_EXPORT_HEADER( rocblas EXPORT_FILE_NAME rocblas-export.h )

# Following Boost conventions of prefixing 'lib' on static built libraries, across all platforms
if( NOT ${BUILD_SHARED_LIBS} )
  set_target_properties( rocblas PROPERTIES PREFIX "lib" )
endif( )

# The following is cmake code to generate a config file package for rocblas
# Documentation for how to use the following package:
# https://cmake.org/cmake/help/v3.0/module/CMakePackageConfigHelpers.html
include( CMakePackageConfigHelpers )

set( LIB_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR} )
set( INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR} )
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

if( BUILD_WITH_TENSILE )
    install( TARGETS rocblas rocblas-tensile
        EXPORT rocblas-targets
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
else()
    install( TARGETS rocblas 
        EXPORT rocblas-targets
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
endif()

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
    ${CMAKE_INSTALL_INCLUDEDIR}
)
