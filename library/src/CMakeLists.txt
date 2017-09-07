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
# package_targets is used as a list of install target
set( package_targets rocblas )

# Set up Tensile  Dependency
if( BUILD_WITH_TENSILE )
  # If we want to build a shared rocblas lib, force Tensile to build as a static lib to absorb into rocblas
  if( BUILD_SHARED_LIBS )
    set( ROCBLAS_SHARED_LIBS ON )
    set( BUILD_SHARED_LIBS OFF )
  else( )
    list( APPEND package_targets Tensile )
  endif( )

  include( ${Tensile_ROOT}/Tensile/Source/TensileConfig.cmake )

  set( Tensile_RUNTIME_LANGUAGE "HIP" )
  set( Tensile_KERNEL_LANGUAGE "HIP" )
  TensileCreateLibrary(
      ${CMAKE_CURRENT_SOURCE_DIR}/blas3/Tensile/Logic/${Tensile_LOGIC}
      ${Tensile_RUNTIME_LANGUAGE}
      ${Tensile_KERNEL_LANGUAGE}
      ${Tensile_MERGE_FILES}
      ${Tensile_SHORT_FILENAMES}
      ${Tensile_PRINT_DEBUG}
      Tensile_ROOT ${Tensile_ROOT}
  )

  # Create a unique name for Tensile compiled for rocBLAS
  set_target_properties( Tensile PROPERTIES OUTPUT_NAME tensile-rocblas CXX_EXTENSIONS NO )
  target_compile_features( Tensile PRIVATE cxx_static_assert cxx_nullptr cxx_auto_type )
  target_link_libraries( Tensile PRIVATE hip::hip_hcc hip::hip_device hcc::hccshared )

  if( CMAKE_CXX_COMPILER MATCHES ".*/hcc$" )
    # Remove following when hcc is fixed; hcc emits following spurious warning ROCm v1.6.1
    # "clang-5.0: warning: argument unused during compilation: '-isystem /opt/rocm/include'"
    target_compile_options( Tensile PRIVATE -Wno-unused-command-line-argument )

    # foreach( target ${AMDGPU_TARGETS} )
    #   target_link_libraries( Tensile PRIVATE --amdgpu-target=${target} )
    # endforeach( )
  endif( )

  if( ROCBLAS_SHARED_LIBS )
    set( BUILD_SHARED_LIBS ON )
    set_target_properties( Tensile PROPERTIES POSITION_INDEPENDENT_CODE ON )
  endif()

  #rocblas_gemm and rocblas_trsm require tensile
  set( Tensile_SRC
    blas3/Tensile/gemm.cpp
    blas3/rocblas_trsm.cpp
  )

  set( Tensile_INC
    ${CMAKE_CURRENT_SOURCE_DIR}/blas3/Tensile
  )
endif( )

set( rocblas_auxiliary_source
  include/handle.h
  include/definitions.h
  include/status.h
  include/rocblas_unique_ptr.hpp
  handle.cpp
  rocblas_auxiliary.cpp
  status.cpp
)

set( rocblas_blas3_source
  blas3/rocblas_trtri.cpp
  blas3/rocblas_trtri_batched.cpp
  ${Tensile_SRC}
)

set( rocblas_blas2_source
  blas2/rocblas_gemv.cpp
  blas2/rocblas_ger.cpp
)

set( rocblas_blas1_source
  blas1/fetch_template.cpp
  blas1/rocblas_amin.cpp
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

add_library( rocblas
  ${rocblas_blas3_source}
  ${rocblas_blas2_source}
  ${rocblas_blas1_source}
  ${relative_rocblas_headers_public}
  ${rocblas_auxiliary_source}
)

# NOTE: 2 issues below
# 1.  rocblas header 'rocblas-types.h' exposes the hip header hip/hip_vector_types.h, requiring exposing hip::hip_hcc as PUBLIC
# 2.  hcc requires -Wl,-Bsymbolic for rocm v1.6.1, should be able to remove in future
target_link_libraries( rocblas PUBLIC hip::hip_hcc PRIVATE hip::hip_device hcc::hccshared )

# Test for specific compiler features if cmake version is recent enough
if( CMAKE_VERSION VERSION_GREATER "3.0" )
  target_compile_features( rocblas PRIVATE cxx_static_assert cxx_nullptr cxx_auto_type )
else( )
  # Otherwise, just try to compile the library with a standards flag
  if( CMAKE_COMPILER_IS_GNUCXX OR ( CMAKE_CXX_COMPILER_ID MATCHES "Clang" ) )
    target_compile_options( rocblas PRIVATE -std=c++11 )
  endif( )
endif( )

if( CMAKE_CXX_COMPILER MATCHES ".*/hcc$" )
  # Remove following when hcc is fixed; hcc emits following spurious warning ROCm v1.6.1
  # "clang-5.0: warning: argument unused during compilation: '-isystem /opt/rocm/include'"
  target_compile_options( rocblas PRIVATE -Wno-unused-command-line-argument )

  foreach( target ${AMDGPU_TARGETS} )
    target_link_libraries( rocblas PRIVATE --amdgpu-target=${target} )
  endforeach( )
endif( )

target_include_directories( rocblas
  PUBLIC  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/library/include>
          $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
          $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
          $<BUILD_INTERFACE:${Tensile_INC}>
          $<INSTALL_INTERFACE:include>
)

if( BUILD_WITH_TENSILE )
  target_link_libraries( rocblas PRIVATE Tensile )
  target_compile_definitions( rocblas PRIVATE BUILD_WITH_TENSILE=1 )
endif()

set_target_properties( rocblas PROPERTIES VERSION ${rocblas_VERSION} SOVERSION ${rocblas_SOVERSION} CXX_EXTENSIONS NO )
set_target_properties( rocblas PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )
set_target_properties( rocblas PROPERTIES DEBUG_POSTFIX "-d" )

# Package that helps me set visibility for function names exported from shared library
include( GenerateExportHeader )
set_target_properties( rocblas PROPERTIES CXX_VISIBILITY_PRESET "hidden" VISIBILITY_INLINES_HIDDEN ON )
generate_export_header( rocblas EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/rocblas-export.h )

# Following Boost conventions of prefixing 'lib' on static built libraries, across all platforms
if( NOT BUILD_SHARED_LIBS )
  set_target_properties( rocblas PROPERTIES PREFIX "lib" )
endif( )

############################################################
# Installation

rocm_install_targets(
  TARGETS ${package_targets}
  INCLUDE
    ${CMAKE_SOURCE_DIR}/library/include
    ${CMAKE_BINARY_DIR}/include
  PREFIX rocblas
)
#         PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ

rocm_export_targets(
  TARGETS rocblas-targets
  PREFIX rocblas
  DEPENDS PACKAGE hip
  NAMESPACE roc::
 )

rocm_install_symlink_subdir( rocblas )