# ########################################################################
# Copyright (C) 2016-2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
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

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Set up Tensile Dependency
if( BUILD_WITH_TENSILE )
  # If we want to build a shared rocblas lib, force Tensile to build as a static lib to absorb into rocblas
  if( BUILD_SHARED_LIBS )
    set( ROCBLAS_SHARED_LIBS ON )
    set( BUILD_SHARED_LIBS OFF )
  else( )
    set( ROCBLAS_SHARED_LIBS OFF )
  endif( )

  set( Tensile_RUNTIME_LANGUAGE "HIP" )

  #TODO update when this feature has been validated
  #set( PACKAGE_TENSILE_LIBRARY ON )
  set( PACKAGE_TENSILE_LIBRARY OFF )

  # Build options list
  if(Tensile_MERGE_FILES)
    set(Tensile_Options ${Tensile_Options} MERGE_FILES)
  endif()
  if(Tensile_SEPARATE_ARCHITECTURES)
    set(Tensile_Options ${Tensile_Options} SEPARATE_ARCHITECTURES)
  endif()
  if(Tensile_LAZY_LIBRARY_LOADING)
  set(Tensile_Options ${Tensile_Options} LAZY_LIBRARY_LOADING)
  endif()
  if(Tensile_SHORT_FILENAMES)
    set(Tensile_Options ${Tensile_Options} SHORT_FILE_NAMES)
  endif()
  if(Tensile_PRINT_DEBUG)
    set(Tensile_Options ${Tensile_Options} PRINT_DEBUG)
  endif()
  if(PACKAGE_TENSILE_LIBRARY)
    set(Tensile_Options ${Tensile_Options} GENERATE_PACKAGE)
  endif()

  # Add a build target for Tensile kernel library
  # Runtime language is HIP by default
  # warning our Tensile_ variables may shadow variable in TensileCreateLibraryFiles
  # thus bypassing the function argument parameter system (mainly the options list) and CPU_THREADS
  if(Tensile_CPU_THREADS MATCHES "^[0-9]+$")
    # only including threads argument if number
    TensileCreateLibraryFiles(
      "${CMAKE_CURRENT_SOURCE_DIR}/blas3/Tensile/Logic/${Tensile_LOGIC}"
      "${PROJECT_BINARY_DIR}/Tensile"
      ARCHITECTURE        ${Tensile_ARCHITECTURE}
      CODE_OBJECT_VERSION ${Tensile_CODE_OBJECT_VERSION}
      COMPILER            ${Tensile_COMPILER}
      LIBRARY_FORMAT      ${Tensile_LIBRARY_FORMAT}
      CPU_THREADS         ${Tensile_CPU_THREADS}
      ${Tensile_Options}
    )
  else()
    TensileCreateLibraryFiles(
      "${CMAKE_CURRENT_SOURCE_DIR}/blas3/Tensile/Logic/${Tensile_LOGIC}"
      "${PROJECT_BINARY_DIR}/Tensile"
      ARCHITECTURE        ${Tensile_ARCHITECTURE}
      CODE_OBJECT_VERSION ${Tensile_CODE_OBJECT_VERSION}
      COMPILER            ${Tensile_COMPILER}
      LIBRARY_FORMAT      ${Tensile_LIBRARY_FORMAT}
      ${Tensile_Options}
    )
  endif()

  # Create a unique name for TensileHost compiled for rocBLAS
  set_target_properties( TensileHost PROPERTIES OUTPUT_NAME rocblas-tensile CXX_EXTENSIONS NO )

  # Tensile host depends on libs build target
  add_dependencies( TensileHost TENSILE_LIBRARY_TARGET )

  if( ROCBLAS_SHARED_LIBS )
    set( BUILD_SHARED_LIBS ON )
    set_target_properties( TensileHost PROPERTIES POSITION_INDEPENDENT_CODE ON )
  endif()

  set( Tensile_SRC
    tensile_host.cpp
  )

  set( Tensile_INC
    ${CMAKE_CURRENT_SOURCE_DIR}/blas3/Tensile
  )

  #rocblas gemm_ex, gemm_ext2 and trsv require tensile
  set( rocblas_ex_source
    blas_ex/rocblas_gemm_ex.cpp
    blas_ex/rocblas_gemm_batched_ex.cpp
    blas_ex/rocblas_gemm_strided_batched_ex.cpp
    blas_ex/rocblas_gemm_ext2.cpp
    blas_ex/rocblas_trsv_ex.cpp
    blas_ex/rocblas_trsv_strided_batched_ex.cpp
    blas_ex/rocblas_trsv_batched_ex.cpp
  )

  #rocblas trsm and trtri require tensile
  set( rocblas_blas3_source
    blas3/rocblas_trsm.cpp
    blas3/rocblas_trsm_batched.cpp
    blas3/rocblas_trsm_strided_batched.cpp
    blas3/rocblas_trtri.cpp
    blas3/rocblas_trtri_batched.cpp
    blas3/rocblas_trtri_strided_batched.cpp
    ${Tensile_SRC}
  )

endif( ) # BUILD_WITH_TENSILE

set (rocblas_ex_source_no_tensile
    blas_ex/rocblas_axpy_ex.cpp
    blas_ex/rocblas_axpy_ex_kernels.cpp
    blas_ex/rocblas_axpy_batched_ex.cpp
    blas_ex/rocblas_axpy_strided_batched_ex.cpp
    blas_ex/rocblas_dot_ex.cpp
    blas_ex/rocblas_dot_batched_ex.cpp
    blas_ex/rocblas_dot_strided_batched_ex.cpp
    blas_ex/rocblas_rot_ex.cpp
    blas_ex/rocblas_rot_ex_kernels.cpp
    blas_ex/rocblas_rot_batched_ex.cpp
    blas_ex/rocblas_rot_strided_batched_ex.cpp
    blas_ex/rocblas_scal_ex.cpp
    blas_ex/rocblas_scal_ex_kernels.cpp
    blas_ex/rocblas_scal_batched_ex.cpp
    blas_ex/rocblas_scal_strided_batched_ex.cpp
    blas_ex/rocblas_nrm2_ex.cpp
    blas_ex/rocblas_nrm2_batched_ex.cpp
    blas_ex/rocblas_nrm2_strided_batched_ex.cpp
    blas_ex/rocblas_trmm_outofplace.cpp
    blas_ex/rocblas_trmm_outofplace_batched.cpp
    blas_ex/rocblas_trmm_outofplace_strided_batched.cpp
)

set( rocblas_blas3_source_no_tensile
    blas3/rocblas_dgmm.cpp
    blas3/rocblas_dgmm_kernels.cpp
    blas3/rocblas_dgmm_batched.cpp
    blas3/rocblas_dgmm_strided_batched.cpp
    blas3/rocblas_geam.cpp
    blas3/rocblas_geam_kernels.cpp
    blas3/rocblas_geam_batched.cpp
    blas3/rocblas_geam_strided_batched.cpp
    blas3/rocblas_hemm.cpp
    blas3/rocblas_hemm_batched.cpp
    blas3/rocblas_hemm_strided_batched.cpp
    blas3/rocblas_herk.cpp
    blas3/rocblas_herk_batched.cpp
    blas3/rocblas_herk_strided_batched.cpp
    blas3/rocblas_her2k.cpp
    blas3/rocblas_her2k_batched.cpp
    blas3/rocblas_her2k_strided_batched.cpp
    blas3/rocblas_herkx.cpp
    blas3/rocblas_herkx_batched.cpp
    blas3/rocblas_herkx_strided_batched.cpp
    blas3/rocblas_symm.cpp
    blas3/rocblas_symm_kernels.cpp
    blas3/rocblas_symm_batched.cpp
    blas3/rocblas_symm_strided_batched.cpp
    blas3/rocblas_syrk.cpp
    blas3/rocblas_syrk_herk_kernels.cpp
    blas3/rocblas_syrk_batched.cpp
    blas3/rocblas_syrk_strided_batched.cpp
    blas3/rocblas_syr2k.cpp
    blas3/rocblas_syr2k_her2k_kernels.cpp
    blas3/rocblas_syr2k_batched.cpp
    blas3/rocblas_syr2k_strided_batched.cpp
    blas3/Tensile/gemm.cpp
    blas3/Tensile/gemm_batched.cpp
    blas3/Tensile/gemm_strided_batched.cpp
    blas3/rocblas_syrkx.cpp
    blas3/rocblas_syrkx_herkx_kernels.cpp
    blas3/rocblas_syrkx_batched.cpp
    blas3/rocblas_syrkx_strided_batched.cpp
    blas3/rocblas_trmm.cpp
    blas3/rocblas_trmm_kernels.cpp
    blas3/rocblas_trmm_batched.cpp
    blas3/rocblas_trmm_strided_batched.cpp
)

set( rocblas_blas2_source
  blas2/rocblas_gemv.cpp
  blas2/rocblas_gemv_kernels.cpp
  blas2/rocblas_gemv_batched.cpp
  blas2/rocblas_gemv_strided_batched.cpp
  blas2/rocblas_tpmv.cpp
  blas2/rocblas_tpmv_kernels.cpp
  blas2/rocblas_tpmv_batched.cpp
  blas2/rocblas_tpmv_strided_batched.cpp
  blas2/rocblas_gbmv.cpp
  blas2/rocblas_gbmv_kernels.cpp
  blas2/rocblas_gbmv_batched.cpp
  blas2/rocblas_gbmv_strided_batched.cpp
  blas2/rocblas_tbsv.cpp
  blas2/rocblas_tbsv_kernels.cpp
  blas2/rocblas_tbsv_batched.cpp
  blas2/rocblas_tbsv_strided_batched.cpp
  blas2/rocblas_trmv.cpp
  blas2/rocblas_trmv_kernels.cpp
  blas2/rocblas_trmv_batched.cpp
  blas2/rocblas_trmv_strided_batched.cpp
  blas2/rocblas_ger.cpp
  blas2/rocblas_ger_kernels.cpp
  blas2/rocblas_ger_batched.cpp
  blas2/rocblas_ger_strided_batched.cpp
  blas2/rocblas_hbmv.cpp
  blas2/rocblas_hbmv_kernels.cpp
  blas2/rocblas_hbmv_batched.cpp
  blas2/rocblas_hbmv_strided_batched.cpp
  blas2/rocblas_hemv.cpp
  blas2/rocblas_hemv_symv_kernels.cpp
  blas2/rocblas_hemv_batched.cpp
  blas2/rocblas_hemv_strided_batched.cpp
  blas2/rocblas_her.cpp
  blas2/rocblas_her_kernels.cpp
  blas2/rocblas_her_batched.cpp
  blas2/rocblas_her_strided_batched.cpp
  blas2/rocblas_her2.cpp
  blas2/rocblas_her2_kernels.cpp
  blas2/rocblas_her2_batched.cpp
  blas2/rocblas_her2_strided_batched.cpp
  blas2/rocblas_hpmv.cpp
  blas2/rocblas_hpmv_kernels.cpp
  blas2/rocblas_hpmv_batched.cpp
  blas2/rocblas_hpmv_strided_batched.cpp
  blas2/rocblas_hpr.cpp
  blas2/rocblas_hpr_kernels.cpp
  blas2/rocblas_hpr_batched.cpp
  blas2/rocblas_hpr_strided_batched.cpp
  blas2/rocblas_hpr2.cpp
  blas2/rocblas_hpr2_kernels.cpp
  blas2/rocblas_hpr2_batched.cpp
  blas2/rocblas_hpr2_strided_batched.cpp
  blas2/rocblas_spr.cpp
  blas2/rocblas_spr_kernels.cpp
  blas2/rocblas_spr_batched.cpp
  blas2/rocblas_spr_strided_batched.cpp
  blas2/rocblas_spr2.cpp
  blas2/rocblas_spr2_kernels.cpp
  blas2/rocblas_spr2_batched.cpp
  blas2/rocblas_spr2_strided_batched.cpp
  blas2/rocblas_syr.cpp
  blas2/rocblas_syr_kernels.cpp
  blas2/rocblas_syr_batched.cpp
  blas2/rocblas_syr_strided_batched.cpp
  blas2/rocblas_syr2.cpp
  blas2/rocblas_syr2_kernels.cpp
  blas2/rocblas_syr2_batched.cpp
  blas2/rocblas_syr2_strided_batched.cpp
  blas2/rocblas_tbmv.cpp
  blas2/rocblas_tbmv_kernels.cpp
  blas2/rocblas_tbmv_batched.cpp
  blas2/rocblas_tbmv_strided_batched.cpp
  blas2/rocblas_tpsv.cpp
  blas2/rocblas_tpsv_kernels.cpp
  blas2/rocblas_tpsv_batched.cpp
  blas2/rocblas_tpsv_strided_batched.cpp
  blas2/rocblas_sbmv.cpp
  blas2/rocblas_sbmv_kernels.cpp
  blas2/rocblas_sbmv_batched.cpp
  blas2/rocblas_sbmv_strided_batched.cpp
  blas2/rocblas_spmv.cpp
  blas2/rocblas_spmv_kernels.cpp
  blas2/rocblas_spmv_batched.cpp
  blas2/rocblas_spmv_strided_batched.cpp
  blas2/rocblas_symv.cpp
  blas2/rocblas_symv_batched.cpp
  blas2/rocblas_symv_strided_batched.cpp
  blas2/rocblas_trsv.cpp
  blas2/rocblas_trsv_kernels.cpp
  blas2/rocblas_trsv_strided_batched.cpp
  blas2/rocblas_trsv_batched.cpp
)

set( rocblas_auxiliary_source
  handle.cpp
  rocblas_auxiliary.cpp
  buildinfo.cpp
  rocblas_ostream.cpp
  check_numerics_vector.cpp
  check_numerics_matrix.cpp
)

set( rocblas_blas1_source
  blas1/rocblas_iamin.cpp
  blas1/rocblas_iamin_batched.cpp
  blas1/rocblas_iamin_strided_batched.cpp
  blas1/rocblas_iamax.cpp
  blas1/rocblas_iamax_batched.cpp
  blas1/rocblas_iamax_strided_batched.cpp
  blas1/rocblas_asum.cpp
  blas1/rocblas_asum_batched.cpp
  blas1/rocblas_asum_strided_batched.cpp
  blas1/rocblas_axpy.cpp
  blas1/rocblas_axpy_kernels.cpp
  blas1/rocblas_axpy_batched.cpp
  blas1/rocblas_axpy_strided_batched.cpp
  blas1/rocblas_copy.cpp
  blas1/rocblas_copy_kernels.cpp
  blas1/rocblas_copy_batched.cpp
  blas1/rocblas_copy_strided_batched.cpp
  blas1/rocblas_dot.cpp
  blas1/rocblas_dot_kernels.cpp
  blas1/rocblas_dot_strided_batched.cpp
  blas1/rocblas_dot_batched.cpp
  blas1/rocblas_nrm2.cpp
  blas1/rocblas_nrm2_batched.cpp
  blas1/rocblas_nrm2_strided_batched.cpp
  blas1/rocblas_rot.cpp
  blas1/rocblas_rot_kernels.cpp
  blas1/rocblas_rot_batched.cpp
  blas1/rocblas_rot_strided_batched.cpp
  blas1/rocblas_rotg.cpp
  blas1/rocblas_rotg_kernels.cpp
  blas1/rocblas_rotg_batched.cpp
  blas1/rocblas_rotg_strided_batched.cpp
  blas1/rocblas_rotm.cpp
  blas1/rocblas_rotm_kernels.cpp
  blas1/rocblas_rotm_batched.cpp
  blas1/rocblas_rotm_strided_batched.cpp
  blas1/rocblas_rotmg.cpp
  blas1/rocblas_rotmg_kernels.cpp
  blas1/rocblas_rotmg_batched.cpp
  blas1/rocblas_rotmg_strided_batched.cpp
  blas1/rocblas_scal.cpp
  blas1/rocblas_scal_kernels.cpp
  blas1/rocblas_scal_batched.cpp
  blas1/rocblas_scal_strided_batched.cpp
  blas1/rocblas_swap.cpp
  blas1/rocblas_swap_kernels.cpp
  blas1/rocblas_swap_batched.cpp
  blas1/rocblas_swap_strided_batched.cpp
)

prepend_path( ".." rocblas_headers_public relative_rocblas_headers_public )

add_library( rocblas
  ${rocblas_ex_source}
  ${rocblas_ex_source_no_tensile}
  ${rocblas_blas3_source}
  ${rocblas_blas3_source_no_tensile}
  ${rocblas_blas2_source}
  ${rocblas_blas1_source}
  ${relative_rocblas_headers_public}
  ${rocblas_auxiliary_source}
)

message(STATUS "*** NOTE: blas2/rocblas_ger_kernels.cpp is compiled with the verbose flag -v for QC purposes.")
SET_SOURCE_FILES_PROPERTIES( blas2/rocblas_ger_kernels.cpp PROPERTIES COMPILE_FLAGS "-v" )

#if( WIN32 )
#  set_target_properties(rocblas_fortran PROPERTIES LINKER_LANGUAGE CXX)
#  target_link_directories(rocblas_fortran PRIVATE "C:\\cygwin64\\lib\\gcc\\x86_64-pc-cygwin\\9.3.0" "C:\\cygwin64\\lib" "C:\\cygwin64\\lib\\w32api")
#endif( )

add_library( roc::rocblas ALIAS rocblas )

target_link_libraries( rocblas INTERFACE hip::host )
if (WIN32)
  target_link_libraries( rocblas PRIVATE hip::device )
else()
  target_link_libraries( rocblas PRIVATE hip::device -lstdc++fs --rtlib=compiler-rt --unwindlib=libgcc )
endif()

#  -fno-gpu-rdc compiler option was used with hcc, so revisit feature at some point

# GCC or hip-clang needs specific flags to turn on f16c intrinsics
target_compile_options( rocblas PRIVATE -mf16c )

# Do not allow Variable Length Arrays (use unique_ptr instead)
target_compile_options( rocblas PRIVATE -Werror=vla )

target_include_directories( rocblas
  PUBLIC  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/library/include>
          $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/library/include/internal>
          $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include/rocblas/internal>
	  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include/rocblas>
	  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
          $<BUILD_INTERFACE:${Tensile_INC}>
	  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
          )

target_link_libraries( rocblas PRIVATE "-Xlinker --exclude-libs=ALL" ) # HIDE symbols

if( BUILD_WITH_TENSILE )

  if( BUILD_SHARED_LIBS )
    target_link_libraries( rocblas PRIVATE TensileHost )
  else()
    target_compile_definitions( rocblas PRIVATE ROCBLAS_STATIC_LIB )

    # bypassing cmake dependencies chain for static link as it won't allow target from different directory

    # including tensile headers into rocblas tensileHost client so get compile properties
    get_target_property(TensileHost_INCLUDES TensileHost INCLUDE_DIRECTORIES)
    target_include_directories( rocblas PRIVATE ${TensileHost_INCLUDES} )
    get_target_property(TensileHost_DEFINES TensileHost COMPILE_DEFINITIONS)
    target_compile_definitions( rocblas PRIVATE ${TensileHost_DEFINES} )

    get_target_property( TensileHost_LIBDIR TensileHost BINARY_DIR )

    message (STATUS "TensileHost_INCLUDES == ${TensileHost_INCLUDES}")
    message (STATUS "TensileHost_DEFINES == ${TensileHost_DEFINES}")
    message (STATUS "TensileHost_LIBDIR == ${TensileHost_LIBDIR}")

    # recreate LLVM static dependencies
    if (${Tensile_LIBRARY_FORMAT} STREQUAL "yaml")
      find_package(LLVM REQUIRED CONFIG)
      find_library(LLVMObjectYAML_LIBRARY
        NAMES LLVMObjectYAML
        PATHS ${LLVM_LIBRARY_DIR})
      message("LLVMObjectYAML_LIBRARY: ${LLVMObjectYAML_LIBRARY}")

      target_link_libraries(rocblas PRIVATE LLVMObjectYAML )  # match tensile
    endif()

    # to get TensileHost built first, not to link target
    # as dependency chain can not be created
    add_dependencies(rocblas TensileHost)

  endif()

  target_compile_definitions( rocblas PRIVATE ${TENSILE_DEFINES} )
endif()

target_compile_definitions( rocblas PRIVATE ROCM_USE_FLOAT16 ROCBLAS_INTERNAL_API )

rocm_set_soversion( rocblas ${rocblas_SOVERSION} )
set_target_properties( rocblas PROPERTIES CXX_EXTENSIONS NO )
set_target_properties( rocblas PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )

if (WIN32 AND BUILD_CLIENTS)
  add_custom_command( TARGET rocblas POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_BINARY_DIR}/staging/$<TARGET_FILE_NAME:rocblas> ${PROJECT_BINARY_DIR}/clients/staging/$<TARGET_FILE_NAME:rocblas> )
  if( ${CMAKE_BUILD_TYPE} MATCHES "Debug")
    add_custom_command( TARGET rocblas POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_BINARY_DIR}/staging/rocblas.pdb ${PROJECT_BINARY_DIR}/clients/staging/rocblas.pdb )
  endif()
endif()

# Package that helps me set visibility for function names exported from shared library
include( GenerateExportHeader )
set_target_properties( rocblas PROPERTIES CXX_VISIBILITY_PRESET "hidden" C_VISIBILITY_PRESET "hidden" VISIBILITY_INLINES_HIDDEN ON )
generate_export_header( rocblas EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/rocblas/internal/rocblas-export.h )

# generate header with prototypes for export reuse
file( GLOB rocblas_prototype_inputs
  LIST_DIRECTORIES OFF
  CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/blas3/Tensile/*.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/blas3/*.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/blas2/*.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/blas1/*.hpp
)
set( ROCBLAS_PROTO_TEMPLATES "${PROJECT_BINARY_DIR}/include/rocblas/internal/rocblas-exported-proto.hpp" )
add_custom_command(OUTPUT ${ROCBLAS_PROTO_TEMPLATES}
  COMMAND ${python} template-proto.py ${CMAKE_CURRENT_SOURCE_DIR}/blas3/Tensile/ ${CMAKE_CURRENT_SOURCE_DIR}/blas3/ ${CMAKE_CURRENT_SOURCE_DIR}/blas2/ ${CMAKE_CURRENT_SOURCE_DIR}/blas1/ > ${ROCBLAS_PROTO_TEMPLATES}
  #DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/blas3/Tensile/ ${CMAKE_CURRENT_SOURCE_DIR}/blas3/ ${CMAKE_CURRENT_SOURCE_DIR}/blas2/ ${CMAKE_CURRENT_SOURCE_DIR}/blas1/
  DEPENDS ${rocblas_prototype_inputs}
  COMMENT "Generating prototypes from ${CMAKE_CURRENT_SOURCE_DIR}."
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
add_custom_target( rocblas_proto_templates DEPENDS ${ROCBLAS_PROTO_TEMPLATES} )
add_dependencies( rocblas rocblas_proto_templates )

# generate rocblas_device_malloc.hpp header for device memory allocation
set( ROCBLAS_DEVICE_MALLOC "${PROJECT_BINARY_DIR}/include/rocblas/internal/rocblas_device_malloc.hpp" )
add_custom_command( OUTPUT ${ROCBLAS_DEVICE_MALLOC}
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/include/rocblas_device_malloc.hpp ${ROCBLAS_DEVICE_MALLOC}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/include/rocblas_device_malloc.hpp
  COMMENT "Generating ${ROCBLAS_DEVICE_MALLOC}"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
add_custom_target( rocblas_device_malloc DEPENDS ${ROCBLAS_DEVICE_MALLOC} )
add_dependencies( rocblas rocblas_device_malloc )

if(BUILD_FILE_REORG_BACKWARD_COMPATIBILITY AND NOT WIN32)
  rocm_wrap_header_dir(
    ${CMAKE_SOURCE_DIR}/library/include
    PATTERNS "*.h"
    GUARDS SYMLINK WRAPPER
    WRAPPER_LOCATIONS ${CMAKE_INSTALL_INCLUDEDIR}
  )
  rocm_wrap_header_file(
    internal/rocblas-version.h internal/rocblas-export.h internal/rocblas-exported-proto.hpp internal/rocblas_device_malloc.hpp
    GUARDS SYMLINK WRAPPER
    WRAPPER_LOCATIONS ${CMAKE_INSTALL_INCLUDEDIR} rocblas/${CMAKE_INSTALL_INCLUDEDIR}
    ORIGINAL_FILES ${PROJECT_BINARY_DIR}/include/rocblas/internal/rocblas-version.h
  )
endif( )


if( NOT BUILD_SHARED_LIBS )
  # Following Boost conventions of prefixing 'lib' on static built libraries, across all platforms
  set_target_properties( rocblas PROPERTIES PREFIX "lib" )

  if( BUILD_WITH_TENSILE )
    add_custom_command( TARGET rocblas POST_BUILD
      COMMAND
        ${python} ${CMAKE_CURRENT_SOURCE_DIR}/merge_archives.py
      ARGS
        -v
        -o "$<TARGET_LINKER_FILE:rocblas>"
        --ar "${CMAKE_AR}"
        -L "${TensileHost_LIBDIR}"
        "$<TARGET_LINKER_FILE:rocblas>"
        "librocblas-tensile.a"
      WORKING_DIRECTORY "${PROJECT_BINARY_DIR}"
      COMMENT "Merging rocblas-tensile library into rocblas"
      )
  endif()

endif( )

############################################################
# Installation

# Force installation of .f90 module file
rocm_install(FILES "../include/rocblas_module.f90"
	DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/rocblas/"
)

execute_process(COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SOURCE_DIR}/library/include ${PROJECT_BINARY_DIR}/include/rocblas)
rocm_install_targets(
  TARGETS ${package_targets}
  INCLUDE
    ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_INCLUDEDIR}
)

if( BUILD_WITH_TENSILE )
  if (WIN32)
    set( ROCBLAS_TENSILE_LIBRARY_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/bin/rocblas" CACHE PATH "path to tensile library" )
  else()
    set( ROCBLAS_TENSILE_LIBRARY_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}${CMAKE_INSTALL_LIBDIR}/rocblas" CACHE PATH "path to tensile library" )
  endif()
  rocm_install(
    DIRECTORY ${CMAKE_BINARY_DIR}/Tensile/library
    DESTINATION ${ROCBLAS_TENSILE_LIBRARY_DIR}
    COMPONENT ${CMAKE_INSTALL_DEFAULT_COMPONENT_NAME}) # Use this cmake variable to be compatible with rocm-cmake 0.6 and 0.7
endif()

#         PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ

if( BUILD_SHARED_LIBS )
  rocm_export_targets(
    TARGETS roc::rocblas
    DEPENDS PACKAGE hip
    NAMESPACE roc::
    )
else()
  # static
  if (DEFINED Tensile_LIBRARY_FORMAT AND ${Tensile_LIBRARY_FORMAT} STREQUAL "msgpack")
    # header only tensile usage of msgpack
    rocm_export_targets(
      TARGETS roc::rocblas
      DEPENDS PACKAGE hip
      NAMESPACE roc::
      )
  else()
    # yaml based Tensile uses LLVM
    rocm_export_targets(
      TARGETS roc::rocblas
      DEPENDS PACKAGE hip
      STATIC_DEPENDS PACKAGE LLVM
      NAMESPACE roc::
      )
  endif()
endif()

if(BUILD_FILE_REORG_BACKWARD_COMPATIBILITY AND NOT WIN32)
  rocm_install(
    DIRECTORY
       "${PROJECT_BINARY_DIR}/rocblas"
        DESTINATION "." )

  if ( NOT WIN32 )

    #Create SymLink for Fortran Object Module for backward compatibility
    rocm_install(
      CODE "
        set(PREFIX \$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX})
        set(INPUT_FILE \${PREFIX}/include/rocblas/rocblas_module.f90)
        set(SYMLINK_LOCATIONS \${PREFIX}/rocblas/include \${PREFIX}/include)
        foreach(LOCATION IN LISTS SYMLINK_LOCATIONS)
          file(MAKE_DIRECTORY \${LOCATION})
          execute_process(COMMAND ln -sfr \${INPUT_FILE} \${LOCATION})
          message(STATUS \"Created symlink in \${LOCATION} to \${INPUT_FILE}.\")
        endforeach()
        "
    )
  endif() #NOT WIN32
endif()

if(RUN_HEADER_TESTING)
# Compilation tests to ensure that header files work independently,
# and that public header files work across several languages
add_custom_command(
  TARGET rocblas
  POST_BUILD
  COMMAND ${CMAKE_HOME_DIRECTORY}/header_compilation_tests.sh
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
 )
endif()
