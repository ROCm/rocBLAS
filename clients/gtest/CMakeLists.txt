# ########################################################################
# Copyright 2016 Advanced Micro Devices, Inc.
# ########################################################################

# set( Boost_DEBUG ON )
set( Boost_USE_MULTITHREADED ON )
set( Boost_DETAILED_FAILURE_MSG ON )
set( Boost_ADDITIONAL_VERSIONS 1.64.0 1.64 )
set( Boost_USE_STATIC_LIBS OFF )
find_package( Boost COMPONENTS program_options )

if( NOT Boost_FOUND )
  message( STATUS "Dynamic boost libraries not found. Attempting to find static libraries " )
  set( Boost_USE_STATIC_LIBS ON )
  find_package( Boost COMPONENTS program_options )

  if( NOT Boost_FOUND )
    message( FATAL_ERROR "boost is a required dependency and is not found;  try adding boost path to CMAKE_PREFIX_PATH" )
  endif( )
endif( )

# Linking lapack library requires fortran flags
enable_language( Fortran )
find_package( cblas CONFIG REQUIRED )
if( NOT cblas_FOUND )
  message( FATAL_ERROR "cblas is a required dependency and is not found;  try adding cblas path to CMAKE_PREFIX_PATH" )
endif( )

if( NOT TARGET rocblas )
  find_package( rocblas CONFIG PATHS /opt/rocm/rocblas )

  if( NOT rocblas_FOUND )
    message( FATAL_ERROR "rocBLAS is a required dependency and is not found; try adding rocblas path to CMAKE_PREFIX_PATH")
  endif( )
endif( )

find_package( GTest REQUIRED )

if( BUILD_WITH_TENSILE )
  set(Tensile_TEST_SRC
      gemm_gtest.cpp
      gemm_strided_batched_gtest.cpp
      trsm_gtest.cpp
      )
endif( )

set(rocblas_test_source
    rocblas_gtest_main.cpp
    set_get_pointer_mode_gtest.cpp
    set_get_vector_gtest.cpp
    set_get_matrix_gtest.cpp
    blas1_gtest.cpp
    gemv_gtest.cpp
    ger_gtest.cpp
    ${Tensile_TEST_SRC}
    )


set( rocblas_benchmark_common
      ../common/utility.cpp
      ../common/cblas_interface.cpp
      ../common/flops.cpp
      ../common/norm.cpp
      ../common/unit.cpp
      ../common/near.cpp
      ../common/arg_check.cpp
      ../common/rocblas_template_specialization.cpp
    )

add_executable( rocblas-test ${rocblas_test_source} ${rocblas_benchmark_common} )

if( BUILD_WITH_TENSILE )
    target_compile_definitions( rocblas-test PRIVATE BUILD_WITH_TENSILE=1 )
else()
    target_compile_definitions( rocblas-test PRIVATE BUILD_WITH_TENSILE=0 )
endif()

# Try to test for specific compiler features if cmake version is recent enough
if( CMAKE_VERSION VERSION_GREATER "3.0" )
    target_compile_features( rocblas-test PRIVATE cxx_static_assert cxx_nullptr cxx_auto_type)
else( )
  # Otherwise, just try to compile the library with a standards flag
  if( CMAKE_COMPILER_IS_GNUCXX OR ( CMAKE_CXX_COMPILER_ID MATCHES "Clang" ) )
    target_compile_options( rocblas-test PRIVATE -std=c++11 -pthread )
  endif( )
endif( )

#  GTEST_USE_OWN_TR1_TUPLE=1
target_compile_definitions( rocblas-test PRIVATE GOOGLE_TEST )
target_include_directories( rocblas-test
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${GTEST_INCLUDE_DIRS}>
)

if( TARGET rocblas )
  target_link_libraries( rocblas-test PRIVATE rocblas )
else( )
  target_link_libraries( rocblas-test PRIVATE roc::rocblas )
endif( )

target_link_libraries( rocblas-test PRIVATE cblas lapack ${GTEST_LIBRARIES} ${Boost_LIBRARIES} )

if( CMAKE_CXX_COMPILER MATCHES ".*/hcc$" )
  # Remove following when hcc is fixed; hcc emits following spurious warning ROCm v1.6.1
  # "clang-5.0: warning: argument unused during compilation: '-isystem /opt/rocm/include'"
  target_compile_options( rocblas-test PRIVATE -Wno-unused-command-line-argument )

  foreach( target ${AMDGPU_TARGETS} )
    target_link_libraries( rocblas-test PRIVATE --amdgpu-target=${target} )
  endforeach( )
endif( )

set_target_properties( rocblas-test PROPERTIES DEBUG_POSTFIX "-d" CXX_EXTENSIONS NO )
set_target_properties( rocblas-test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )
