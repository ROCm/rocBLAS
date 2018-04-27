# ########################################################################
# Copyright 2016 Advanced Micro Devices, Inc.
# ########################################################################

# set( Boost_DEBUG ON )
set( Boost_USE_MULTITHREADED ON )
set( Boost_DETAILED_FAILURE_MSG ON )
set( Boost_ADDITIONAL_VERSIONS 1.65.1 1.65 )
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

find_package( GTest REQUIRED )

set( THREADS_PREFER_PTHREAD_FLAG ON )
find_package( Threads REQUIRED )

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
    logging_mode_gtest.cpp
    set_get_vector_gtest.cpp
    set_get_matrix_gtest.cpp
    blas1_gtest.cpp
    gemv_gtest.cpp
    ger_gtest.cpp
    syr_gtest.cpp
    geam_gtest.cpp
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
target_compile_features( rocblas-test PRIVATE cxx_static_assert cxx_nullptr cxx_auto_type )

if( BUILD_WITH_TENSILE )
    target_compile_definitions( rocblas-test PRIVATE BUILD_WITH_TENSILE=1 GOOGLE_TEST )
else()
    target_compile_definitions( rocblas-test PRIVATE BUILD_WITH_TENSILE=0 GOOGLE_TEST )
endif()

# Internal header includes
target_include_directories( rocblas-test
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
)

# External header includes included as system files
target_include_directories( rocblas-test
  SYSTEM PRIVATE
    $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${GTEST_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${CBLAS_INCLUDE_DIRS}>
    )

target_link_libraries( rocblas-test PRIVATE ${GTEST_LIBRARIES} ${Boost_LIBRARIES} cblas lapack roc::rocblas Threads::Threads )
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_SHARED_LIBRARY_RUNTIME_CXX_FLAG}/opt/rocm/lib")

get_target_property( HIPHCC_LOCATION hip::hip_hcc IMPORTED_LOCATION_RELEASE )

if( CUDA_FOUND )
  target_include_directories( rocblas-test
    PRIVATE
      $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
      $<BUILD_INTERFACE:${hip_INCLUDE_DIRS}>
    )
  target_compile_definitions( rocblas-test PRIVATE __HIP_PLATFORM_NVCC__ )
  target_link_libraries( rocblas-test PRIVATE ${CUDA_LIBRARIES} )
else( )
  target_compile_definitions( rocblas-test PRIVATE __HIP_PLATFORM_HCC__ )
  target_link_libraries( rocblas-test PRIVATE ${HIPHCC_LOCATION} )
endif( )

if( CMAKE_CXX_COMPILER MATCHES ".*/hcc$" )
  # Remove following when hcc is fixed; hcc emits following spurious warning ROCm v1.6.1
  # "clang-5.0: warning: argument unused during compilation: '-isystem /opt/rocm/include'"
  target_compile_options( rocblas-test PRIVATE -Wno-unused-command-line-argument -mf16c )

elseif( CMAKE_COMPILER_IS_GNUCXX )
  # GCC needs specific flag to turn on f16c intrinsics
  target_compile_options( rocblas-test PRIVATE -mf16c )
endif( )

set_target_properties( rocblas-test PROPERTIES CXX_EXTENSIONS NO )
set_target_properties( rocblas-test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )
