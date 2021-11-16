# ########################################################################
# Copyright 2016-2021 Advanced Micro Devices, Inc.
# ########################################################################

# Linking lapack library requires fortran flags
find_package( cblas CONFIG REQUIRED PATHS ${LAPACK_DIR} )
if( NOT cblas_FOUND )
  message( FATAL_ERROR "cblas is a required dependency and is not found;  try adding cblas path to CMAKE_PREFIX_PATH" )
endif( )

if(LINK_BLIS)
  set( BLIS_CPP ../common/blis_interface.cpp )
endif()

set( rocblas_benchmark_common
      ../common/utility.cpp
      ../common/cblas_interface.cpp
      ../common/rocblas_arguments.cpp
      ${BLIS_CPP}
      ../common/rocblas_parse_data.cpp
    )

add_executable( rocblas-bench client.cpp ${rocblas_benchmark_common} )

target_compile_definitions( rocblas-bench PRIVATE ${TENSILE_DEFINES} )

target_compile_features( rocblas-bench PRIVATE cxx_static_assert cxx_nullptr cxx_auto_type )

# Internal header includes
target_include_directories( rocblas-bench
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/blas1>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/blas2>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/blas3>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/blas_ex>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/library/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/library/src/include>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
)

if (NOT WIN32)
  set( BLIS_INCLUDE_DIR ${BUILD_DIR}/deps/blis/include/blis )
  set( BLIS_LIBRARY ${BUILD_DIR}/deps/blis/lib/libblis.so )
else()
  set( BLIS_INCLUDE_DIR ${BLIS_DIR}/include/blis CACHE PATH "Blis library include path" )
  find_library( BLIS_LIBRARY libblis
                  PATHS ${BLIS_DIR}/lib
                  REQUIRED
                  NO_DEFAULT_PATH
              )
endif()


# External header includes included as system files
target_include_directories( rocblas-bench
  SYSTEM PRIVATE
    $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${CBLAS_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${BLIS_INCLUDE_DIR}>
)

if( NOT WIN32 )
  target_link_libraries( rocblas-bench PRIVATE rocblas_fortran_client )
endif( )
target_link_libraries( rocblas-bench PRIVATE roc::rocblas lapack cblas )

if(LINK_BLIS)
  target_link_libraries( rocblas-bench PRIVATE ${BLIS_LIBRARY} )
endif()

if( CUDA_FOUND )
  target_include_directories( rocblas-bench
    PRIVATE
      $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
      $<BUILD_INTERFACE:${hip_INCLUDE_DIRS}>
    )
  target_compile_definitions( rocblas-bench PRIVATE __HIP_PLATFORM_NVCC__ )
  target_link_libraries( rocblas-bench PRIVATE ${CUDA_LIBRARIES} )
else( )
  # auto set in hip_common.h
  #target_compile_definitions( rocblas-bench PRIVATE __HIP_PLATFORM_HCC__ )
  target_link_libraries( rocblas-bench PRIVATE hip::host hip::device )
endif( )

if( CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # GCC or hip-clang needs specific flags to turn on f16c intrinsics
  target_compile_options( rocblas-bench PRIVATE -mf16c )
endif( )

target_compile_options(rocblas-bench PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${COMMON_CXX_OPTIONS}>)
# target_compile_options does not go to linker like CMAKE_CXX_FLAGS does, so manually add
if (NOT WIN32)
  list( APPEND COMMON_LINK_LIBS "-lm -lstdc++fs")
else()
  list( APPEND COMMON_LINK_LIBS "libomp")
endif()
target_link_libraries( rocblas-bench PRIVATE ${COMMON_LINK_LIBS} )

set_target_properties( rocblas-bench PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
)
add_dependencies( rocblas-bench rocblas-common )
add_subdirectory ( ./perf_script )
target_compile_definitions( rocblas-bench PRIVATE ROCBLAS_BENCH ROCM_USE_FLOAT16 ROCBLAS_INTERNAL_API )
