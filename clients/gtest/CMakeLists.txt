# ########################################################################
# Copyright 2018-2021 Advanced Micro Devices, Inc.
# ########################################################################


# Linking lapack library requires fortran flags 
find_package( cblas CONFIG REQUIRED PATHS ${LAPACK_DIR} )
if( NOT cblas_FOUND )
  message( FATAL_ERROR "cblas is a required dependency and is not found;  try adding cblas path to CMAKE_PREFIX_PATH" )
endif( )

enable_testing()
find_package( GTest REQUIRED )

if( BUILD_WITH_TENSILE )
  set(rocblas_tensile_test_source
      # Keep multiheaded_gtest.cpp first as we want
      # to allow it to create the first TensileHost !
      # Current GTESTs are run in the linking order as added with global variables
      multiheaded_gtest.cpp
      # use of tensile based functions (gemm)
      atomics_mode_gtest.cpp
      gemm_gtest.cpp
      trmm_gtest.cpp
      trsm_gtest.cpp
      trsv_gtest.cpp
      trtri_gtest.cpp
      )
endif()

set(rocblas_no_tensile_test_source
    rocblas_gtest_main.cpp
    rocblas_test.cpp
    general_gtest.cpp
    set_get_pointer_mode_gtest.cpp
    set_get_atomics_mode_gtest.cpp
    logging_mode_gtest.cpp
    ostream_threadsafety_gtest.cpp
    set_get_vector_gtest.cpp
    set_get_matrix_gtest.cpp
    blas1_gtest.cpp
    blas1_ex_gtest.cpp
    # blas2
    gbmv_gtest.cpp
    gemv_gtest.cpp
    hbmv_gtest.cpp
    hemv_gtest.cpp
    her_gtest.cpp
    her2_gtest.cpp
    hpmv_gtest.cpp
    hpr_gtest.cpp
    hpr2_gtest.cpp
    trmv_gtest.cpp
    tpmv_gtest.cpp
    tbmv_gtest.cpp
    tbsv_gtest.cpp
    tpsv_gtest.cpp
    ger_gtest.cpp
    geru_gtest.cpp
    gerc_gtest.cpp
    spr_gtest.cpp
    spr2_gtest.cpp
    syr_gtest.cpp
    syr2_gtest.cpp
    sbmv_gtest.cpp
    spmv_gtest.cpp
    symv_gtest.cpp
    # blas3
    hemm_gtest.cpp
    herk_gtest.cpp
    her2k_gtest.cpp
    herkx_gtest.cpp
    symm_gtest.cpp
    syrk_gtest.cpp
    syr2k_gtest.cpp
    syrkx_gtest.cpp
    geam_gtest.cpp
    dgmm_gtest.cpp
    )

if(LINK_BLIS)
  set( BLIS_CPP ../common/blis_interface.cpp )
endif()

set( rocblas_benchmark_common
      ../common/rocblas_arguments.cpp
      ../common/utility.cpp
      ../common/cblas_interface.cpp
      ${BLIS_CPP}
      ../common/rocblas_parse_data.cpp
    )

# Keep ${rocblas_tensile_test_source} first, so that multiheaded tests are the
# first to initialize Tensile.
if (NOT WIN32)
  add_executable( rocblas-test ${rocblas_tensile_test_source} ${rocblas_no_tensile_test_source} $<TARGET_OBJECTS:rocblas_fortran> ${rocblas_benchmark_common} )
else()
  add_executable( rocblas-test ${rocblas_tensile_test_source} ${rocblas_no_tensile_test_source} ${rocblas_benchmark_common} )
  target_compile_definitions( rocblas-test PRIVATE _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING )
endif()

target_compile_definitions( rocblas-test PRIVATE ${TENSILE_DEFINES} GOOGLE_TEST )

# Internal header includes
target_include_directories( rocblas-test
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/blas1>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/blas2>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/blas3>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/blas_ex>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/include>
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
target_include_directories( rocblas-test
  SYSTEM PRIVATE
    $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${GTEST_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${CBLAS_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${BLIS_INCLUDE_DIR}>
)

if( NOT WIN32 )
  target_link_libraries( rocblas-test PRIVATE rocblas_fortran_client )
endif( )
target_link_libraries( rocblas-test PRIVATE roc::rocblas ${GTEST_BOTH_LIBRARIES} lapack cblas )

if(LINK_BLIS)
  target_link_libraries( rocblas-test PRIVATE ${BLIS_LIBRARY} )
endif()


if( CUDA_FOUND )
  target_include_directories( rocblas-test
    PRIVATE
      $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
      $<BUILD_INTERFACE:${hip_INCLUDE_DIRS}>
    )
  target_compile_definitions( rocblas-test PRIVATE __HIP_PLATFORM_NVCC__ )
  target_link_libraries( rocblas-test PRIVATE ${CUDA_LIBRARIES} )
else( )
  # auto set in hip_common.h
  #target_compile_definitions( rocblas-test PRIVATE __HIP_PLATFORM_HCC__ )
  target_link_libraries( rocblas-test PRIVATE hip::host hip::device )
endif( )

set( ROCBLAS_TEST_DATA "${PROJECT_BINARY_DIR}/staging/rocblas_gtest.data")
add_custom_command( OUTPUT "${ROCBLAS_TEST_DATA}"
                    COMMAND ${python} ../common/rocblas_gentest.py -I ../include rocblas_gtest.yaml -o "${ROCBLAS_TEST_DATA}"
                    DEPENDS ../common/rocblas_gentest.py ../include/rocblas_common.yaml general_gtest.yaml blas1_gtest.yaml dgmm_gtest.yaml gbmv_gtest.yaml geam_gtest.yaml gemm_batched_gtest.yaml gemm_gtest.yaml gemm_strided_batched_gtest.yaml gemv_gtest.yaml ger_gtest.yaml geruc_gtest.yaml hbmv_gtest.yaml hemm_gtest.yaml hemv_gtest.yaml her2_gtest.yaml her2k_gtest.yaml her_gtest.yaml herk_gtest.yaml herkx_gtest.yaml hpmv_gtest.yaml hpr2_gtest.yaml hpr_gtest.yaml known_bugs.yaml logging_mode_gtest.yaml atomics_mode_gtest.yaml ostream_threadsafety_gtest.yaml rocblas_gtest.yaml sbmv_gtest.yaml set_get_matrix_gtest.yaml set_get_pointer_mode_gtest.yaml set_get_atomics_mode_gtest.yaml set_get_vector_gtest.yaml spmv_gtest.yaml spr2_gtest.yaml spr_gtest.yaml symm_gtest.yaml symv_gtest.yaml syr2_gtest.yaml syr2k_gtest.yaml syr_gtest.yaml syrk_gtest.yaml syrkx_gtest.yaml tbmv_gtest.yaml tbsv_gtest.yaml tpmv_gtest.yaml tpsv_gtest.yaml trmm_gtest.yaml trmv_gtest.yaml trsm_gtest.yaml trsv_gtest.yaml trtri_gtest.yaml multiheaded_gtest.yaml
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )
add_custom_target( rocblas-test-data
                   DEPENDS "${ROCBLAS_TEST_DATA}" )

add_dependencies( rocblas-test rocblas-test-data rocblas-common )

if( CMAKE_COMPILER_IS_GNUCXX OR CXX_VERSION_STRING MATCHES "clang" )
  # GCC or hip-clang needs specific flag to turn on f16c intrinsics
  target_compile_options( rocblas-test PRIVATE -mf16c )
endif( )

target_compile_options(rocblas-test PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${COMMON_CXX_OPTIONS}>)
# target_compile_options does not go to linker like CMAKE_CXX_FLAGS does, so manually add
if (NOT WIN32)
  list( APPEND COMMON_LINK_LIBS "-lm -lstdc++fs")
  else()
  list( APPEND COMMON_LINK_LIBS "libomp")
endif()
target_link_libraries( rocblas-test PRIVATE ${COMMON_LINK_LIBS} )

set_target_properties( rocblas-test PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  IMPORT_PREFIX ""
  IMPORT_SUFFIX ".lib"
  LINKER_LANGUAGE CXX
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
)
target_compile_definitions( rocblas-test PRIVATE ROCM_USE_FLOAT16 ROCBLAS_INTERNAL_API )
