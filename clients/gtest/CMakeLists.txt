# ########################################################################
# Copyright 2018-2019 Advanced Micro Devices, Inc.
# ########################################################################

# For debugging, uncomment this
# set( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g -O0" )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

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
      gemm_parallel_gtest.cpp
      trsm_gtest.cpp
      trsm_ex_gtest.cpp
      trsv_gtest.cpp
      )

  set(rocblas_test_source
    ${Tensile_TEST_SRC}
    geam_gtest.cpp
    trtri_gtest.cpp
    )
endif( )

set(rocblas_no_tensile_test_source
    rocblas_gtest_main.cpp
    set_get_pointer_mode_gtest.cpp
    logging_mode_gtest.cpp
    set_get_vector_gtest.cpp
    set_get_matrix_gtest.cpp
    blas1_gtest.cpp
    # blas2
    gemv_gtest.cpp
    ger_gtest.cpp
    syr_gtest.cpp
    symv_gtest.cpp
    )

if(LINK_BLIS)
  set( BLIS_CPP ../common/blis_interface.cpp )
endif()

set( rocblas_benchmark_common
      ../common/utility.cpp
      ../common/cblas_interface.cpp
      ${BLIS_CPP}
      ../common/rocblas_parse_data.cpp
    )

add_executable( rocblas-test ${rocblas_no_tensile_test_source} ${rocblas_test_source} ${rocblas_benchmark_common} )

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
    $<BUILD_INTERFACE:${HCC_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${GTEST_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${CBLAS_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${BLIS_INCLUDE_DIR}>
    )

set( BLIS_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/build/deps/blis/include/blis )
set( BLIS_LIBRARY ${CMAKE_SOURCE_DIR}/build/deps/blis/lib/libblis.so )

if( OS_ID_rhel OR OS_ID_centos OR OS_ID_sles)
    if( OS_ID_rhel OR OS_ID_centos)
        # defer OpenMP include as search order must come after clang
        set( XXX_OPENMP_INCLUDE_DIR /opt/rh/devtoolset-7/root/usr/lib/gcc/x86_64-redhat-linux/7/include )
        set( OPENMP_LIBRARY /opt/rh/devtoolset-7/root/usr/lib/gcc/x86_64-redhat-linux/7/libgomp.so )
    else()
    #SLES
        set( OPENMP_INCLUDE_DIR /usr/lib64/gcc/x86_64-suse-linux/7/include/ )
        set( OPENMP_LIBRARY /usr/lib64/gcc/x86_64-suse-linux/7/libgomp.so )
    endif()

    if(EXISTS /opt/rocm/hcc/lib/clang/10.0.0/include/immintrin.h)
        set( CLANG_INCLUDE_DIR /opt/rocm/hcc/lib/clang/10.0.0/include )
    elseif (EXISTS /opt/rocm/hcc/lib/clang/9.0.0/include/immintrin.h)
        set( CLANG_INCLUDE_DIR /opt/rocm/hcc/lib/clang/9.0.0/include )
    else()
        error("cannot find immintrin.h")
    endif()

    # External header includes included as system files
    target_include_directories( rocblas-test
      SYSTEM PRIVATE
        $<BUILD_INTERFACE:${CLANG_INCLUDE_DIR}>
        $<BUILD_INTERFACE:${BLIS_INCLUDE_DIR}>
        $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
        $<BUILD_INTERFACE:${HCC_INCLUDE_DIRS}>
        $<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>
        $<BUILD_INTERFACE:${GTEST_INCLUDE_DIRS}>
        $<BUILD_INTERFACE:${CBLAS_INCLUDE_DIRS}>
        $<BUILD_INTERFACE:${OPENMP_INCLUDE_DIR}>
        )

    if(LINK_BLIS)
      target_link_libraries( rocblas-test PRIVATE ${GTEST_LIBRARIES} ${Boost_LIBRARIES} ${BLIS_LIBRARY} ${OPENMP_LIBRARY} cblas lapack roc::rocblas )
    else()
      target_link_libraries( rocblas-test PRIVATE ${GTEST_LIBRARIES} ${Boost_LIBRARIES} ${OPENMP_LIBRARY} cblas lapack roc::rocblas )
    endif()
else()
    # External header includes included as system files
    target_include_directories( rocblas-test
      SYSTEM PRIVATE
        $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
        $<BUILD_INTERFACE:${HCC_INCLUDE_DIRS}>
        $<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>
        $<BUILD_INTERFACE:${GTEST_INCLUDE_DIRS}>
        $<BUILD_INTERFACE:${CBLAS_INCLUDE_DIRS}>
        $<BUILD_INTERFACE:${BLIS_INCLUDE_DIR}>
    )

    if(LINK_BLIS)
      target_link_libraries( rocblas-test PRIVATE ${GTEST_LIBRARIES} ${Boost_LIBRARIES} ${BLIS_LIBRARY} cblas lapack roc::rocblas )
    else()
      target_link_libraries( rocblas-test PRIVATE ${GTEST_LIBRARIES} ${Boost_LIBRARIES} cblas lapack roc::rocblas )
    endif()
endif()

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

set( ROCBLAS_TEST_DATA "${PROJECT_BINARY_DIR}/staging/rocblas_gtest.data")
add_custom_command( OUTPUT "${ROCBLAS_TEST_DATA}"
                    COMMAND ../common/rocblas_gentest.py -I ../include rocblas_gtest.yaml -o "${ROCBLAS_TEST_DATA}"
                    DEPENDS ../common/rocblas_gentest.py rocblas_gtest.yaml ../include/rocblas_common.yaml known_bugs.yaml blas1_gtest.yaml gemm_gtest.yaml gemm_batched_gtest.yaml gemm_strided_batched_gtest.yaml gemv_gtest.yaml symv_gtest.yaml syr_gtest.yaml ger_gtest.yaml trsm_gtest.yaml trtri_gtest.yaml geam_gtest.yaml set_get_vector_gtest.yaml set_get_matrix_gtest.yaml trsv_gtest.yaml logging_mode_gtest.yaml set_get_pointer_mode_gtest.yaml
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )
add_custom_target( rocblas-test-data
                   DEPENDS "${ROCBLAS_TEST_DATA}" )

add_dependencies( rocblas-test rocblas-test-data rocblas-common )

if( CMAKE_CXX_COMPILER MATCHES ".*/hcc$" )
  # Remove following when hcc is fixed; hcc emits following spurious warning ROCm v1.6.1
  # "clang-5.0: warning: argument unused during compilation: '-isystem /opt/rocm/include'"
  target_compile_options( rocblas-test PRIVATE -Wno-unused-command-line-argument -mf16c )
  target_include_directories( rocblas-test PRIVATE /opt/rocm/hsa/include)

elseif( CMAKE_COMPILER_IS_GNUCXX OR CXX_VERSION_STRING MATCHES "clang" )
  # GCC or hip-clang needs specific flag to turn on f16c intrinsics
  target_compile_options( rocblas-test PRIVATE -mf16c )
endif( )

if( OS_ID_rhel OR OS_ID_centos)
    # force clang includes to take precedence over devtoolset-7 which we only want for OpenMP
    set(CMAKE_CXX_FLAGS "-isystem ${CLANG_INCLUDE_DIR} -isystem ${XXX_OPENMP_INCLUDE_DIR} ${CMAKE_CXX_FLAGS}")
endif()

if( CXX_VERSION_STRING MATCHES "clang" )
  target_link_libraries( rocblas-test PRIVATE -lpthread -lm )
endif( )

set_target_properties( rocblas-test PROPERTIES CXX_EXTENSIONS NO )
set_target_properties( rocblas-test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )
