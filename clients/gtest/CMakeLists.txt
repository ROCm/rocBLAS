# ########################################################################
# Copyright 2018-2020 Advanced Micro Devices, Inc.
# ########################################################################

# For debugging, uncomment this
# set( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g -O0" )
if(EXISTS /etc/redhat-release)
    if(CXX_VERSION_STRING MATCHES "clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp=libgomp -pthread -stdlib=libstdc++ -std=c++14")
    else()
        set(CMAKE_CXX_FLAGS "-isystem /opt/rocm/include ${CMAKE_CXX_FLAGS} -fopenmp=libgomp -pthread")
    endif()
else()
    set(CMAKE_CXX_FLAGS "-isystem /opt/rocm/include ${CMAKE_CXX_FLAGS} -fopenmp -pthread")
endif()

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
      trmm_gtest.cpp
      trsm_gtest.cpp
      trsv_gtest.cpp
      )

  set(rocblas_test_source
    ${Tensile_TEST_SRC}
    trtri_gtest.cpp
    )
endif( )

set(rocblas_no_tensile_test_source
    rocblas_gtest_main.cpp
    rocblas_test.cpp
    set_get_pointer_mode_gtest.cpp
    logging_mode_gtest.cpp
    ostream_threadsafety_gtest.cpp
    set_get_vector_gtest.cpp
    set_get_matrix_gtest.cpp
    blas1_gtest.cpp
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

add_executable( rocblas-test ${rocblas_f90_source} ${rocblas_no_tensile_test_source} ${rocblas_test_source} ${rocblas_benchmark_common} )

if( BUILD_WITH_TENSILE )
    target_compile_definitions( rocblas-test PRIVATE BUILD_WITH_TENSILE=1 GOOGLE_TEST )
else()
    target_compile_definitions( rocblas-test PRIVATE BUILD_WITH_TENSILE=0 GOOGLE_TEST )
endif()

# Internal header includes
target_include_directories( rocblas-test
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/include>
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

set( BLIS_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/${BUILD_DIR}/deps/blis/include/blis )
set( BLIS_LIBRARY ${CMAKE_SOURCE_DIR}/${BUILD_DIR}/deps/blis/lib/libblis.so )

if( OS_ID_rhel OR OS_ID_sles OR OS_ID_centos)
    if( OS_ID_rhel OR OS_ID_centos)
        if( EXISTS "/usr/lib/gcc/x86_64-redhat-linux/8/" )
            set( ROCM_OPENMP_PATH /usr/lib/gcc/x86_64-redhat-linux/8 )
        else()
            set( ROCM_OPENMP_PATH /opt/rh/devtoolset-7/root/usr/lib/gcc/x86_64-redhat-linux/7 )
        endif()

        # defer OpenMP include as search order must come after clang
        set( XXX_OPENMP_INCLUDE_DIR ${ROCM_OPENMP_PATH}/include )
        set( OPENMP_LIBRARY ${ROCM_OPENMP_PATH}/libgomp.so )
        if( CXX_VERSION_STRING MATCHES "clang")
            set( XXX_OPENMP_INCLUDE_DIR ${ROCM_OPENMP_PATH}/include )
        endif()
    else()
    #SLES
        set( OPENMP_INCLUDE_DIR /usr/lib64/gcc/x86_64-suse-linux/7/include/ )
        set( OPENMP_LIBRARY /usr/lib64/gcc/x86_64-suse-linux/7/libgomp.so )
    endif()

    message(STATUS "RocmPath: ${ROCM_PATH}")
    if(EXISTS ${ROCM_PATH}/llvm/lib/clang/11.0.0/include/immintrin.h AND CMAKE_CXX_COMPILER MATCHES ".*/hcc$|.*/hipcc$" )
        set( CLANG_INCLUDE_DIR ${ROCM_PATH}/llvm/lib/clang/11.0.0/include )
    elseif (EXISTS ${ROCM_PATH}/hcc/lib/clang/10.0.0/include/immintrin.h AND CMAKE_CXX_COMPILER MATCHES ".*/hcc$|.*/hipcc$" )
        set( CLANG_INCLUDE_DIR ${ROCM_PATH}/hcc/lib/clang/10.0.0/include )
    elseif (EXISTS ${ROCM_PATH}/hcc/lib/clang/9.0.0/include/immintrin.h AND CMAKE_CXX_COMPILER MATCHES ".*/hcc$|.*/hipcc$" )
        set( CLANG_INCLUDE_DIR ${ROCM_PATH}/hcc/lib/clang/9.0.0/include )
    else()
        set( CLANG_INCLUDE_DIR )
    endif()

    # External header includes included as system files
    target_include_directories( rocblas-test
      SYSTEM PRIVATE
        $<BUILD_INTERFACE:${CLANG_INCLUDE_DIR}>
        $<BUILD_INTERFACE:${OPENMP_INCLUDE_DIR}>
        $<BUILD_INTERFACE:${BLIS_INCLUDE_DIR}>
        $<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>
        $<BUILD_INTERFACE:${GTEST_INCLUDE_DIRS}>
        $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
        $<BUILD_INTERFACE:${HCC_INCLUDE_DIRS}>
        $<BUILD_INTERFACE:${CBLAS_INCLUDE_DIRS}>
        )

    if(LINK_BLIS)
      target_link_libraries( rocblas-test PRIVATE ${OPENMP_LIBRARY} ${BLIS_LIBRARY} ${Boost_LIBRARIES} ${GTEST_LIBRARIES} cblas lapack roc::rocblas rocblas_fortran_client )
    else()
      target_link_libraries( rocblas-test PRIVATE ${OPENMP_LIBRARY} ${Boost_LIBRARIES} ${GTEST_LIBRARIES} cblas lapack roc::rocblas rocblas_fortran_client )
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
      target_link_libraries( rocblas-test PRIVATE ${GTEST_LIBRARIES} ${Boost_LIBRARIES} ${BLIS_LIBRARY} cblas lapack roc::rocblas rocblas_fortran_client )
    else()
      target_link_libraries( rocblas-test PRIVATE ${GTEST_LIBRARIES} ${Boost_LIBRARIES} cblas lapack roc::rocblas rocblas_fortran_client )
    endif()
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
  target_link_libraries( rocblas-test PRIVATE hip::device )
endif( )

set( ROCBLAS_TEST_DATA "${PROJECT_BINARY_DIR}/staging/rocblas_gtest.data")
add_custom_command( OUTPUT "${ROCBLAS_TEST_DATA}"
                    COMMAND ../common/rocblas_gentest.py -I ../include rocblas_gtest.yaml -o "${ROCBLAS_TEST_DATA}"
                    DEPENDS ../common/rocblas_gentest.py ../include/rocblas_common.yaml blas1_gtest.yaml dgmm_gtest.yaml gbmv_gtest.yaml geam_gtest.yaml gemm_batched_gtest.yaml gemm_gtest.yaml gemm_strided_batched_gtest.yaml gemv_gtest.yaml ger_gtest.yaml geruc_gtest.yaml hbmv_gtest.yaml hemm_gtest.yaml hemv_gtest.yaml her2_gtest.yaml her2k_gtest.yaml her_gtest.yaml herk_gtest.yaml herkx_gtest.yaml hpmv_gtest.yaml hpr2_gtest.yaml hpr_gtest.yaml known_bugs.yaml logging_mode_gtest.yaml ostream_threadsafety_gtest.yaml rocblas_gtest.yaml sbmv_gtest.yaml set_get_matrix_gtest.yaml set_get_pointer_mode_gtest.yaml set_get_vector_gtest.yaml spmv_gtest.yaml spr2_gtest.yaml spr_gtest.yaml symm_gtest.yaml symv_gtest.yaml syr2_gtest.yaml syr2k_gtest.yaml syr_gtest.yaml syrk_gtest.yaml syrkx_gtest.yaml tbmv_gtest.yaml tbsv_gtest.yaml tpmv_gtest.yaml tpsv_gtest.yaml trmm_gtest.yaml trmv_gtest.yaml trsm_gtest.yaml trsv_gtest.yaml trtri_gtest.yaml
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )
add_custom_target( rocblas-test-data
                   DEPENDS "${ROCBLAS_TEST_DATA}" )

add_dependencies( rocblas-test rocblas-test-data rocblas-common )

if( CMAKE_CXX_COMPILER MATCHES ".*/hcc$" )
  # Remove following when hcc is fixed; hcc emits following spurious warning ROCm v1.6.1
  # "clang-5.0: warning: argument unused during compilation: '-isystem ${ROCM_PATH}/include'"
  target_compile_options( rocblas-test PRIVATE -Wno-unused-command-line-argument -mf16c )
  target_include_directories( rocblas-test PRIVATE ${ROCM_PATH}/hsa/include)
elseif( CMAKE_COMPILER_IS_GNUCXX OR CXX_VERSION_STRING MATCHES "clang" )
  # GCC or hip-clang needs specific flag to turn on f16c intrinsics
  target_compile_options( rocblas-test PRIVATE -mf16c )
endif( )

if( CXX_VERSION_STRING MATCHES "clang" )
  target_link_libraries( rocblas-test PRIVATE -lpthread -lstdc++ -lgcc_s -lm -rtlib=compiler-rt )
  if(OS_ID_rhel OR OS_ID_centos)
    set(CMAKE_CXX_FLAGS "-isystem ${CLANG_INCLUDE_DIR} -isystem ${XXX_OPENMP_INCLUDE_DIR} ${CMAKE_CXX_FLAGS}")
  endif( )
else( )
  if(OS_ID_rhel OR OS_ID_centos)
    set(CMAKE_CXX_FLAGS "-isystem ${CLANG_INCLUDE_DIR} -isystem ${XXX_OPENMP_INCLUDE_DIR} ${CMAKE_CXX_FLAGS}")
  endif( )
  set(CMAKE_CXX_FLAGS "-isystem ${ROCM_PATH}/include ${CMAKE_CXX_FLAGS}")
endif( )

set_target_properties( rocblas-test PROPERTIES CXX_EXTENSIONS NO )
set_target_properties( rocblas-test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )
target_compile_definitions( rocblas-test PRIVATE ROCM_USE_FLOAT16 )
