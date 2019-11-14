# ########################################################################
# Copyright 2016-2019 Advanced Micro Devices, Inc.
# ########################################################################

get_target_property( HIPHCC_LOCATION hip::hip_hcc IMPORTED_LOCATION_RELEASE )

set( rocblas_samples_common ../common/utility.cpp )

add_executable( example-sscal example_sscal.cpp ${rocblas_samples_common} )
add_executable( example-sgemm example_sgemm.cpp ${rocblas_samples_common} )
add_executable( example-sgemm-strided-batched example_sgemm_strided_batched.cpp ${rocblas_samples_common} )
add_executable( example-scal-template example_scal_template.cpp ${rocblas_samples_common} )
add_executable( example-i8_r_gemm_ex example_i8_r_gemm_ex.cpp ${rocblas_samples_common} )
add_executable( example_bf16_r_gemm_ex example_bf16_r_gemm_ex.cpp ${rocblas_samples_common} )

set( sample_list example-sscal example-sgemm example-scal-template example-sgemm-strided-batched example-i8_r_gemm_ex example_bf16_r_gemm_ex)
foreach( exe ${sample_list} )
  target_link_libraries( ${exe} PRIVATE roc::rocblas )
  target_compile_features( ${exe} PRIVATE cxx_static_assert cxx_nullptr cxx_auto_type )

  set_target_properties( ${exe} PROPERTIES CXX_EXTENSIONS NO )
  set_target_properties( ${exe} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )
  target_include_directories( ${exe}
    PRIVATE
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
      )

  target_include_directories( ${exe}
    SYSTEM PRIVATE
      $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
      $<BUILD_INTERFACE:${HCC_INCLUDE_DIRS}>
      )

  if( CUDA_FOUND )
    target_include_directories( ${exe}
      PRIVATE
        $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
      )
    target_compile_definitions( ${exe} PRIVATE __HIP_PLATFORM_NVCC__ )
    target_link_libraries( ${exe} PRIVATE ${CUDA_LIBRARIES} )
  else( )
    # Unfortunatley, we can't link to hip::hip_hcc because it pulls in flags specific to the
    # hcc compiler, and we want to be able to build with host compilers
    target_compile_definitions( ${exe} PRIVATE __HIP_PLATFORM_HCC__ )
    target_link_libraries( ${exe} PRIVATE ${HIPHCC_LOCATION} )
  endif( )

  if( CMAKE_CXX_COMPILER MATCHES ".*/hcc$" )
    # Remove following when hcc is fixed; hcc emits following spurious warning ROCm v1.6.1
    # "clang-5.0: warning: argument unused during compilation: '-isystem /opt/rocm/include'"
    target_compile_options( ${exe} PRIVATE -Wno-unused-command-line-argument )
    target_include_directories( ${exe} PRIVATE /opt/rocm/hsa/include)
  elseif( CMAKE_COMPILER_IS_GNUCXX OR CXX_VERSION_STRING MATCHES "clang" )
    # GCC or hip-clang needs specific flags to turn on f16c intrinsics
    target_compile_options( ${exe} PRIVATE -mf16c )
  endif( )
endforeach( )

