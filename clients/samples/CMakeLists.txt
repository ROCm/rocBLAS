# ########################################################################
# Copyright 2016 Advanced Micro Devices, Inc.
# ########################################################################

set( rocblas_samples_common ../common/utility.cpp )

add_executable( example-sscal example_sscal.cpp ${rocblas_samples_common} )
add_executable( example-scal-template example_scal_template.cpp ../common/rocblas_template_specialization.cpp ${rocblas_samples_common} )

#add_executable( example-openmp example_openmp.cpp ${rocblas_samples_common} )

if( NOT TARGET rocblas )
  find_package( rocblas CONFIG PATHS /opt/rocm/rocblas )

  if( NOT rocblas_FOUND )
    message( FATAL_ERROR "rocBLAS is a required dependency and is not found; try adding rocblas path to CMAKE_PREFIX_PATH")
  endif( )
endif( )

foreach( exe example-sscal;example-scal-template )
  if( TARGET rocblas )
    target_link_libraries( ${exe} PRIVATE rocblas )
  else( )
    target_link_libraries( ${exe} PRIVATE roc::rocblas )
  endif( )

  # Try to test for specific compiler features if cmake version is recent enough
  if( CMAKE_VERSION VERSION_GREATER "3.0" )
    target_compile_features( ${exe} PRIVATE cxx_static_assert cxx_nullptr cxx_auto_type )
  else( )
    # Otherwise, just try to compile the library with a standards flag
    if( CMAKE_COMPILER_IS_GNUCXX OR ( CMAKE_CXX_COMPILER_ID MATCHES "Clang" ) )
      target_compile_options( ${exe} PRIVATE -std=c++11 -pthread )
    endif( )
  endif( )

  set_target_properties( ${exe} PROPERTIES DEBUG_POSTFIX "-d" CXX_EXTENSIONS NO )
  set_target_properties( ${exe} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )
  target_include_directories( ${exe} PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include> )

  if( CMAKE_CXX_COMPILER MATCHES ".*/hcc$" )
    # Remove following when hcc is fixed; hcc emits following spurious warning ROCm v1.6.1
    # "clang-5.0: warning: argument unused during compilation: '-isystem /opt/rocm/include'"
    target_compile_options( ${exe} PRIVATE -Wno-unused-command-line-argument )

    foreach( target ${AMDGPU_TARGETS} )
      target_link_libraries( ${exe} PRIVATE --amdgpu-target=${target} )
    endforeach( )
  endif( )
endforeach( )

