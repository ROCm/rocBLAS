
if (NOT python)
  set(python "python3") # default for linux
endif()

if (DEFINED ENV{ROCM_PATH})
  set(rocm_bin "$ENV{ROCM_PATH}/bin")
else()
  set(rocm_bin "/opt/rocm/bin")
endif()

# relying on env and path for backward compatibility with external recipes
if (NOT DEFINED ENV{CXX} AND NOT CMAKE_CXX_COMPILER)
  set(CMAKE_CXX_COMPILER "${rocm_bin}/amdclang++")
endif()

if (NOT DEFINED ENV{CC} AND NOT CMAKE_C_COMPILER)
  set(CMAKE_C_COMPILER "${rocm_bin}/amdclang")
endif()

if (NOT DEFINED ENV{FC} AND NOT CMAKE_Fortran_COMPILER)
  set(CMAKE_Fortran_COMPILER "gfortran")
endif()



if (NOT ROCBLAS_TOOLCHAIN_VARS_APPENDED)
  set(ROCBLAS_TOOLCHAIN_VARS_APPENDED True)

  # flags for clang direct use

endif()

if (CONFIG_NO_COMPILER_CHECKS)
  set(CMAKE_CXX_COMPILER_WORKS 1)
  set(CMAKE_C_COMPILER_WORKS 1)
  set(CMAKE_Fortran_COMPILER_WORKS 1)
endif()
