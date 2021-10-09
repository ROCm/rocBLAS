
if (DEFINED ENV{ROCM_PATH})
  set(rocm_bin "$ENV{ROCM_PATH}/bin")
else()
  set(rocm_bin "/opt/rocm/bin")
endif()

set(CMAKE_CXX_COMPILER "${rocm_bin}/hipcc")
set(CMAKE_C_COMPILER "${rocm_bin}/hipcc")
set(python "python3")

# TODO remove, just to speed up slow cmake
# set(CMAKE_C_COMPILER_WORKS 1)
# set(CMAKE_CXX_COMPILER_WORKS 1)
