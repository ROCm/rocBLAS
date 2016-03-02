# ########################################################################
# Copyright 2016 Advanced Micro Devices, Inc.
# ########################################################################

add_executable( gemm-expert gemm-expert.cpp )
target_link_libraries( gemm-expert rocblas )
