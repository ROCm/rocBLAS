/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "handle.hpp"
#include "logging.hpp"
#include "rocblas.h"
#include "utility.hpp"
#include <hip/hip_runtime.h>

/**
  *
  * rocblas_check_numerics_vector_kernel(n, xa, offset_x, inc_x, stride_x, abnormal)
  *
  * Info about rocblas_check_numerics_vector_kernel function:
  *
  *    It is the kernel function which checks a vector for numerical abnormalities such as NaN/zero/Inf and updates the structure.
  *
  * Parameters   : n            : Total number of elements in the vector
  *                xa           : Pointer to the vector which is under consideration for numerical abnormalities
  *                offset_x     : Offset of vector 'xa'
  *                inc_x        : Stride between consecutive values of vector 'xa'
  *                stride_x     : Specifies the pointer increment between one vector 'x_i' and the next one (xa_i+1) (where (xa_i) is the i-th instance of the batch)
  *                abnormal     : Device pointer to the rocblas_check_numerics_t structure
  *
  * Return Value : Nothing --
  *
**/

template <typename T>
__global__ void rocblas_check_numerics_vector_kernel(rocblas_int               n,
                                                     T*                        xa,
                                                     ptrdiff_t                 offset_x,
                                                     rocblas_int               inc_x,
                                                     rocblas_stride            stride_x,
                                                     rocblas_check_numerics_t* abnormal)
{
    auto*     x   = load_ptr_batch(xa, hipBlockIdx_y, offset_x, stride_x);
    ptrdiff_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    //Check every element of the x vector for a NaN/zero/Inf
    if(tid < n)
    {
        T value = (T)x[tid * inc_x];
        if(!abnormal->has_zero && rocblas_iszero(value))
            abnormal->has_zero = true;
        if(!abnormal->has_NaN && rocblas_isnan(value))
            abnormal->has_NaN = true;
        if(!abnormal->has_Inf && rocblas_isinf(value))
            abnormal->has_Inf = true;
    }
}

template <typename T>
ROCBLAS_EXPORT_NOINLINE rocblas_status
    rocblas_check_numerics_vector_template(const char*    function_name,
                                           rocblas_handle handle,
                                           rocblas_int    n,
                                           T*             x,
                                           rocblas_int    offset_x,
                                           rocblas_int    inc_x,
                                           rocblas_stride stride_x,
                                           rocblas_int    batch_count,
                                           const int      check_numerics,
                                           bool           is_input);
