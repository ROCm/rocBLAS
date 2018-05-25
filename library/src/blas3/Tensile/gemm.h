#include "rocblas-types.h"
#include "Tensile.h"

/*******************************************************************************
 * Infer Batch Strides
 ******************************************************************************/
inline void infer_batch_strides(rocblas_operation trans_a,
                                rocblas_operation trans_b,
                                rocblas_int m,
                                rocblas_int n,
                                rocblas_int k,
                                rocblas_int ld_a,
                                rocblas_int* stride_a,
                                rocblas_int ld_b,
                                rocblas_int* stride_b,
                                rocblas_int ld_c,
                                rocblas_int* stride_c)
{

    rocblas_int num_cols_c = n;
    rocblas_int num_rows_c = m;
    rocblas_int num_cols_a = (trans_a == rocblas_operation_none ? k : m);
    rocblas_int num_rows_a = (trans_a == rocblas_operation_none ? m : k);
    rocblas_int num_cols_b = (trans_b == rocblas_operation_none ? n : k);
    rocblas_int num_rows_b = (trans_b == rocblas_operation_none ? k : n);

    *stride_a = ld_a * num_cols_a;
    *stride_b = ld_b * num_cols_b;
    *stride_c = ld_c * num_cols_c;

} // infer batched strides

/*******************************************************************************
 * Validate Arguments
 ******************************************************************************/
inline rocblas_status validateArgs(rocblas_handle handle,
                                   rocblas_operation trans_a,
                                   rocblas_operation trans_b,
                                   rocblas_int m,
                                   rocblas_int n,
                                   rocblas_int k,
                                   const void* alpha,
                                   const void* a,
                                   rocblas_int ld_a,
                                   rocblas_int stride_a,
                                   const void* b,
                                   rocblas_int ld_b,
                                   rocblas_int stride_b,
                                   const void* beta,
                                   void* c,
                                   rocblas_int ld_c,
                                   rocblas_int stride_c,
                                   rocblas_int batch_count)
{

    // quick return 0 is valid in BLAS
    if(m == 0 || n == 0 || k == 0 || batch_count == 0)
    {
        return rocblas_status_success;
    }

    // sizes must not be negative
    if(m < 0 || n < 0 || k < 0 || batch_count < 0)
    {
        return rocblas_status_invalid_size;
    }

    // handle must be valid
    if(handle == nullptr)
    {
        return rocblas_status_invalid_handle;
    }

    // pointers must be valid
    if(c == nullptr || a == nullptr || b == nullptr || alpha == nullptr || beta == nullptr)
    {
        return rocblas_status_invalid_pointer;
    }

    rocblas_int num_cols_c = n;
    rocblas_int num_rows_c = m;
    rocblas_int num_cols_a = (trans_a == rocblas_operation_none) ? k : m;
    rocblas_int num_rows_a = (trans_a == rocblas_operation_none) ? m : k;
    rocblas_int num_cols_b = (trans_b == rocblas_operation_none) ? n : k;
    rocblas_int num_rows_b = (trans_b == rocblas_operation_none) ? k : n;

    // leading dimensions must be valid
    if(num_rows_a > ld_a || num_rows_b > ld_b || num_rows_c > ld_c)
    {
        return rocblas_status_invalid_size;
    }

    return rocblas_status_success;
} // validate parameters
