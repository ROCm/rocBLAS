#include "rocblas-types.h"
#include "Tensile.h"

/*******************************************************************************
 * Infer Batch Strides
 ******************************************************************************/
inline void infer_batch_strides(
    rocblas_order order,
    rocblas_operation trans_a, rocblas_operation trans_b,
    rocblas_int m, rocblas_int n, rocblas_int k,
    rocblas_int ld_a, rocblas_int *bs_a,
    rocblas_int ld_b, rocblas_int *bs_b,
    rocblas_int ld_c, rocblas_int *bs_c ) {

  rocblas_int num_cols_c = n;
  rocblas_int num_rows_c = m;
  rocblas_int num_cols_a = (trans_a == rocblas_operation_none ? k : m);
  rocblas_int num_rows_a = (trans_a == rocblas_operation_none ? m : k);
  rocblas_int num_cols_b = (trans_b == rocblas_operation_none ? n : k);
  rocblas_int num_rows_b = (trans_b == rocblas_operation_none ? k : n);

  rocblas_int dim1_size_a = (order==rocblas_order_column_major)
      ? num_cols_a : num_rows_a;
  rocblas_int dim1_size_b = (order==rocblas_order_column_major)
      ? num_cols_b : num_rows_b;
  rocblas_int dim1_size_c = (order==rocblas_order_column_major)
      ? num_cols_c : num_rows_c;

  *bs_a = ld_a * dim1_size_a;
  *bs_b = ld_b * dim1_size_b;
  *bs_c = ld_c * dim1_size_c;

} // infer batched strides


/*******************************************************************************
 * Validate Arguments
 ******************************************************************************/
inline rocblas_status validateArgs(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation trans_a, rocblas_operation trans_b,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const void *alpha,
    const void *a, rocblas_int ld_a, rocblas_int bs_a,
    const void *b, rocblas_int ld_b, rocblas_int bs_b,
    const void *beta,
    void *c, rocblas_int ld_c, rocblas_int bs_c, rocblas_int b_c
    ) {

  // quick return 0 is valid in BLAS
  if ( m == 0 || n == 0 || k == 0 || b_c == 0) {
    return rocblas_status_success;
  }

  // sizes must not be negative
  if ( m < 0 || n < 0 || k < 0 || b_c < 0) {
    return rocblas_status_invalid_size;
  }

  // strides must not be negative
  if ( m < 0 || n < 0 || k < 0 || b_c < 0) {
    return rocblas_status_invalid_size;
  }

  // handle must be valid
  if (handle == nullptr) {
    return rocblas_status_invalid_handle;
  }

  // pointers must be valid
  if ( c == nullptr
      || a == nullptr
      || b == nullptr
      || alpha == nullptr
      || beta == nullptr ) {
    return rocblas_status_invalid_pointer;
  }

  rocblas_int num_cols_c = n;
  rocblas_int num_rows_c = m;
  rocblas_int num_cols_a = (trans_a == rocblas_operation_none) ? k : m;
  rocblas_int num_rows_a = (trans_a == rocblas_operation_none) ? m : k;
  rocblas_int num_cols_b = (trans_b == rocblas_operation_none) ? n : k;
  rocblas_int num_rows_b = (trans_b == rocblas_operation_none) ? k : n;

  // valid strides
  if(order==rocblas_order_column_major){
    if( num_rows_a > ld_a
        || num_rows_b > ld_b
        || num_rows_c > ld_c) {
      return rocblas_status_invalid_size;
    }
  } else {
    if( num_cols_a > ld_a
        || num_cols_b > ld_b
        || num_cols_c > ld_c) {
      return rocblas_status_invalid_size;
    }
  }

  // TODO re-write these checks
#if 0
  // validate tensor c
  if (tensor_c.dimensions[0].stride < 1) {
    // user gave invalid ls_c
    return rocblas_status_invalid_size;
  }
  if (tensor_c.dimensions[1].stride < tensor_c.dimensions[0].stride * tensor_c.dimensions[0].size ) {
    // user gave invalid ld_c
    return rocblas_status_invalid_size;
  }
  if (tensor_c.dimensions[2].stride < tensor_c.dimensions[1].stride * tensor_c.dimensions[1].size ) {
    // user gave invalid bs_c
    return rocblas_status_invalid_size;
  }

  // validate tensor a
  if (tensor_a.dimensions[0].stride < 1) {
    // user gave invalid ls_a
    return rocblas_status_invalid_size;
  }
  if (tensor_a.dimensions[1].stride < tensor_a.dimensions[0].stride * tensor_a.dimensions[0].size ) {
    // user gave invalid ld_a
    return rocblas_status_invalid_size;
  }
  if (tensor_a.dimensions[2].stride < tensor_a.dimensions[1].stride * tensor_a.dimensions[1].size ) {
    // user gave invalid bs_a
    return rocblas_status_invalid_size;
  }

  // validate tensor b
  if (tensor_b.dimensions[0].stride < 1) {
    // user gave invalid ls_b
    return rocblas_status_invalid_size;
  }
  if (tensor_b.dimensions[1].stride < tensor_b.dimensions[0].stride * tensor_b.dimensions[0].size ) {
    // user gave invalid ld_b
    return rocblas_status_invalid_size;
  }
  if (tensor_b.dimensions[2].stride < tensor_b.dimensions[1].stride * tensor_b.dimensions[1].size ) {
    // user gave invalid bs_b
    return rocblas_status_invalid_size;
  }
#endif
  return rocblas_status_success;
} // validate parameters

