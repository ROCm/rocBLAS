#include "rocblas_types.h"
#include "Tensile.h"

TensileDataType conjugate_if_necessary( TensileDataType type, rocblas_operation trans );

void infer_batch_strides(
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    rocblas_int ld_a, rocblas_int *bs_a,
    rocblas_int ld_b, rocblas_int *bs_b,
    rocblas_int ld_c, rocblas_int *bs_c );
