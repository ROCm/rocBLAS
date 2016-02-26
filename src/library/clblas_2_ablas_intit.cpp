#include "rocblas.h"

rocblas_status
clblas_2_rocblas_init_matrix(
                           const rocblas_precision *precision,
                           const rocblas_order *order,
                           const rocblas_transpose *trans,
                           const rocblas_uplo *uplo,
                           const rocblas_diag *diag,
                           const size_t *M,
                           const size_t *N,
                           const size_t *K,
                           size_t offset,
                           size_t ldX,
                           void *X,
                           rocblas_matrix *rocblas_X)
{
    //unfinished
    if (rocblas_X == NULL)
        return rocblas_invalid_matA;//throw some kind of error

    rocblas_X->data = X;
    rocblas_X->precision = *precision;
    rocblas_X->offset = offset;

    if ((uplo == NULL) && (diag == NULL))
    {
        //not a symmetric matrix

        if (*order == rocblas_column_major)
        {
            if (N == NULL)
            {
                if (trans == NULL)
                {
                    //matrix C does not have trans
                    rocblas_X->num_cols = *M;
                    rocblas_X->num_rows = *K;
                    rocblas_X->num_matrices = 1;
                    rocblas_X->col_stride = ldX;
                    rocblas_X->row_stride = 1;
                    rocblas_X->matrix_stride = ldX*(*K);
                    return rocblas_success;
                }
                else if (*trans == rocblas_no_trans)
                {
                    rocblas_X->num_cols = *M;
                    rocblas_X->num_rows = *K;
                    rocblas_X->num_matrices = 1;
                    rocblas_X->col_stride = ldX;
                    rocblas_X->row_stride = 1;
                    rocblas_X->matrix_stride = ldX*(*K);
                    return rocblas_success;
                }
                else if (*trans == rocblas_trans)
                {
                    rocblas_X->num_cols = *K;
                    rocblas_X->num_rows = *M;
                    rocblas_X->num_matrices = 1;
                    rocblas_X->col_stride = ldX;
                    rocblas_X->row_stride = 1;
                    rocblas_X->matrix_stride = ldX*(*M);
                    return rocblas_success;
                }
                else if (*trans == rocblas_conj_trans)
                {
                }
            }
            else if (M == NULL)
            {
            }
            else if (K == NULL)
            {
            }
            else
            {
               //it would be confusing to have values for all M, N and K
               return rocblas_invalid_matA;
            }

        }



        return rocblas_invalid_matA;
    }

return rocblas_success;
}
