#include "ablas.h"

ablas_status
clblas_2_ablas_init_matrix(
                           const ablas_precision *precision,
                           const ablas_order *order,
                           const ablas_transpose *trans,
                           const ablas_uplo *uplo,
                           const ablas_diag *diag,
                           const size_t *M,
                           const size_t *N,
                           const size_t *K,
                           size_t offset,
                           size_t ldX,
                           void *X,
                           ablas_matrix *ablas_X)
{
    //unfinished
    if (ablas_X == NULL)
        return -1;//throw some kind of error
    
    ablas_X->data = X;
    ablas_X->precision = *precision;
    ablas_X->offset = offset;

    if ((uplo == NULL) && (diag == NULL))
    { 
        //not a symmetric matrix
            
        if (*order == ablas_column_major)
        { 
            if (N == NULL)
            { 
                if (trans == NULL)
                {
                    //matrix C does not have trans 
                    ablas_X->num_cols = *M;
                    ablas_X->num_rows = *K;
                    ablas_X->num_matrices = 1;
                    ablas_X->col_stride = ldX;
                    ablas_X->row_stride = 1;
                    ablas_X->matrix_stride = ldX*(*K);
                    return 0;
                }
                else if (*trans == ablas_no_trans)
                { 
                    ablas_X->num_cols = *M;
                    ablas_X->num_rows = *K;
                    ablas_X->num_matrices = 1; 
                    ablas_X->col_stride = ldX;
                    ablas_X->row_stride = 1;
                    ablas_X->matrix_stride = ldX*(*K);
                    return 0;
                }
                else if (*trans == ablas_trans)
                {
                    ablas_X->num_cols = *K;
                    ablas_X->num_rows = *M;
                    ablas_X->num_matrices = 1;
                    ablas_X->col_stride = ldX;
                    ablas_X->row_stride = 1;
                    ablas_X->matrix_stride = ldX*(*M);
                    return 0;
                }
                else if (*trans == ablas_conjugate_trans)
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
               return -1;
            }

        }



        return -1;
    }
}