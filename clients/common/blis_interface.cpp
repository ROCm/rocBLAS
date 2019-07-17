#include "blis_interface.hpp"
#include "blis.h"
#include "omp.h"

trans_t blis_transpose(rocblas_operation trans)
{
    if(trans == rocblas_operation_none)
    {
        return BLIS_CONJ_NO_TRANSPOSE;
    }
    else if(trans == rocblas_operation_transpose)
    {
        return BLIS_CONJ_TRANSPOSE;
    }
    else if(trans == rocblas_operation_conjugate_transpose)
    {
        return BLIS_CONJ_TRANSPOSE;
    }
    else
    {
        std::cerr << "rocblas ERROR: trans != N, T, C" << std::endl;
        exit(1);
    }
}

void setup_blis()
{
    bli_init();
    bli_thread_set_num_threads(omp_get_max_threads());
}

void blis_dgemm(rocblas_operation transA,
                rocblas_operation transB,
                rocblas_int       m,
                rocblas_int       n,
                rocblas_int       k,
                double            alpha,
                double*           A,
                rocblas_int       lda,
                double*           B,
                rocblas_int       ldb,
                double            beta,
                double*           C,
                rocblas_int       ldc)
{

    bli_dgemm(blis_transpose(transA),
              blis_transpose(transB),
              m,
              n,
              k,
              (double*)&alpha,
              (double*)A,
              1,
              lda,
              (double*)B,
              1,
              ldb,
              (double*)&beta,
              (double*)C,
              1,
              ldc);
}

void blis_sgemm(rocblas_operation transA,
                rocblas_operation transB,
                rocblas_int       m,
                rocblas_int       n,
                rocblas_int       k,
                float             alpha,
                float*            A,
                rocblas_int       lda,
                float*            B,
                rocblas_int       ldb,
                float             beta,
                float*            C,
                rocblas_int       ldc)
{
    bli_sgemm(blis_transpose(transA),
              blis_transpose(transB),
              m,
              n,
              k,
              (float*)&alpha,
              (float*)A,
              1,
              lda,
              (float*)B,
              1,
              ldb,
              (float*)&beta,
              (float*)C,
              1,
              ldc);
}
