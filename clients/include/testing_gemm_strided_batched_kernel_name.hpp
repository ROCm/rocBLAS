/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <sys/time.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>

#include "rocblas.hpp"
#include "arg_check.h"
#include "rocblas_test_unique_ptr.hpp"
#include "utility.h"
#include "cblas_interface.h"
#include "norm.h"
#include "unit.h"
#include "flops.h"
#include <typeinfo>

using namespace std;

template <typename T>
rocblas_status testing_gemm_strided_batched_kernel_name(Arguments argus)
{
    rocblas_int M = argus.M;
    rocblas_int N = argus.N;
    rocblas_int K = argus.K;

    T h_alpha = argus.alpha;
    T h_beta  = argus.beta;

    rocblas_int lda          = argus.lda;
    rocblas_int ldb          = argus.ldb;
    rocblas_int ldc          = argus.ldc;
    rocblas_int batch_count  = argus.batch_count;
    rocblas_operation transA = char2rocblas_operation(argus.transA_option);
    rocblas_operation transB = char2rocblas_operation(argus.transB_option);

    rocblas_int safe_size = 100; // arbitrarily set to 100

    rocblas_status status;

    std::unique_ptr<rocblas_test::handle_struct> unique_ptr_handle(new rocblas_test::handle_struct);
    rocblas_handle handle = unique_ptr_handle->handle;

    rocblas_int A_row = transA == rocblas_operation_none ? M : K;
    rocblas_int A_col = transA == rocblas_operation_none ? K : M;
    rocblas_int B_row = transB == rocblas_operation_none ? K : N;
    rocblas_int B_col = transB == rocblas_operation_none ? N : K;

    //  make bsa, bsb, bsc two times minimum size so matrices are non-contiguous
    rocblas_int bsa = lda * A_col * 2;
    rocblas_int bsb = ldb * B_col * 2;
    rocblas_int bsc = ldc * N * 2;

    T *dA, *dB, *dC;

    return rocblas_gemm_strided_batched_kernel_name<T>(handle,
                                                       transA,
                                                       transB,
                                                       M,
                                                       N,
                                                       K,
                                                       &h_alpha,
                                                       dA,
                                                       lda,
                                                       bsa,
                                                       dB,
                                                       ldb,
                                                       bsb,
                                                       &h_beta,
                                                       dC,
                                                       ldc,
                                                       bsc,
                                                       batch_count);
}
