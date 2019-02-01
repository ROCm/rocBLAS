/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include <hip/hip_runtime.h>

#include "rocblas.h"
#include "rocblas_trsm.hpp"
#include "status.h"
#include "definitions.h"
#include "handle.h"
#include "logging.h"
#include "utility.h"
#include "rocblas_unique_ptr.hpp"

namespace {

template <bool to_temp, typename T>
__global__ void strided_vector_copy_kernel(T* x, rocblas_int m, T* x_temp, rocblas_int incx)
{
    ssize_t tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    ssize_t ty = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    ssize_t id_temp = ty * hipBlockDim_x * hipGridDim_x + tx;
    ssize_t id_x    = incx * id_temp;

    if(id_temp < m)
    {
        if(to_temp)
            x_temp[id_temp] = x[id_x];
        else
            x[id_x] = x_temp[id_temp];
    }
}

template <bool to_temp, typename T>
void strided_vector_copy(
    hipStream_t rocblas_stream, T* x, rocblas_int m, T* x_temp, rocblas_int incx)
{
    static constexpr int NB_X = 128;
    static constexpr int NB_Y = 8;
    rocblas_int blocksX       = (m - 1) / NB_X + 1; // parameters for device kernel
    rocblas_int blocksY       = (m - 1) / NB_Y + 1;
    dim3 grid(blocksX, blocksY);
    dim3 threads(NB_X, NB_Y);

    hipLaunchKernelGGL(
        strided_vector_copy_kernel<to_temp>, grid, threads, 0, rocblas_stream, x, m, x_temp, incx);
}

template <typename>
constexpr char rocblas_trsv_name[] = "unknown";
template <>
constexpr char rocblas_trsv_name<float>[] = "rocblas_strsv";
template <>
constexpr char rocblas_trsv_name<double>[] = "rocblas_dtrsv";

/*! \brief BLAS Level 2 API

    \details
    trsv solves

         A*x = b or A^T*x = b,

    where x and b are vectors and A is a triangular matrix.

    The vector x is overwritten on b.

    @param[in]
    handle    rocblas_handle.
              handle to the rocblas library context queue.

    @param[in]
    uplo    rocblas_fill.
            rocblas_fill_upper:  A is an upper triangular matrix.
            rocblas_fill_lower:  A is a  lower triangular matrix.

    @param[in]
    transA     rocblas_operation

    @param[in]
    diag    rocblas_diagonal.
            rocblas_diagonal_unit:     A is assumed to be unit triangular.
            rocblas_diagonal_non_unit:  A is not assumed to be unit triangular.

    @param[in]
    m         rocblas_int
              m specifies the number of rows of b. m >= 0.

    @param[in]
    A         pointer storing matrix A on the GPU,
              of dimension ( lda, m )

    @param[in]
    lda       rocblas_int
              specifies the leading dimension of A.
              lda >= max( 1, m ).

    @param[in]
    x         pointer storing vector x on the GPU.

    @param[in]
    incx      specifies the increment for the elements of x.

    ********************************************************************/

template <rocblas_int BLOCK, typename T>
rocblas_status rocblas_trsv(rocblas_handle handle,
                            rocblas_fill uplo,
                            rocblas_operation transA,
                            rocblas_diagonal diag,
                            rocblas_int m,
                            const T* A,
                            rocblas_int lda,
                            T* x,
                            rocblas_int incx)
{
    if(!handle)
        return rocblas_status_invalid_handle;

    auto pointer_mode = handle->pointer_mode;
    auto layer_mode   = handle->layer_mode;
    if(layer_mode & rocblas_layer_mode_log_trace)
        log_trace(handle, rocblas_trsv_name<T>, uplo, transA, diag, m, A, lda, x, incx);

    if(layer_mode & (rocblas_layer_mode_log_bench | rocblas_layer_mode_log_profile))
    {
        auto uplo_letter   = rocblas_fill_letter(uplo);
        auto transA_letter = rocblas_transpose_letter(transA);
        auto diag_letter   = rocblas_diag_letter(diag);

        if(layer_mode & rocblas_layer_mode_log_bench)
        {
            if(pointer_mode == rocblas_pointer_mode_host)
                log_bench(handle,
                          "./rocblas-bench -f trsv -r",
                          rocblas_precision_string<T>,
                          "--uplo",
                          uplo_letter,
                          "--transposeA",
                          transA_letter,
                          "--diag",
                          diag_letter,
                          "-m",
                          m,
                          "--lda",
                          lda,
                          "--incx",
                          incx);
        }

        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile(handle,
                        rocblas_trsv_name<T>,
                        "uplo",
                        uplo_letter,
                        "transA",
                        transA_letter,
                        "diag",
                        diag_letter,
                        "M",
                        m,
                        "lda",
                        lda,
                        "incx",
                        incx);
    }

    if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
        return rocblas_status_not_implemented;
    if(!A || !x)
        return rocblas_status_invalid_pointer;
    if(m < 0 || lda < m || lda < 1 || !incx)
        return rocblas_status_invalid_size;

    // quick return if possible.
    if(!m)
        return rocblas_status_success;

    // Calling TRSM for now
    rocblas_status status;

    static constexpr T alpha_h{1};
    const T* alpha = &alpha_h;
    if(pointer_mode == rocblas_pointer_mode_device)
    {
        T* alpha_d = (T*)handle->get_trsv_alpha();
        RETURN_IF_HIP_ERROR(hipMemcpy(alpha_d, &alpha_h, sizeof(T), hipMemcpyHostToDevice));
        alpha = alpha_d;
    }

    if(incx == 1)
    {
        status = rocblas_trsm_template<BLOCK>(
            handle, rocblas_side_left, uplo, transA, diag, m, 1, alpha, A, lda, x, m);
    }
    else
    {
        if(incx < 0)
            x -= ssize_t(incx) * (m - 1);

        T* dx_mod = sizeof(T) * m <= WORKBUF_TRSV_X_SZ
                        ? (T*)handle->get_trsv_x()
                        : (T*)rocblas_unique_ptr(rocblas::device_malloc(sizeof(T) * m),
                                                 rocblas::device_free)
                              .get();

        strided_vector_copy<true>(handle->rocblas_stream, x, m, dx_mod, incx);

        status = rocblas_trsm_template<BLOCK>(
            handle, rocblas_side_left, uplo, transA, diag, m, 1, alpha, A, lda, dx_mod, m);

        strided_vector_copy<false>(handle->rocblas_stream, x, m, dx_mod, incx);
    }

    return status;
}

} // namespace

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_strsv(rocblas_handle handle,
                             rocblas_fill uplo,
                             rocblas_operation transA,
                             rocblas_diagonal diag,
                             rocblas_int m,
                             const float* A,
                             rocblas_int lda,
                             float* x,
                             rocblas_int incx)
{
    static constexpr rocblas_int STRSV_BLOCK = 128;
    return rocblas_trsv<STRSV_BLOCK>(handle, uplo, transA, diag, m, A, lda, x, incx);
}

rocblas_status rocblas_dtrsv(rocblas_handle handle,
                             rocblas_fill uplo,
                             rocblas_operation transA,
                             rocblas_diagonal diag,
                             rocblas_int m,
                             const double* A,
                             rocblas_int lda,
                             double* x,
                             rocblas_int incx)
{
    static constexpr rocblas_int DTRSV_BLOCK = 128;
    return rocblas_trsv<DTRSV_BLOCK>(handle, uplo, transA, diag, m, A, lda, x, incx);
}

} // extern "C"
