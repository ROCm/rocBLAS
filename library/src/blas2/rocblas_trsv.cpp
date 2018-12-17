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

#define STRSV_BLOCK 128
#define DTRSV_BLOCK 128

template <typename T, bool to_temp>
__global__ void strided_vector_copy_kernel(T* x, rocblas_int m, T* x_temp, rocblas_int incx)
{
    size_t tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    size_t ty = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    int id_temp = ty * hipBlockDim_x * hipGridDim_x + tx;
    int id_x    = incx * id_temp;

    if(id_temp < m)
    {
        if(to_temp)
        {
            x_temp[id_temp] = x[id_x];
        }
        else
        {
            x[id_x] = x_temp[id_temp];
        }
    }
}

template <typename T, bool to_temp>
void strided_vector_copy(
    hipStream_t rocblas_stream, T* x, rocblas_int m, T* x_temp, rocblas_int incx)
{
    rocblas_int blocksX = ((m - 1) / 128) + 1; // parameters for device kernel
    rocblas_int blocksY = ((m - 1) / 8) + 1;
    dim3 grid(blocksX, blocksY, 1);
    dim3 threads(128, 8, 1);

    hipLaunchKernelGGL(strided_vector_copy_kernel<T, to_temp>,
                       dim3(grid),
                       dim3(threads),
                       0,
                       rocblas_stream,
                       x,
                       m,
                       x_temp,
                       incx);
}

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

template <typename T, rocblas_int BLOCK>
rocblas_status rocblas_trsv_template(rocblas_handle handle,
                                     rocblas_fill uplo,
                                     rocblas_operation transA,
                                     rocblas_diagonal diag,
                                     rocblas_int m,
                                     const T* A,
                                     rocblas_int lda,
                                     T* x,
                                     rocblas_int incx)
{
    if(handle == nullptr)
        return rocblas_status_invalid_handle;

    if(handle->pointer_mode == rocblas_pointer_mode_host)
    {
        log_trace(handle,
                  replaceX<T>("rocblas_Xtrsv"),
                  uplo,
                  transA,
                  diag,
                  m,
                  (const void*&)A,
                  lda,
                  (const void*&)x,
                  incx);

        std::string uplo_letter   = rocblas_fill_letter(uplo);
        std::string transA_letter = rocblas_transpose_letter(transA);
        std::string diag_letter   = rocblas_diag_letter(diag);

        log_bench(handle,
                  "./rocblas-bench -f trsv -r",
                  replaceX<T>("X"),
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
    else
    {
        log_trace(handle,
                  replaceX<T>("rocblas_Xtrsv"),
                  uplo,
                  transA,
                  diag,
                  m,
                  (const void*&)A,
                  lda,
                  (const void*&)x,
                  incx);
    }

    if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
        return rocblas_status_not_implemented;
    else if(nullptr == A)
        return rocblas_status_invalid_pointer;
    else if(nullptr == x)
        return rocblas_status_invalid_pointer;
    else if(m < 0)
        return rocblas_status_invalid_size;
    else if(lda < m || lda < 1)
        return rocblas_status_invalid_size;
    else if(0 == incx)
        return rocblas_status_invalid_size;

    // quick return if possible.
    if(m == 0)
        return rocblas_status_success;

    // Calling TRSM for now
    rocblas_status status;
    rocblas_pointer_mode pointer_mode = handle->pointer_mode;
    T alpha_h                         = 1.0f;

    void* alpha_d;
    if(pointer_mode == rocblas_pointer_mode_device)
    {
        alpha_d = handle->get_trsv_alpha();
        hipMemcpy((T*)alpha_d, &alpha_h, sizeof(T), hipMemcpyHostToDevice);
    }

    if(incx == 1)
    {
        status = rocblas_trsm_template<T, BLOCK>(
            handle,
            rocblas_side_left,
            uplo,
            transA,
            diag,
            m,
            1,
            pointer_mode == rocblas_pointer_mode_host ? &alpha_h : (T*)alpha_d,
            A,
            lda,
            x,
            m);
    }
    else
    {
        int offest = (m - 1) * abs(incx);
        if(WORKBUF_TRSV_X_SZ <= m)
        {
            void* dx_mod = handle->get_trsv_x();
            strided_vector_copy<T, true>(
                handle->rocblas_stream, incx < 0 ? x + offest : x, m, (T*)dx_mod, incx);

            status = rocblas_trsm_template<T, BLOCK>(
                handle,
                rocblas_side_left,
                uplo,
                transA,
                diag,
                m,
                1,
                pointer_mode == rocblas_pointer_mode_host ? &alpha_h : (T*)alpha_d,
                A,
                lda,
                (T*)dx_mod,
                m);

            strided_vector_copy<T, false>(
                handle->rocblas_stream, incx < 0 ? x + offest : x, m, (T*)dx_mod, incx);
        }
        else
        {
            auto dx_mod =
                rocblas_unique_ptr{rocblas::device_malloc(sizeof(T) * m), rocblas::device_free};
            strided_vector_copy<T, true>(
                handle->rocblas_stream, incx < 0 ? x + offest : x, m, (T*)dx_mod.get(), incx);

            status = rocblas_trsm_template<T, BLOCK>(
                handle,
                rocblas_side_left,
                uplo,
                transA,
                diag,
                m,
                1,
                pointer_mode == rocblas_pointer_mode_host ? &alpha_h : (T*)alpha_d,
                A,
                lda,
                (T*)dx_mod.get(),
                m);

            strided_vector_copy<T, false>(
                handle->rocblas_stream, incx < 0 ? x + offest : x, m, (T*)dx_mod.get(), incx);
        }
    }

    return status;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocblas_status rocblas_strsv(rocblas_handle handle,
                                        rocblas_fill uplo,
                                        rocblas_operation transA,
                                        rocblas_diagonal diag,
                                        rocblas_int m,
                                        const float* A,
                                        rocblas_int lda,
                                        float* x,
                                        rocblas_int incx)
{
    return rocblas_trsv_template<float, STRSV_BLOCK>(
        handle, uplo, transA, diag, m, A, lda, x, incx);
}

extern "C" rocblas_status rocblas_dtrsv(rocblas_handle handle,
                                        rocblas_fill uplo,
                                        rocblas_operation transA,
                                        rocblas_diagonal diag,
                                        rocblas_int m,
                                        const double* A,
                                        rocblas_int lda,
                                        double* x,
                                        rocblas_int incx)
{
    return rocblas_trsv_template<double, DTRSV_BLOCK>(
        handle, uplo, transA, diag, m, A, lda, x, incx);
}
