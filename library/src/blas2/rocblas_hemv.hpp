/* ************************************************************************
 * Copyright 2019-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "check_numerics_vector.hpp"
#include "handle.hpp"

/**
  *  Computes y := alpha*A*x + beta*y where A is a Hermitian matrix.
  *  If uplo == upper, the strictly lower part of A is not referenced,
  *  if uplo == lower, the strictly upper part of A is not referenced.
  *  The imaginary part of the main diagonal is assumed to always be == 0.
  */
template <rocblas_int DIM_X, rocblas_int DIM_Y, typename T>
__device__ void hemvn_kernel_calc(rocblas_fill uplo,
                                  rocblas_int  n,
                                  T            alpha,
                                  const T*     A,
                                  ptrdiff_t    lda,
                                  const T*     x,
                                  ptrdiff_t    incx,
                                  T            beta,
                                  T*           y,
                                  ptrdiff_t    incy)
{
    rocblas_int thread_id = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x;

    if(!alpha)
    {
        if(thread_id < DIM_X)
        {
            rocblas_int ind = hipBlockIdx_x * DIM_X + thread_id;
            if(ind < n)
                y[ind * incy] = beta ? beta * y[ind * incy] : 0;
        }
        return;
    }

    bool upper = uplo == rocblas_fill_upper;

    // threads are all configurated locally
    rocblas_int tx = thread_id % DIM_X;
    rocblas_int ty = thread_id / DIM_X;

    __shared__ T sdata[DIM_X * DIM_Y];

    T res_A;
    T tmp_A;

    res_A = tmp_A   = T(0);
    rocblas_int col = ty;

    for(col = ty; col < n; col += DIM_Y)
    {
        rocblas_int ind = hipBlockIdx_x * DIM_X + tx;
        if(ind < n)
        {
            if(col > ind)
                tmp_A = upper ? A[ind + col * lda] : conj(A[col + ind * lda]);
            else if(col < ind)
                tmp_A = !upper ? A[ind + col * lda] : conj(A[col + ind * lda]);
            else if(col == ind)
                tmp_A = std::real(A[ind + col * lda]);

            res_A += tmp_A * x[(col)*incx];
        }
    }

    sdata[tx + ty * DIM_X] = res_A;
    __syncthreads();

    if(thread_id < DIM_X)
    {
        for(rocblas_int i = 1; i < DIM_Y; i++)
            sdata[thread_id] += sdata[thread_id + DIM_X * i];

        rocblas_int ind = hipBlockIdx_x * DIM_X + thread_id;
        if(ind < n)
            y[ind * incy]
                = beta ? alpha * sdata[thread_id] + beta * y[ind * incy] : alpha * sdata[thread_id];
    }
}

/**
  *  U is either: const T* OR T
  *  V is either: const T* OR const T* const*
  *  W is either:       T* OR       T* const*
  */
template <rocblas_int DIM_X, rocblas_int DIM_Y, typename U, typename V, typename W>
__launch_bounds__(DIM_X* DIM_Y) __global__ void hemvn_kernel(rocblas_fill   uplo,
                                                             rocblas_int    n,
                                                             U              alpha_device_host,
                                                             rocblas_stride stride_alpha,
                                                             V              Aa,
                                                             ptrdiff_t      shifta,
                                                             rocblas_int    lda,
                                                             rocblas_stride strideA,
                                                             V              xa,
                                                             ptrdiff_t      shiftx,
                                                             rocblas_int    incx,
                                                             rocblas_stride stridex,
                                                             U              beta_device_host,
                                                             rocblas_stride stride_beta,
                                                             W              ya,
                                                             ptrdiff_t      shifty,
                                                             rocblas_int    incy,
                                                             rocblas_stride stridey)
{
    rocblas_int num_threads = hipBlockDim_x * hipBlockDim_y * hipBlockDim_z;
    if(DIM_X * DIM_Y != num_threads)
        return; // need to launch exactly the same number of threads as template parameters indicate

    auto alpha = load_scalar(alpha_device_host, hipBlockIdx_y, stride_alpha);
    auto beta  = load_scalar(beta_device_host, hipBlockIdx_y, stride_beta);

    if(!alpha && beta == 1)
        return;

    const auto* A = cond_load_ptr_batch(alpha, Aa, hipBlockIdx_y, shifta, strideA);
    const auto* x = cond_load_ptr_batch(alpha, xa, hipBlockIdx_y, shiftx, stridex);

    auto* y = load_ptr_batch(ya, hipBlockIdx_y, shifty, stridey);

    hemvn_kernel_calc<DIM_X, DIM_Y>(uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}

/**
  *  V is either: const T* OR const T* const*
  *  W is either:       T* OR       T* const*
  *  Note stride_alpha and stride_beta are only used AND only tested by rocSOLVER
  *  These strided scalar fetches are only supported for device_ptr mode
  */
template <typename U, typename V, typename W>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_hemv_template(rocblas_handle handle,
                                   rocblas_fill   uplo,
                                   rocblas_int    n,
                                   const U*       alpha,
                                   rocblas_stride stride_alpha,
                                   V              A,
                                   rocblas_int    offseta,
                                   rocblas_int    lda,
                                   rocblas_stride strideA,
                                   V              x,
                                   rocblas_int    offsetx,
                                   rocblas_int    incx,
                                   rocblas_stride stridex,
                                   const U*       beta,
                                   rocblas_stride stride_beta,
                                   W              y,
                                   rocblas_int    offsety,
                                   rocblas_int    incy,
                                   rocblas_stride stridey,
                                   rocblas_int    batch_count)
{
    //quick return
    if(!n || !batch_count)
        return rocblas_status_success;

    hipStream_t rocblas_stream = handle->get_stream();

    // in case of negative inc shift pointer to end of data for negative indexing tid*inc
    auto shiftx = incx < 0 ? offsetx - ptrdiff_t(incx) * (n - 1) : offsetx;
    auto shifty = incy < 0 ? offsety - ptrdiff_t(incy) * (n - 1) : offsety;

    // HEMVN_DIM_Y must be at least 4, 8 * 8 is very slow only 40Gflop/s
    static constexpr int HEMVN_DIM_X = 64;
    static constexpr int HEMVN_DIM_Y = 16;
    rocblas_int          blocks      = (n - 1) / (HEMVN_DIM_X) + 1;
    dim3                 hemvn_grid(blocks, batch_count);
    dim3                 hemvn_threads(HEMVN_DIM_X, HEMVN_DIM_Y);

    if(handle->pointer_mode == rocblas_pointer_mode_device)
    {
        hipLaunchKernelGGL((hemvn_kernel<HEMVN_DIM_X, HEMVN_DIM_Y>),
                           hemvn_grid,
                           hemvn_threads,
                           0,
                           rocblas_stream,
                           uplo,
                           n,
                           alpha,
                           stride_alpha,
                           A,
                           offseta,
                           lda,
                           strideA,
                           x,
                           shiftx,
                           incx,
                           stridex,
                           beta,
                           stride_beta,
                           y,
                           shifty,
                           incy,
                           stridey);
    }
    else
    {
        if(!*alpha && *beta == 1)
            return rocblas_status_success;

        hipLaunchKernelGGL((hemvn_kernel<HEMVN_DIM_X, HEMVN_DIM_Y>),
                           hemvn_grid,
                           hemvn_threads,
                           0,
                           rocblas_stream,
                           uplo,
                           n,
                           *alpha,
                           stride_alpha,
                           A,
                           offseta,
                           lda,
                           strideA,
                           x,
                           shiftx,
                           incx,
                           stridex,
                           *beta,
                           stride_beta,
                           y,
                           shifty,
                           incy,
                           stridey);
    }

    return rocblas_status_success;
}

//TODO :-Add rocblas_check_numerics_he_matrix_template for checking Matrix `A` which is a Hermitian Matrix
template <typename T, typename U>
rocblas_status rocblas_hemv_check_numerics(const char*    function_name,
                                           rocblas_handle handle,
                                           rocblas_int    n,
                                           T              A,
                                           rocblas_int    offset_a,
                                           rocblas_int    lda,
                                           rocblas_stride stride_a,
                                           T              x,
                                           rocblas_int    offset_x,
                                           rocblas_int    inc_x,
                                           rocblas_stride stride_x,
                                           U              y,
                                           rocblas_int    offset_y,
                                           rocblas_int    inc_y,
                                           rocblas_stride stride_y,
                                           rocblas_int    batch_count,
                                           const int      check_numerics,
                                           bool           is_input)
{
    rocblas_status check_numerics_status
        = rocblas_internal_check_numerics_vector_template(function_name,
                                                          handle,
                                                          n,
                                                          x,
                                                          offset_x,
                                                          inc_x,
                                                          stride_x,
                                                          batch_count,
                                                          check_numerics,
                                                          is_input);
    if(check_numerics_status != rocblas_status_success)
        return check_numerics_status;

    check_numerics_status = rocblas_internal_check_numerics_vector_template(function_name,
                                                                            handle,
                                                                            n,
                                                                            y,
                                                                            offset_y,
                                                                            inc_y,
                                                                            stride_y,
                                                                            batch_count,
                                                                            check_numerics,
                                                                            is_input);

    return check_numerics_status;
}
