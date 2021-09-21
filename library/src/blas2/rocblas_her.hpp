/* ************************************************************************
 * Copyright 2016-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "check_numerics_vector.hpp"
#include "handle.hpp"

template <rocblas_int DIM_X, typename T_lda, typename T, typename U>
ROCBLAS_KERNEL_ILF void her_kernel_calc(bool        upper,
                                        rocblas_int n,
                                        U           alpha,
                                        const T* __restrict__ x,
                                        rocblas_int incx,
                                        T* __restrict__ A,
                                        T_lda lda)
{
    rocblas_int tx  = hipThreadIdx_x;
    rocblas_int col = hipBlockIdx_x;

    if(tx < n)
        A += tx;

    //Each BlockIdx.x takes care of each column of matrix A
    A += col * lda;

    const T res_x = conj(x[col * incx]) * alpha;

    if(upper)
    {
        //scalar-vector-vector product and add the result to a Hermitian matrix 'A'.
        //If n > DIM_X, then the threads are reused and the multiplied values will be accumalated with matrix A.
        rocblas_int i = 0;
        for(; tx + i < col; i += DIM_X)
        {
            A[i] += res_x * x[(tx + i) * incx];
        }
        //Diagonal elements must be real
        if(tx + i == col)
        {
            A[i] = std::real(A[i]) + std::real(x[col * incx] * res_x);
        }
    }
    else
    {
        rocblas_int i = col + 1;
        //Diagonal elements must be real
        if(tx == 0)
        {
            A[i - 1] = std::real(A[i - 1]) + std::real(x[col * incx] * res_x);
        }
        //scalar-vector-vector product and add the result to a Hermitian matrix 'A'.
        //If n > DIM_X, then the threads are reused and the multiplied values will be accumalated with matrix A.
        for(; tx + i < n; i += DIM_X)
        {
            A[i] += res_x * x[(tx + i) * incx];
        }
    }
}

template <rocblas_int DIM_X, typename T_lda, typename TScal, typename TConstPtr, typename TPtr>
ROCBLAS_KERNEL __launch_bounds__(DIM_X) void rocblas_her_kernel(bool           upper,
                                                                rocblas_int    n,
                                                                TScal          alpha_device_host,
                                                                TConstPtr      xa,
                                                                ptrdiff_t      shift_x,
                                                                rocblas_int    incx,
                                                                rocblas_stride stride_x,
                                                                TPtr           Aa,
                                                                rocblas_int    lda,
                                                                ptrdiff_t      shift_A,
                                                                rocblas_stride stride_A)
{
    auto alpha = load_scalar(alpha_device_host);
    if(!alpha)
        return;

    auto*       A = load_ptr_batch(Aa, hipBlockIdx_y, shift_A, stride_A);
    const auto* x = load_ptr_batch(xa, hipBlockIdx_y, shift_x, stride_x);

    her_kernel_calc<DIM_X, T_lda>(upper, n, alpha, x, incx, A, lda);
}

/**
 * TScal     is always: const U* (either host or device)
 * TConstPtr is either: const T* OR const T* const*
 * TPtr      is either:       T* OR       T* const*
 * Where T is the base type (rocblas_float_complex or rocblas_double_complex)
 * and U is the scalar type (float or double)
 */
template <typename TScal, typename TConstPtr, typename TPtr>
rocblas_status rocblas_her_template(rocblas_handle handle,
                                    rocblas_fill   uplo,
                                    rocblas_int    n,
                                    TScal          alpha,
                                    TConstPtr      x,
                                    rocblas_int    offset_x,
                                    rocblas_int    incx,
                                    rocblas_stride stride_x,
                                    TPtr           A,
                                    rocblas_int    lda,
                                    rocblas_int    offset_A,
                                    rocblas_stride stride_A,
                                    rocblas_int    batch_count)
{
    // Quick return if possible. Not Argument error
    if(!n || !batch_count)
        return rocblas_status_success;

    hipStream_t rocblas_stream = handle->get_stream();

    // in case of negative inc, shift pointer to end of data for negative indexing tid*inc
    ptrdiff_t shift_x = incx < 0 ? offset_x - ptrdiff_t(incx) * (n - 1) : offset_x;

    bool i64_indices = n * size_t(lda) > std::numeric_limits<rocblas_int>::max();

#define her_KARGS(alpha_)                                                                        \
    her_grid, her_threads, 0, rocblas_stream, uplo == rocblas_fill_upper, n, alpha_, x, shift_x, \
        incx, stride_x, A, lda, offset_A, stride_A

    static constexpr int HER_DIM_X = 1024;

    dim3 her_grid(n, batch_count);
    dim3 her_threads(HER_DIM_X);

    if(rocblas_pointer_mode_device == handle->pointer_mode)
    {
        if(i64_indices)
            hipLaunchKernelGGL((rocblas_her_kernel<HER_DIM_X, size_t>), her_KARGS(alpha));
        else
            hipLaunchKernelGGL((rocblas_her_kernel<HER_DIM_X, rocblas_int>), her_KARGS(alpha));
    }
    else
    {
        if(i64_indices)
            hipLaunchKernelGGL((rocblas_her_kernel<HER_DIM_X, size_t>), her_KARGS(*alpha));
        else
            hipLaunchKernelGGL((rocblas_her_kernel<HER_DIM_X, rocblas_int>), her_KARGS(*alpha));
    }
#undef her_KARGS
    return rocblas_status_success;
}

//TODO :-Add rocblas_check_numerics_he_matrix_template for checking Matrix `A` which is a Hermitian Matrix
template <typename T, typename U>
rocblas_status rocblas_her_check_numerics(const char*    function_name,
                                          rocblas_handle handle,
                                          rocblas_int    n,
                                          T              A,
                                          rocblas_int    offset_a,
                                          rocblas_int    lda,
                                          rocblas_stride stride_a,
                                          U              x,
                                          rocblas_int    offset_x,
                                          rocblas_int    inc_x,
                                          rocblas_stride stride_x,
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

    return check_numerics_status;
}
