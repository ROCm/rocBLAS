/* ************************************************************************
 * Copyright (C) 2016-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#include "handle.hpp"
#include "rocblas_dgmm.hpp"

template <int DIM_X, int DIM_Y, bool side_right, typename TConstPtr, typename TPtr>
ROCBLAS_KERNEL(DIM_X* DIM_Y)
rocblas_dgmm_device(rocblas_int    m,
                    rocblas_int    n,
                    TConstPtr      Aa,
                    rocblas_stride offset_a,
                    rocblas_int    lda,
                    rocblas_stride stride_a,
                    TConstPtr      Xa,
                    rocblas_int    shift_x,
                    rocblas_int    incx,
                    rocblas_stride stride_x,
                    TPtr           Ca,
                    rocblas_stride offset_c,
                    rocblas_int    ldc,
                    rocblas_stride stride_c)
{
    rocblas_int tx = blockIdx.x * blockDim.x + threadIdx.x;
    rocblas_int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if(tx < m && ty < n)
    {
        auto* A = load_ptr_batch(Aa, blockIdx.z, offset_a, stride_a);
        auto* X = load_ptr_batch(Xa, blockIdx.z, shift_x, stride_x);
        auto* C = load_ptr_batch(Ca, blockIdx.z, offset_c, stride_c);

        if(side_right)
        {
            C[tx + size_t(ldc) * ty] = A[tx + size_t(lda) * ty] * X[ty * int64_t(incx)];
        }
        else
        {
            C[tx + size_t(ldc) * ty] = A[tx + size_t(lda) * ty] * X[tx * int64_t(incx)];
        }
    }
}

/*
 * ===========================================================================
 *    template interface
 *    template specialization
 *    call DGMM C interfaces (see rocblas_dgmm*.cpp in the same dir)
 * ===========================================================================
 */

/**
 * TConstPtr is either: const T* OR const T* const*
 * TPtr      is either:       T* OR       T* const*
 * Where T is the base type (float, double, rocblas_complex, or rocblas_double_complex)
 */

template <typename TConstPtr, typename TPtr>
rocblas_status rocblas_dgmm_template(rocblas_handle handle,
                                     rocblas_side   side,
                                     rocblas_int    m,
                                     rocblas_int    n,
                                     TConstPtr      A,
                                     rocblas_stride offset_a,
                                     rocblas_int    lda,
                                     rocblas_stride stride_a,
                                     TConstPtr      X,
                                     rocblas_stride offset_x,
                                     rocblas_int    incx,
                                     rocblas_stride stride_x,
                                     TPtr           C,
                                     rocblas_stride offset_c,
                                     rocblas_int    ldc,
                                     rocblas_stride stride_c,
                                     rocblas_int    batch_count)

{
    hipStream_t rocblas_stream = handle->get_stream();

    {
        // in case of negative incx shift pointer to end of data for negative indexing
        rocblas_int k       = side == rocblas_side_left ? m : n;
        ptrdiff_t   shift_x = offset_x - ((incx < 0) ? ptrdiff_t(incx) * (k - 1) : 0);

        // general case, any transA, transB, lda, incx, ldc
        static constexpr int DGMM_DIM_X = 16;
        static constexpr int DGMM_DIM_Y = 16;

        rocblas_int blocksX = (m - 1) / DGMM_DIM_X + 1;
        rocblas_int blocksY = (n - 1) / DGMM_DIM_Y + 1;

        dim3 dgmm_grid(blocksX, blocksY, batch_count);
        dim3 dgmm_threads(DGMM_DIM_X, DGMM_DIM_Y);

        if(rocblas_side_left == side)
        {
            ROCBLAS_LAUNCH_KERNEL((rocblas_dgmm_device<DGMM_DIM_X, DGMM_DIM_Y, false>),
                                  dgmm_grid,
                                  dgmm_threads,
                                  0,
                                  rocblas_stream,
                                  m,
                                  n,
                                  A,
                                  offset_a,
                                  lda,
                                  stride_a,
                                  X,
                                  shift_x,
                                  incx,
                                  stride_x,
                                  C,
                                  offset_c,
                                  ldc,
                                  stride_c);
        }
        else
        {
            ROCBLAS_LAUNCH_KERNEL((rocblas_dgmm_device<DGMM_DIM_X, DGMM_DIM_Y, true>),
                                  dgmm_grid,
                                  dgmm_threads,
                                  0,
                                  rocblas_stream,
                                  m,
                                  n,
                                  A,
                                  offset_a,
                                  lda,
                                  stride_a,
                                  X,
                                  shift_x,
                                  incx,
                                  stride_x,
                                  C,
                                  offset_c,
                                  ldc,
                                  stride_c);
        }
    }
    return rocblas_status_success;
}

template <typename TConstPtr, typename TPtr>
rocblas_status rocblas_dgmm_check_numerics(const char*    function_name,
                                           rocblas_handle handle,
                                           rocblas_side   side,
                                           rocblas_int    m,
                                           rocblas_int    n,
                                           TConstPtr      A,
                                           rocblas_int    lda,
                                           rocblas_stride stride_a,
                                           TConstPtr      x,
                                           rocblas_int    incx,
                                           rocblas_stride stride_x,
                                           TPtr           C,
                                           rocblas_int    ldc,
                                           rocblas_stride stride_c,
                                           rocblas_int    batch_count,
                                           const int      check_numerics,
                                           bool           is_input)
{

    rocblas_status check_numerics_status = rocblas_status_success;
    if(is_input)
    {
        rocblas_int dim_x = (side == rocblas_side_left) ? m : n;
        check_numerics_status
            = rocblas_internal_check_numerics_matrix_template(function_name,
                                                              handle,
                                                              rocblas_operation_none,
                                                              rocblas_fill_full,
                                                              rocblas_client_general_matrix,
                                                              m,
                                                              n,
                                                              A,
                                                              0,
                                                              lda,
                                                              stride_a,
                                                              batch_count,
                                                              check_numerics,
                                                              is_input);
        if(check_numerics_status != rocblas_status_success)
            return check_numerics_status;

        check_numerics_status = rocblas_internal_check_numerics_vector_template(function_name,
                                                                                handle,
                                                                                dim_x,
                                                                                x,
                                                                                0,
                                                                                incx,
                                                                                stride_x,
                                                                                batch_count,
                                                                                check_numerics,
                                                                                is_input);
        if(check_numerics_status != rocblas_status_success)
            return check_numerics_status;
    }
    check_numerics_status
        = rocblas_internal_check_numerics_matrix_template(function_name,
                                                          handle,
                                                          rocblas_operation_none,
                                                          rocblas_fill_full,
                                                          rocblas_client_general_matrix,
                                                          m,
                                                          n,
                                                          C,
                                                          0,
                                                          ldc,
                                                          stride_c,
                                                          batch_count,
                                                          check_numerics,
                                                          is_input);

    return check_numerics_status;
}

// Instantiations below will need to be manually updated to match any change in
// template parameters in the files dgmm*.cpp

// clang-format off
#ifdef INSTANTIATE_DGMM_TEMPLATE
#error INSTANTIATE_DGMM_TEMPLATE already defined
#endif

#define INSTANTIATE_DGMM_TEMPLATE(TConstPtr_, TPtr_)              \
template rocblas_status rocblas_dgmm_template<TConstPtr_, TPtr_>  \
                                    (rocblas_handle handle,       \
                                     rocblas_side   side,         \
                                     rocblas_int    m,            \
                                     rocblas_int    n,            \
                                     TConstPtr_     A,            \
                                     rocblas_stride offset_a,     \
                                     rocblas_int    lda,          \
                                     rocblas_stride stride_a,     \
                                     TConstPtr_     X,            \
                                     rocblas_stride offset_x,     \
                                     rocblas_int    incx,         \
                                     rocblas_stride stride_x,     \
                                     TPtr_          C,            \
                                     rocblas_stride offset_c,     \
                                     rocblas_int    ldc,          \
                                     rocblas_stride stride_c,     \
                                     rocblas_int    batch_count);

// instantiate for rocblas_Xdgmm and rocblas_Xdgmm_strided_batched
INSTANTIATE_DGMM_TEMPLATE( float const*,  float*)
INSTANTIATE_DGMM_TEMPLATE(double const*, double*)
INSTANTIATE_DGMM_TEMPLATE( rocblas_float_complex const*,  rocblas_float_complex*)
INSTANTIATE_DGMM_TEMPLATE(rocblas_double_complex const*, rocblas_double_complex*)

// instantiate for rocblas_Xdgmm_batched
INSTANTIATE_DGMM_TEMPLATE( float const* const*,  float* const*)
INSTANTIATE_DGMM_TEMPLATE(double const* const*, double* const*)
INSTANTIATE_DGMM_TEMPLATE( rocblas_float_complex const* const*,  rocblas_float_complex* const*)
INSTANTIATE_DGMM_TEMPLATE(rocblas_double_complex const* const*, rocblas_double_complex* const*)
#undef INSTANTIATE_DGMM_TEMPLATE


#ifdef INSTANTIATE_DGMM_NUMERICS
#error INSTANTIATE_DGMM_NUMERICS already defined
#endif

#define INSTANTIATE_DGMM_NUMERICS(TConstPtr_, TPtr_)                         \
template rocblas_status rocblas_dgmm_check_numerics<TConstPtr_, TPtr_>       \
                                          (const char*       function_name,  \
                                           rocblas_handle    handle,         \
                                           rocblas_side   side,              \
                                           rocblas_int       m,              \
                                           rocblas_int       n,              \
                                           TConstPtr_        A,              \
                                           rocblas_int       lda,            \
                                           rocblas_stride    stride_a,       \
                                           TConstPtr_        x,              \
                                           rocblas_int       inc,            \
                                           rocblas_stride    stride_x,       \
                                           TPtr_             C,              \
                                           rocblas_int       ldc,            \
                                           rocblas_stride    stride_c,       \
                                           rocblas_int       batch_count,    \
                                           const int         check_numerics, \
                                           bool              is_input);

// instantiate for rocblas_Xdgmm and rocblas_Xdgmm_strided_batched
INSTANTIATE_DGMM_NUMERICS(float const*,  float*)
INSTANTIATE_DGMM_NUMERICS(double const*, double*)
INSTANTIATE_DGMM_NUMERICS(rocblas_float_complex const*,  rocblas_float_complex*)
INSTANTIATE_DGMM_NUMERICS(rocblas_double_complex const*, rocblas_double_complex*)

// instantiate for rocblas_Xdgmm_batched
INSTANTIATE_DGMM_NUMERICS(float const* const*, float* const*)
INSTANTIATE_DGMM_NUMERICS(double const* const*, double* const*)
INSTANTIATE_DGMM_NUMERICS(rocblas_float_complex const* const*,  rocblas_float_complex* const*)
INSTANTIATE_DGMM_NUMERICS(rocblas_double_complex const* const*, rocblas_double_complex* const*)

#undef INSTANTIATE_DGMM_NUMERICS
// clang-format on
