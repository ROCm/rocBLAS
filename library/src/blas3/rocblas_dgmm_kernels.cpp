/* ************************************************************************
 * Copyright 2016-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "handle.hpp"
#include "rocblas_dgmm.hpp"

template <int DIM_X, int DIM_Y, bool side_right, typename TConstPtr, typename TPtr>
ROCBLAS_KERNEL(DIM_X* DIM_Y)
dgmm_device(rocblas_int    m,
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
    rocblas_int tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    rocblas_int ty = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if(tx < m && ty < n)
    {
        auto* A = load_ptr_batch(Aa, hipBlockIdx_z, offset_a, stride_a);
        auto* X = load_ptr_batch(Xa, hipBlockIdx_z, shift_x, stride_x);
        auto* C = load_ptr_batch(Ca, hipBlockIdx_z, offset_c, stride_c);

        if(side_right)
        {
            C[tx + size_t(ldc) * ty] = A[tx + size_t(lda) * ty] * X[ty * incx];
        }
        else
        {
            C[tx + size_t(ldc) * ty] = A[tx + size_t(lda) * ty] * X[tx * incx];
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
            hipLaunchKernelGGL((dgmm_device<DGMM_DIM_X, DGMM_DIM_Y, false>),
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
            hipLaunchKernelGGL((dgmm_device<DGMM_DIM_X, DGMM_DIM_Y, true>),
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
// clang-format on
