/* ************************************************************************
 * Copyright 2019-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "../blas1/rocblas_copy.hpp"
#include "../blas1/rocblas_dot.hpp"
#include "rocblas/rocblas.h"
#include "rocblas_trmv.hpp"
#include <cstddef>

template <rocblas_int DIM_X, rocblas_int DIM_Y, bool LOWER, bool UNIT, typename T>
ROCBLAS_KERNEL_ILF void trmvn_kernel_calc(
    rocblas_int m, const T* A, rocblas_int lda, const T* x, rocblas_int incx, T* workspace)
{
    rocblas_int tid = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x;

    // tx corresponds to row in block, good for memory coalescing
    // ty corresponds to column
    rocblas_int tx = hipThreadIdx_x;
    rocblas_int ty = hipThreadIdx_y;

    rocblas_int row = hipBlockIdx_x * DIM_X + tx;

    __shared__ T sdata[DIM_X * DIM_Y];
    T            res_A = 0;

    // handle diagonal separately
    if(ty == 0 && row < m)
    {
        if(UNIT)
            res_A = x[row * incx];
        else
            res_A = A[row + row * lda] * x[row * incx];
    }

    // multiply and sum across columns
    for(rocblas_int col = ty; col < m; col += DIM_Y)
    {
        if(row < m && ((!LOWER && col > row) || (LOWER && col < row)))
            res_A += A[row + col * lda] * x[col * incx];
    }

    // move partial sum to shared memory to sum further
    sdata[tx + ty * DIM_X] = res_A;

    __syncthreads();

    if(tid < DIM_X)
    {
        // sum DIM_Y elements to get result
        for(rocblas_int i = 1; i < DIM_Y; i++)
            sdata[tid] += sdata[tid + DIM_X * i];

        if(row < m)
            workspace[row] = sdata[tid];
    }
}

template <rocblas_int NB, bool LOWER, bool CONJ, bool UNIT, typename T>
ROCBLAS_KERNEL_ILF void trmvt_kernel_calc(
    rocblas_int m, const T* A, rocblas_int lda, const T* x, rocblas_int incx, T* workspace)
{
    // tx is assigned to rows
    rocblas_int tx  = hipThreadIdx_x;
    rocblas_int col = hipBlockIdx_x;

    if(tx < m)
        A += tx;
    A += col * size_t(lda);

    T res = 0;

    // handle diagonal separately
    if(tx == 0)
    {
        if(UNIT)
            res += x[col * incx];
        else
            res += (CONJ ? conj(A[col]) : A[col]) * x[col * incx];
    }

    for(rocblas_int i = 0; tx + i < m; i += NB)
    {
        if((tx + i > col && LOWER) || (tx + i < col && !LOWER))
            res += (CONJ ? conj(A[i]) : A[i]) * x[(tx + i) * incx];
    }

    res = rocblas_dot_block_reduce<NB>(res);

    if(tx == 0)
    {
        workspace[col] = res;
    }
}

template <rocblas_int DIM_X,
          rocblas_int DIM_Y,
          bool        LOWER,
          bool        UNIT,
          typename A,
          typename X,
          typename W>
ROCBLAS_KERNEL void trmvn_kernel(rocblas_int    m,
                                 A              a,
                                 ptrdiff_t      shifta,
                                 rocblas_int    lda,
                                 rocblas_stride stridea,
                                 X              x,
                                 ptrdiff_t      shiftx,
                                 rocblas_int    incx,
                                 rocblas_stride stridex,
                                 W              workspace,
                                 rocblas_stride stridew)
{
    static constexpr ptrdiff_t shiftw = 0;
    trmvn_kernel_calc<DIM_X, DIM_Y, LOWER, UNIT>(
        m,
        load_ptr_batch(a, hipBlockIdx_y, shifta, stridea),
        lda,
        load_ptr_batch(x, hipBlockIdx_y, shiftx, stridex),
        incx,
        load_ptr_batch(workspace, hipBlockIdx_y, shiftw, stridew));
}

template <rocblas_int NB, bool LOWER, bool CONJ, bool UNIT, typename A, typename X, typename W>
ROCBLAS_KERNEL void trmvt_kernel(rocblas_int    m,
                                 A              a,
                                 ptrdiff_t      shifta,
                                 rocblas_int    lda,
                                 rocblas_stride stridea,
                                 X              x,
                                 ptrdiff_t      shiftx,
                                 rocblas_int    incx,
                                 rocblas_stride stridex,
                                 W              workspace,
                                 rocblas_stride stridew)
{
    static constexpr ptrdiff_t shiftw = 0;
    trmvt_kernel_calc<NB, LOWER, CONJ, UNIT>(
        m,
        load_ptr_batch(a, hipBlockIdx_y, shifta, stridea),
        lda,
        load_ptr_batch(x, hipBlockIdx_y, shiftx, stridex),
        incx,
        load_ptr_batch(workspace, hipBlockIdx_y, shiftw, stridew));
}

template <typename A, typename X, typename W>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_trmv_template(rocblas_handle    handle,
                                   rocblas_fill      uplo,
                                   rocblas_operation transA,
                                   rocblas_diagonal  diag,
                                   rocblas_int       m,
                                   A                 a,
                                   ptrdiff_t         offseta,
                                   rocblas_int       lda,
                                   rocblas_stride    stridea,
                                   X                 x,
                                   ptrdiff_t         offsetx,
                                   rocblas_int       incx,
                                   rocblas_stride    stridex,
                                   W                 workspace,
                                   rocblas_stride    stridew,
                                   rocblas_int       batch_count)
{
    //
    // quick return
    //
    if(!m || !batch_count)
    {
        return rocblas_status_success;
    }

    hipStream_t rocblas_stream = handle->get_stream();

    ptrdiff_t shiftx = incx < 0 ? offsetx + ptrdiff_t(incx) * (1 - m) : offsetx;

    // NOTE: NB is currently hardcoded as 512
    constexpr int TRMVT_NB    = 512;
    constexpr int TRMVN_DIM_X = 64;
    constexpr int TRMVN_DIM_Y = 16;

    dim3 trmvn_grid((m - 1) / TRMVN_DIM_X + 1, batch_count);
    dim3 trmvn_threads(TRMVN_DIM_X, TRMVN_DIM_Y);

    dim3 trmvt_grid(m, batch_count);
    dim3 trmvt_threads(TRMVT_NB);

#define TRMV_TEMPLATE_PARAMS \
    0, rocblas_stream, m, a, offseta, lda, stridea, x, shiftx, incx, stridex, workspace, stridew

    if(uplo == rocblas_fill_upper)
    {
        if(diag == rocblas_diagonal_unit)
        {
            if(transA == rocblas_operation_none)
                hipLaunchKernelGGL((trmvn_kernel<TRMVN_DIM_X, TRMVN_DIM_Y, false, true>),
                                   trmvn_grid,
                                   trmvn_threads,
                                   TRMV_TEMPLATE_PARAMS);
            else if(transA == rocblas_operation_transpose)
                hipLaunchKernelGGL((trmvt_kernel<TRMVT_NB, false, false, true>),
                                   trmvt_grid,
                                   trmvt_threads,
                                   TRMV_TEMPLATE_PARAMS);
            else if(transA == rocblas_operation_conjugate_transpose)
                hipLaunchKernelGGL((trmvt_kernel<TRMVT_NB, false, true, true>),
                                   trmvt_grid,
                                   trmvt_threads,
                                   TRMV_TEMPLATE_PARAMS);
        }
        else
        {
            if(transA == rocblas_operation_none)
                hipLaunchKernelGGL((trmvn_kernel<TRMVN_DIM_X, TRMVN_DIM_Y, false, false>),
                                   trmvn_grid,
                                   trmvn_threads,
                                   TRMV_TEMPLATE_PARAMS);
            else if(transA == rocblas_operation_transpose)
                hipLaunchKernelGGL((trmvt_kernel<TRMVT_NB, false, false, false>),
                                   trmvt_grid,
                                   trmvt_threads,
                                   TRMV_TEMPLATE_PARAMS);
            else if(transA == rocblas_operation_conjugate_transpose)
                hipLaunchKernelGGL((trmvt_kernel<TRMVT_NB, false, true, false>),
                                   trmvt_grid,
                                   trmvt_threads,
                                   TRMV_TEMPLATE_PARAMS);
        }
    }
    else if(uplo == rocblas_fill_lower)
    {
        if(diag == rocblas_diagonal_unit)
        {
            if(transA == rocblas_operation_none)
                hipLaunchKernelGGL((trmvn_kernel<TRMVN_DIM_X, TRMVN_DIM_Y, true, true>),
                                   trmvn_grid,
                                   trmvn_threads,
                                   TRMV_TEMPLATE_PARAMS);
            else if(transA == rocblas_operation_transpose)
                hipLaunchKernelGGL((trmvt_kernel<TRMVT_NB, true, false, true>),
                                   trmvt_grid,
                                   trmvt_threads,
                                   TRMV_TEMPLATE_PARAMS);
            else if(transA == rocblas_operation_conjugate_transpose)
                hipLaunchKernelGGL((trmvt_kernel<TRMVT_NB, true, true, true>),
                                   trmvt_grid,
                                   trmvt_threads,
                                   TRMV_TEMPLATE_PARAMS);
        }
        else
        {
            if(transA == rocblas_operation_none)
                hipLaunchKernelGGL((trmvn_kernel<TRMVN_DIM_X, TRMVN_DIM_Y, true, false>),
                                   trmvn_grid,
                                   trmvn_threads,
                                   TRMV_TEMPLATE_PARAMS);
            else if(transA == rocblas_operation_transpose)
                hipLaunchKernelGGL((trmvt_kernel<TRMVT_NB, true, false, false>),
                                   trmvt_grid,
                                   trmvt_threads,
                                   TRMV_TEMPLATE_PARAMS);
            else if(transA == rocblas_operation_conjugate_transpose)
                hipLaunchKernelGGL((trmvt_kernel<TRMVT_NB, true, true, false>),
                                   trmvt_grid,
                                   trmvt_threads,
                                   TRMV_TEMPLATE_PARAMS);
        }
    }

    //
    // Copy workspace to x.
    //
    {
        static constexpr rocblas_int offsetw = 0;
        static constexpr rocblas_int incw    = 1;
        return rocblas_copy_template<false, TRMVT_NB>(
            handle, m, workspace, offsetw, incw, stridew, x, offsetx, incx, stridex, batch_count);
    }

#undef TRMV_TEMPLATE_PARAMS
}

//TODO :-Add rocblas_check_numerics_tr_matrix_template for checking Matrix `A` which is a Triangular Matrix
template <typename T, typename U>
rocblas_status rocblas_trmv_check_numerics(const char*    function_name,
                                           rocblas_handle handle,
                                           rocblas_int    m,
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
                                                          m,
                                                          x,
                                                          offset_x,
                                                          inc_x,
                                                          stride_x,
                                                          batch_count,
                                                          check_numerics,
                                                          is_input);

    return check_numerics_status;
}

// Instantiations below will need to be manually updated to match any change in
// template parameters in the files *trmv*.cpp

// clang-format off

#ifdef INSTANTIATE_TRMV_TEMPLATE
#error INSTANTIATE_TRMV_TEMPLATE already defined
#endif

#define INSTANTIATE_TRMV_TEMPLATE(A_, X_, W_)                                           \
template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status rocblas_internal_trmv_template \
                                  <A_, X_, W_>                                          \
                                  (rocblas_handle    handle,                            \
                                   rocblas_fill      uplo,                              \
                                   rocblas_operation transA,                            \
                                   rocblas_diagonal  diag,                              \
                                   rocblas_int       m,                                 \
                                   A_                 a,                                \
                                   ptrdiff_t         offseta,                           \
                                   rocblas_int       lda,                               \
                                   rocblas_stride    stridea,                           \
                                   X_                 x,                                \
                                   ptrdiff_t         offsetx,                           \
                                   rocblas_int       incx,                              \
                                   rocblas_stride    stridex,                           \
                                   W_                 workspace,                        \
                                   rocblas_stride    stridew,                           \
                                   rocblas_int       batch_count);

INSTANTIATE_TRMV_TEMPLATE(float const*, float*, float*)
INSTANTIATE_TRMV_TEMPLATE(double const*, double*, double*)
INSTANTIATE_TRMV_TEMPLATE(rocblas_float_complex const*, rocblas_float_complex*, rocblas_float_complex*)
INSTANTIATE_TRMV_TEMPLATE(rocblas_double_complex const*, rocblas_double_complex*, rocblas_double_complex*)
INSTANTIATE_TRMV_TEMPLATE(float const* const*, float* const*, float*)
INSTANTIATE_TRMV_TEMPLATE(double const* const*, double* const*, double*)
INSTANTIATE_TRMV_TEMPLATE(rocblas_float_complex const* const*, rocblas_float_complex* const*, rocblas_float_complex*)
INSTANTIATE_TRMV_TEMPLATE(rocblas_double_complex const* const*, rocblas_double_complex* const*, rocblas_double_complex*)

#undef INSTANTIATE_TRMV_TEMPLATE

#ifdef INSTANTIATE_TRMV_NUMERICS
#error INSTANTIATE_TRMV_NUMERICS already defined
#endif

#define INSTANTIATE_TRMV_NUMERICS(T_, U_)                                 \
template rocblas_status rocblas_trmv_check_numerics <T_, U_>              \
                                          (const char*    function_name,  \
                                           rocblas_handle handle,         \
                                           rocblas_int    m,              \
                                           T_              A,             \
                                           rocblas_int    offset_a,       \
                                           rocblas_int    lda,            \
                                           rocblas_stride stride_a,       \
                                           U_              x,             \
                                           rocblas_int    offset_x,       \
                                           rocblas_int    inc_x,          \
                                           rocblas_stride stride_x,       \
                                           rocblas_int    batch_count,    \
                                           const int      check_numerics, \
                                           bool           is_input);

INSTANTIATE_TRMV_NUMERICS(float const*, float*)
INSTANTIATE_TRMV_NUMERICS(double const*, double*)
INSTANTIATE_TRMV_NUMERICS(rocblas_float_complex const*, rocblas_float_complex*)
INSTANTIATE_TRMV_NUMERICS(rocblas_double_complex const*, rocblas_double_complex*)
INSTANTIATE_TRMV_NUMERICS(float const* const*, float* const*)
INSTANTIATE_TRMV_NUMERICS(double const* const*, double* const*)
INSTANTIATE_TRMV_NUMERICS(rocblas_float_complex const* const*, rocblas_float_complex* const*)
INSTANTIATE_TRMV_NUMERICS(rocblas_double_complex const* const*, rocblas_double_complex* const*)

#undef INSTANTIATE_TRMV_NUMERICS

// clang-format on
