/* ************************************************************************
 * Copyright 2016-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "../blas1/rocblas_copy.hpp"
#include "check_numerics_vector.hpp"
#include "rocblas_tpmv.hpp"

#include "utility.hpp"

#ifdef tmpv_calc_upperat
#error tmpv_calc_upperat is already defined
#endif
#define tmpv_calc_upperat(_i, _j) (((_j) * ((_j) + 1)) / 2 + (_i))

#ifdef tmpv_calc_lowerat
#error tmpv_calc_lowerat is already defined
#endif
#define tmpv_calc_lowerat(_i, _j) ((_j)*m + ((_i) - (_j)) - (((_j)-1) * (_j)) / 2)

template <rocblas_int NB, typename T>
ROCBLAS_KERNEL_ILF void tpmvn_kernel_calc(rocblas_fill     uplo,
                                          rocblas_diagonal diag,
                                          rocblas_int      m,
                                          const T*         A,
                                          T*               x,
                                          rocblas_int      incx,
                                          T*               workspace)
{
    ptrdiff_t   tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    rocblas_int tx  = hipThreadIdx_x;
    if(tid < m)
    {
        T res = x[tid * incx];
        if(rocblas_fill_upper == uplo)
        {
            if(diag == rocblas_diagonal_non_unit)
            {
                res *= A[tmpv_calc_upperat(tid, tid)];
            }
            for(rocblas_int col = tid + 1; col < m; ++col)
            {
                res += A[tmpv_calc_upperat(tid, col)] * x[col * incx];
            }
        }
        else
        {
            if(diag == rocblas_diagonal_non_unit)
            {
                res *= A[tmpv_calc_lowerat(tid, tid)];
            }
            for(rocblas_int col = 0; col < tid; ++col)
            {
                res += A[tmpv_calc_lowerat(tid, col)] * x[col * incx];
            }
        }

        workspace[tid] = res;
    }
}

template <rocblas_int NB, typename T>
ROCBLAS_KERNEL_ILF void tpmvc_kernel_calc(rocblas_fill     uplo,
                                          rocblas_diagonal diag,
                                          rocblas_int      m,
                                          const T*         A,
                                          T*               x,
                                          rocblas_int      incx,
                                          T*               workspace)
{
    ptrdiff_t   tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    rocblas_int tx  = hipThreadIdx_x;
    if(tid < m)
    {
        T res = x[tid * incx];
        if(rocblas_fill_upper == uplo)
        {
            if(diag == rocblas_diagonal_non_unit)
            {
                res *= conj(A[tmpv_calc_upperat(tid, tid)]);
            }
            for(rocblas_int row = 0; row < tid; ++row)
            {
                res += conj(A[tmpv_calc_upperat(row, tid)]) * x[row * incx];
            }
        }
        else
        {
            if(diag == rocblas_diagonal_non_unit)
            {
                res *= conj(A[tmpv_calc_lowerat(tid, tid)]);
            }
            for(rocblas_int row = tid + 1; row < m; ++row)
            {
                res += conj(A[tmpv_calc_lowerat(row, tid)]) * x[row * incx];
            }
        }
        workspace[tid] = res;
    }
}

template <rocblas_int NB, typename T>
ROCBLAS_KERNEL_ILF void tpmvt_kernel_calc(rocblas_fill     uplo,
                                          rocblas_diagonal diag,
                                          rocblas_int      m,
                                          const T*         A,
                                          T*               x,
                                          rocblas_int      incx,
                                          T*               workspace)
{
    ptrdiff_t   tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    rocblas_int tx  = hipThreadIdx_x;

    T           res;
    rocblas_int row;

    if(tid < m)
    {

        T res = x[tid * incx];
        if(rocblas_fill_upper == uplo)
        {
            if(diag == rocblas_diagonal_non_unit)
            {
                res *= A[tmpv_calc_upperat(tid, tid)];
            }

            for(rocblas_int row = 0; row < tid; ++row)
            {
                res += A[tmpv_calc_upperat(row, tid)] * x[row * incx];
            }
        }
        else
        {
            if(diag == rocblas_diagonal_non_unit)
            {
                res *= A[tmpv_calc_lowerat(tid, tid)];
            }

            for(rocblas_int row = tid + 1; row < m; ++row)
            {
                res += A[tmpv_calc_lowerat(row, tid)] * x[row * incx];
            }
        }
        workspace[tid] = res;
    }
}

template <rocblas_int NB, typename A, typename X, typename W>
ROCBLAS_KERNEL void tpmvn_kernel(rocblas_fill     uplo,
                                 rocblas_diagonal diag,
                                 rocblas_int      m,
                                 A                a,
                                 ptrdiff_t        shifta,
                                 rocblas_stride   stridea,
                                 X                x,
                                 ptrdiff_t        shiftx,
                                 rocblas_int      incx,
                                 rocblas_stride   stridex,
                                 W                workspace,
                                 rocblas_stride   stridew)
{
    static constexpr ptrdiff_t shiftw = 0;
    tpmvn_kernel_calc<NB>(uplo,
                          diag,
                          m,
                          load_ptr_batch(a, hipBlockIdx_y, shifta, stridea),
                          load_ptr_batch(x, hipBlockIdx_y, shiftx, stridex),
                          incx,
                          load_ptr_batch(workspace, hipBlockIdx_y, shiftw, stridew));
}

template <rocblas_int NB, typename A, typename X, typename W>
ROCBLAS_KERNEL void tpmvt_kernel(rocblas_fill     uplo,
                                 rocblas_diagonal diag,
                                 rocblas_int      m,
                                 A                a,
                                 ptrdiff_t        shifta,
                                 rocblas_stride   stridea,
                                 X                x,
                                 ptrdiff_t        shiftx,
                                 rocblas_int      incx,
                                 rocblas_stride   stridex,
                                 W                workspace,
                                 rocblas_stride   stridew)
{
    static constexpr ptrdiff_t shiftw = 0;
    tpmvt_kernel_calc<NB>(uplo,
                          diag,
                          m,
                          load_ptr_batch(a, hipBlockIdx_y, shifta, stridea),
                          load_ptr_batch(x, hipBlockIdx_y, shiftx, stridex),
                          incx,
                          load_ptr_batch(workspace, hipBlockIdx_y, shiftw, stridew));
}

template <rocblas_int NB, typename A, typename X, typename W>
ROCBLAS_KERNEL void tpmvc_kernel(rocblas_fill     uplo,
                                 rocblas_diagonal diag,
                                 rocblas_int      m,
                                 A                a,
                                 ptrdiff_t        shifta,
                                 rocblas_stride   stridea,
                                 X                x,
                                 ptrdiff_t        shiftx,
                                 rocblas_int      incx,
                                 rocblas_stride   stridex,
                                 W                workspace,
                                 rocblas_stride   stridew)
{
    static constexpr ptrdiff_t shiftw = 0;
    tpmvc_kernel_calc<NB>(uplo,
                          diag,
                          m,
                          load_ptr_batch(a, hipBlockIdx_y, shifta, stridea),
                          load_ptr_batch(x, hipBlockIdx_y, shiftx, stridex),
                          incx,
                          load_ptr_batch(workspace, hipBlockIdx_y, shiftw, stridew));
}

#undef tmpv_calc_upperat
#undef tmpv_calc_lowerat

template <rocblas_int NB, typename A, typename X, typename W>
rocblas_status rocblas_tpmv_template(rocblas_handle    handle,
                                     rocblas_fill      uplo,
                                     rocblas_operation transa,
                                     rocblas_diagonal  diag,
                                     rocblas_int       m,
                                     A                 a,
                                     ptrdiff_t         offseta,
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

    dim3 tpmv_grid((m - 1) / NB + 1, batch_count);
    dim3 tpmv_threads(NB);

    switch(transa)
    {
    case rocblas_operation_none:
    {
        hipLaunchKernelGGL(tpmvn_kernel<NB>,
                           tpmv_grid,
                           tpmv_threads,
                           0,
                           rocblas_stream,
                           uplo,
                           diag,
                           m,
                           a,
                           offseta,
                           stridea,
                           x,
                           shiftx,
                           incx,
                           stridex,
                           workspace,
                           stridew);
        break;
    }

    case rocblas_operation_transpose:
    {
        hipLaunchKernelGGL(tpmvt_kernel<NB>,
                           tpmv_grid,
                           tpmv_threads,
                           0,
                           rocblas_stream,
                           uplo,
                           diag,
                           m,
                           a,
                           offseta,
                           stridea,
                           x,
                           shiftx,
                           incx,
                           stridex,
                           workspace,
                           stridew);
        break;
    }

    case rocblas_operation_conjugate_transpose:
    {
        hipLaunchKernelGGL(tpmvc_kernel<NB>,
                           tpmv_grid,
                           tpmv_threads,
                           0,
                           rocblas_stream,
                           uplo,
                           diag,
                           m,
                           a,
                           offseta,
                           stridea,
                           x,
                           shiftx,
                           incx,
                           stridex,
                           workspace,
                           stridew);

        break;
    }
    }

    //
    // Copy workspace to x.
    //
    {
        static constexpr rocblas_int offsetw = 0;
        static constexpr rocblas_int incw    = 1;
        return rocblas_copy_template<false, NB>(
            handle, m, workspace, offsetw, incw, stridew, x, offsetx, incx, stridex, batch_count);
    }
}

//TODO :-Add rocblas_check_numerics_tp_matrix_template for checking Matrix `A` which is a Triangular Packed Matrix
template <typename T, typename U>
rocblas_status rocblas_tpmv_check_numerics(const char*    function_name,
                                           rocblas_handle handle,
                                           rocblas_int    m,
                                           T              A,
                                           rocblas_int    offset_a,
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
// template parameters in the files *tpmv*.cpp

// clang-format off

#ifdef INSTANTIATE_TPMV_NUMERICS
#error INSTANTIATE_TPMV_NUMERICS  already defined
#endif

#define INSTANTIATE_TPMV_NUMERICS(T_, U_)                                 \
template rocblas_status rocblas_tpmv_check_numerics<T_, U_>               \
                                          (const char*    function_name,  \
                                           rocblas_handle handle,         \
                                           rocblas_int    m,              \
                                           T_             A,              \
                                           rocblas_int    offset_a,       \
                                           rocblas_stride stride_a,       \
                                           U_             x,              \
                                           rocblas_int    offset_x,       \
                                           rocblas_int    inc_x,          \
                                           rocblas_stride stride_x,       \
                                           rocblas_int    batch_count,    \
                                           const int      check_numerics, \
                                           bool           is_input);

INSTANTIATE_TPMV_NUMERICS(float const*, float*)
INSTANTIATE_TPMV_NUMERICS(double const*, double*)
INSTANTIATE_TPMV_NUMERICS(rocblas_float_complex const*, rocblas_float_complex*)
INSTANTIATE_TPMV_NUMERICS(rocblas_double_complex const*, rocblas_double_complex*)
INSTANTIATE_TPMV_NUMERICS(float const* const*, float* const*)
INSTANTIATE_TPMV_NUMERICS(double const* const*, double* const*)
INSTANTIATE_TPMV_NUMERICS(rocblas_float_complex const* const*, rocblas_float_complex* const*)
INSTANTIATE_TPMV_NUMERICS(rocblas_double_complex const* const*, rocblas_double_complex* const*)

#undef INSTANTIATE_TPMV_NUMERICS


#ifdef INSTANTIATE_TPMV_TEMPLATE
#error INSTANTIATE_TPMV_TEMPLATE  already defined
#endif

#define INSTANTIATE_TPMV_TEMPLATE(NB_, A_, X_, W_)           \
template rocblas_status rocblas_tpmv_template                \
                            <NB_, A_, X_, W_>                \
                            (rocblas_handle    handle,       \
                             rocblas_fill      uplo,         \
                             rocblas_operation transa,       \
                             rocblas_diagonal  diag,         \
                             rocblas_int       m,            \
                             A_                a,            \
                             ptrdiff_t         offseta,      \
                             rocblas_stride    stridea,      \
                             X_                x,            \
                             ptrdiff_t         offsetx,      \
                             rocblas_int       incx,         \
                             rocblas_stride    stridex,      \
                             W_                workspace,    \
                             rocblas_stride    stridew,      \
                             rocblas_int       batch_count);

INSTANTIATE_TPMV_TEMPLATE(512, float const*, float*, float*)
INSTANTIATE_TPMV_TEMPLATE(512, double const*, double*, double*)
INSTANTIATE_TPMV_TEMPLATE(512, rocblas_float_complex const*, rocblas_float_complex*, rocblas_float_complex*)
INSTANTIATE_TPMV_TEMPLATE(512, rocblas_double_complex const*, rocblas_double_complex*, rocblas_double_complex*)
INSTANTIATE_TPMV_TEMPLATE(512, float const* const*, float* const*, float*)
INSTANTIATE_TPMV_TEMPLATE(512, double const* const*, double* const*, double*)
INSTANTIATE_TPMV_TEMPLATE(512, rocblas_float_complex const* const*, rocblas_float_complex* const*, rocblas_float_complex*)
INSTANTIATE_TPMV_TEMPLATE(512, rocblas_double_complex const* const*, rocblas_double_complex* const*, rocblas_double_complex*)

#undef INSTANTIATE_TPMV_TEMPLATE

// clang-format on
