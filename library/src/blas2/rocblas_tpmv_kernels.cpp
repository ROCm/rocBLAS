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

#include "../blas1/rocblas_copy.hpp"
#include "check_numerics_vector.hpp"
#include "rocblas_tpmv.hpp"

#include "utility.hpp"

#ifdef tmpv_calc_upperat
#error tmpv_calc_upperat is already defined
#endif
#define tmpv_calc_upperat(_i, _j) ((size_t(_j) * ((_j) + 1)) / 2 + (_i))

#ifdef tmpv_calc_lowerat
#error tmpv_calc_lowerat is already defined
#endif
#define tmpv_calc_lowerat(_i, _j) (size_t(_j) * n + ((_i) - (_j)) - ((size_t(_j) - 1) * (_j)) / 2)

template <rocblas_int NB, typename T>
ROCBLAS_KERNEL_ILF void rocblas_tpmvn_kernel_calc(bool        is_upper,
                                                  bool        is_unit_diag,
                                                  rocblas_int n,
                                                  const T*    A,
                                                  const T*    x,
                                                  rocblas_int incx,
                                                  T*          workspace)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < n)
    {
        T res = x[tid * int64_t(incx)];
        if(is_upper)
        {
            if(!is_unit_diag)
            {
                res *= A[tmpv_calc_upperat(tid, tid)];
            }
            for(rocblas_int col = tid + 1; col < n; ++col)
            {
                res += A[tmpv_calc_upperat(tid, col)] * x[col * int64_t(incx)];
            }
        }
        else
        {
            if(!is_unit_diag)
            {
                //cppcheck-suppress duplicateExpression
                res *= A[tmpv_calc_lowerat(tid, tid)];
            }
            for(rocblas_int col = 0; col < tid; ++col)
            {
                res += A[tmpv_calc_lowerat(tid, col)] * x[col * int64_t(incx)];
            }
        }

        workspace[tid] = res;
    }
}

template <rocblas_int NB, typename T>
ROCBLAS_KERNEL_ILF void rocblas_tpmvc_kernel_calc(bool        is_upper,
                                                  bool        is_unit_diag,
                                                  rocblas_int n,
                                                  const T*    A,
                                                  const T*    x,
                                                  rocblas_int incx,
                                                  T*          workspace)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < n)
    {
        T res = x[tid * int64_t(incx)];
        if(is_upper)
        {
            if(!is_unit_diag)
            {
                res *= conj(A[tmpv_calc_upperat(tid, tid)]);
            }
            for(rocblas_int row = 0; row < tid; ++row)
            {
                res += conj(A[tmpv_calc_upperat(row, tid)]) * x[row * int64_t(incx)];
            }
        }
        else
        {
            if(!is_unit_diag)
            {
                //cppcheck-suppress duplicateExpression
                res *= conj(A[tmpv_calc_lowerat(tid, tid)]);
            }
            for(rocblas_int row = tid + 1; row < n; ++row)
            {
                res += conj(A[tmpv_calc_lowerat(row, tid)]) * x[row * int64_t(incx)];
            }
        }
        workspace[tid] = res;
    }
}

template <rocblas_int NB, typename T>
ROCBLAS_KERNEL_ILF void rocblas_tpmvt_kernel_calc(bool        is_upper,
                                                  bool        is_unit_diag,
                                                  rocblas_int n,
                                                  const T*    A,
                                                  const T*    x,
                                                  rocblas_int incx,
                                                  T*          workspace)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    T           res;
    rocblas_int row;

    if(tid < n)
    {
        res = x[tid * int64_t(incx)];
        if(is_upper)
        {
            if(!is_unit_diag)
            {
                res *= A[tmpv_calc_upperat(tid, tid)];
            }

            for(row = 0; row < tid; ++row)
            {
                res += A[tmpv_calc_upperat(row, tid)] * x[row * int64_t(incx)];
            }
        }
        else
        {
            if(!is_unit_diag)
            {
                //cppcheck-suppress duplicateExpression
                res *= A[tmpv_calc_lowerat(tid, tid)];
            }

            for(row = tid + 1; row < n; ++row)
            {
                res += A[tmpv_calc_lowerat(row, tid)] * x[row * int64_t(incx)];
            }
        }
        workspace[tid] = res;
    }
}

template <rocblas_int NB, typename A, typename X, typename W>
ROCBLAS_KERNEL(NB)
rocblas_tpmvn_kernel(bool           is_upper,
                     bool           is_unit_diag,
                     rocblas_int    n,
                     A              a,
                     rocblas_stride shifta,
                     rocblas_stride stridea,
                     X              x,
                     rocblas_stride shiftx,
                     rocblas_int    incx,
                     rocblas_stride stridex,
                     W              workspace,
                     rocblas_stride stridew)
{
    static constexpr ptrdiff_t shiftw = 0;
    rocblas_tpmvn_kernel_calc<NB>(is_upper,
                                  is_unit_diag,
                                  n,
                                  load_ptr_batch(a, blockIdx.y, shifta, stridea),
                                  load_ptr_batch(x, blockIdx.y, shiftx, stridex),
                                  incx,
                                  load_ptr_batch(workspace, blockIdx.y, shiftw, stridew));
}

template <rocblas_int NB, typename A, typename X, typename W>
ROCBLAS_KERNEL(NB)
rocblas_tpmvt_kernel(bool           is_upper,
                     bool           is_unit_diag,
                     rocblas_int    n,
                     A              a,
                     rocblas_stride shifta,
                     rocblas_stride stridea,
                     X              x,
                     rocblas_stride shiftx,
                     rocblas_int    incx,
                     rocblas_stride stridex,
                     W              workspace,
                     rocblas_stride stridew)
{
    static constexpr ptrdiff_t shiftw = 0;
    rocblas_tpmvt_kernel_calc<NB>(is_upper,
                                  is_unit_diag,
                                  n,
                                  load_ptr_batch(a, blockIdx.y, shifta, stridea),
                                  load_ptr_batch(x, blockIdx.y, shiftx, stridex),
                                  incx,
                                  load_ptr_batch(workspace, blockIdx.y, shiftw, stridew));
}

template <rocblas_int NB, typename A, typename X, typename W>
ROCBLAS_KERNEL(NB)
rocblas_tpmvc_kernel(bool           is_upper,
                     bool           is_unit_diag,
                     rocblas_int    n,
                     A              a,
                     rocblas_stride shifta,
                     rocblas_stride stridea,
                     X              x,
                     rocblas_stride shiftx,
                     rocblas_int    incx,
                     rocblas_stride stridex,
                     W              workspace,
                     rocblas_stride stridew)
{
    static constexpr ptrdiff_t shiftw = 0;
    rocblas_tpmvc_kernel_calc<NB>(is_upper,
                                  is_unit_diag,
                                  n,
                                  load_ptr_batch(a, blockIdx.y, shifta, stridea),
                                  load_ptr_batch(x, blockIdx.y, shiftx, stridex),
                                  incx,
                                  load_ptr_batch(workspace, blockIdx.y, shiftw, stridew));
}

#undef tmpv_calc_upperat
#undef tmpv_calc_lowerat

template <rocblas_int NB, typename A, typename X, typename W>
rocblas_status rocblas_tpmv_template(rocblas_handle    handle,
                                     rocblas_fill      uplo,
                                     rocblas_operation transa,
                                     rocblas_diagonal  diag,
                                     rocblas_int       n,
                                     A                 a,
                                     rocblas_stride    offseta,
                                     rocblas_stride    stridea,
                                     X                 x,
                                     rocblas_stride    offsetx,
                                     rocblas_int       incx,
                                     rocblas_stride    stridex,
                                     W                 workspace,
                                     rocblas_stride    stridew,
                                     rocblas_int       batch_count)
{
    //
    // quick return
    //
    if(!n || !batch_count)
    {
        return rocblas_status_success;
    }

    hipStream_t rocblas_stream = handle->get_stream();

    int64_t shiftx = incx < 0 ? offsetx + int64_t(incx) * (1 - n) : offsetx;

    dim3 tpmv_grid((n - 1) / NB + 1, batch_count);
    dim3 tpmv_threads(NB);

    switch(transa)
    {
    case rocblas_operation_none:
    {
        ROCBLAS_LAUNCH_KERNEL(rocblas_tpmvn_kernel<NB>,
                              tpmv_grid,
                              tpmv_threads,
                              0,
                              rocblas_stream,
                              uplo == rocblas_fill_upper,
                              diag == rocblas_diagonal_unit,
                              n,
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
        ROCBLAS_LAUNCH_KERNEL(rocblas_tpmvt_kernel<NB>,
                              tpmv_grid,
                              tpmv_threads,
                              0,
                              rocblas_stream,
                              uplo == rocblas_fill_upper,
                              diag == rocblas_diagonal_unit,
                              n,
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
        ROCBLAS_LAUNCH_KERNEL(rocblas_tpmvc_kernel<NB>,
                              tpmv_grid,
                              tpmv_threads,
                              0,
                              rocblas_stream,
                              uplo == rocblas_fill_upper,
                              diag == rocblas_diagonal_unit,
                              n,
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
        return rocblas_copy_template<NB>(
            handle, n, workspace, offsetw, incw, stridew, x, offsetx, incx, stridex, batch_count);
    }
}

//TODO :-Add rocblas_check_numerics_tp_matrix_template for checking Matrix `A` which is a Triangular Packed Matrix
template <typename T, typename U>
rocblas_status rocblas_tpmv_check_numerics(const char*    function_name,
                                           rocblas_handle handle,
                                           rocblas_int    n,
                                           T              A,
                                           rocblas_stride offset_a,
                                           rocblas_stride stride_a,
                                           U              x,
                                           rocblas_stride offset_x,
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
                                           rocblas_int    n,              \
                                           T_             A,              \
                                           rocblas_stride    offset_a,       \
                                           rocblas_stride stride_a,       \
                                           U_             x,              \
                                           rocblas_stride    offset_x,       \
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
                             rocblas_int       n,            \
                             A_                a,            \
                             rocblas_stride    offseta,      \
                             rocblas_stride    stridea,      \
                             X_                x,            \
                             rocblas_stride    offsetx,      \
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
