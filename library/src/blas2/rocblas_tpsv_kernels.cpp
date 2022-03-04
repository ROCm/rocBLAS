/* ************************************************************************
 * Copyright 2016-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "../blas1/rocblas_copy.hpp"
#include "check_numerics_vector.hpp"
#include "rocblas_tpsv.hpp"

ROCBLAS_KERNEL_ILF inline rocblas_int
    packed_matrix_index(bool upper, bool trans, rocblas_int n, rocblas_int row, rocblas_int col)
{
    return upper ? (trans ? ((row * (row + 1) / 2) + col) : ((col * (col + 1) / 2) + row))
                 : (trans ? (((row * (2 * n - row + 1)) / 2) + (col - row))
                          : (((col * (2 * n - col + 1)) / 2) + (row - col)));
}

// Uses forward substitution to solve Ax = b. Used for a non-transposed lower-triangular matrix
// or a transposed upper-triangular matrix.
template <bool CONJ, rocblas_int BLK_SIZE, typename T>
ROCBLAS_KERNEL_ILF void tpsv_forward_substitution_calc(
    bool diag, bool trans, int n, const T* __restrict__ A, T* __restrict__ x, rocblas_int incx)
{
    __shared__ T xshared[BLK_SIZE];
    int          tx = threadIdx.x;

    // main loop - iterate forward in BLK_SIZE chunks
    for(rocblas_int i = 0; i < n; i += BLK_SIZE)
    {
        // cache x into shared memory
        if(tx + i < n)
            xshared[tx] = x[(tx + i) * incx];

        __syncthreads();

        // iterate through the current block and solve elements
        for(rocblas_int j = 0; j < BLK_SIZE; j++)
        {
            // solve element that can be solved
            if(tx == j && !diag && j + i < n)
            {
                rocblas_int colA   = j + i;
                rocblas_int rowA   = j + i;
                rocblas_int indexA = packed_matrix_index(trans, trans, n, rowA, colA);
                xshared[tx]        = xshared[tx] / (CONJ ? conj(A[indexA]) : A[indexA]);
            }

            __syncthreads();

            // for rest of block, subtract previous solved part
            if(tx > j && j + i < n)
            {
                rocblas_int colA   = j + i;
                rocblas_int rowA   = tx + i;
                rocblas_int indexA = packed_matrix_index(trans, trans, n, rowA, colA);

                // Ensure row is in range, and subtract
                if(rowA < n)
                    xshared[tx] -= (CONJ ? conj(A[indexA]) : A[indexA]) * xshared[j];
            }
        }

        __syncthreads();

        // apply solved diagonal block to the rest of the array
        // 1. Iterate down rows
        for(rocblas_int j = BLK_SIZE + i; j < n; j += BLK_SIZE)
        {
            if(tx + j >= n)
                break;

            // 2. Sum result (across columns) to be subtracted from original value
            T val = 0;
            for(rocblas_int p = 0; p < BLK_SIZE; p++)
            {
                rocblas_int colA   = i + p;
                rocblas_int rowA   = tx + j;
                rocblas_int indexA = packed_matrix_index(trans, trans, n, rowA, colA);

                if(diag && colA == rowA)
                    val += xshared[p];
                else if(colA < n)
                    val += (CONJ ? conj(A[indexA]) : A[indexA]) * xshared[p];
            }

            x[(tx + j) * incx] -= val;
        }

        // store solved part back to global memory
        if(tx + i < n)
            x[(tx + i) * incx] = xshared[tx];

        __syncthreads();
    }
}

// Uses backward substitution to solve Ax = b. Used for a non-transposed upper-triangular matrix
// or a transposed lower-triangular matrix.
template <bool CONJ, rocblas_int BLK_SIZE, typename T>
ROCBLAS_KERNEL_ILF void tpsv_backward_substitution_calc(
    bool diag, bool trans, int n, const T* __restrict__ A, T* __restrict__ x, rocblas_int incx)
{
    __shared__ T xshared[BLK_SIZE];
    int          tx = threadIdx.x;

    // main loop - Start at end of array and iterate backwards in BLK_SIZE chunks
    for(rocblas_int i = n - BLK_SIZE; i > -BLK_SIZE; i -= BLK_SIZE)
    {
        // cache x into shared memory
        if(tx + i >= 0)
            xshared[tx] = x[(tx + i) * incx];

        __syncthreads();

        // Iterate backwards through the current block to solve elements.
        for(rocblas_int j = BLK_SIZE - 1; j >= 0; j--)
        {
            // Solve the new element that can be solved
            if(tx == j && !diag && j + i >= 0)
            {
                rocblas_int colA   = j + i;
                rocblas_int rowA   = j + i;
                rocblas_int indexA = packed_matrix_index(!trans, trans, n, rowA, colA);
                xshared[tx]        = xshared[tx] / (CONJ ? conj(A[indexA]) : A[indexA]);
            }

            __syncthreads();

            // for rest of block, subtract previous solved part
            if(tx < j && j + i >= 0)
            {
                rocblas_int colA   = j + i;
                rocblas_int rowA   = tx + i;
                rocblas_int indexA = packed_matrix_index(!trans, trans, n, rowA, colA);

                // Ensure row is in range, and subtract
                if(rowA >= 0)
                    xshared[tx] -= (CONJ ? conj(A[indexA]) : A[indexA]) * xshared[j];
            }
        }

        __syncthreads();

        // apply solved diagonal block to the rest of the array
        // 1. Iterate up rows, starting at the block above the current block
        for(rocblas_int j = i - BLK_SIZE; j > -BLK_SIZE; j -= BLK_SIZE)
        {
            if(tx + j < 0)
                break;

            // 2. Sum result (across columns) to be subtracted from the original value
            T val = 0;
            for(rocblas_int p = 0; p < BLK_SIZE; p++)
            {
                rocblas_int colA   = i + p;
                rocblas_int rowA   = tx + j;
                rocblas_int indexA = packed_matrix_index(!trans, trans, n, rowA, colA);

                if(diag && colA == rowA)
                    val += xshared[p];
                else
                    val += (CONJ ? conj(A[indexA]) : A[indexA]) * xshared[p];
            }

            x[(tx + j) * incx] -= val;
        }

        // store solved part back to global memory
        if(tx + i >= 0)
            x[(tx + i) * incx] = xshared[tx];

        __syncthreads();
    }
}

/**
     *  Calls forwards/backwards substitution kernels with appropriate arguments.
     *  Note the attribute here - apparently this is needed for group sizes > 256.
     *  Currently BLK_SIZE is set to 512, so this is required or else we get
     *  incorrect behaviour.
     *
     *  To optimize we can probably use multiple kernels (a substitution kernel and a
     *  multiplication kernel) so we can use multiple blocks instead of a single one.
     */
template <bool CONJ, rocblas_int BLK_SIZE, typename TConstPtr, typename TPtr>
ROCBLAS_KERNEL(BLK_SIZE)
rocblas_tpsv_kernel(rocblas_fill      uplo,
                    rocblas_operation transA,
                    rocblas_diagonal  diag,
                    rocblas_int       n,
                    TConstPtr __restrict__ APa,
                    ptrdiff_t      shift_A,
                    rocblas_stride stride_A,
                    TPtr __restrict__ xa,
                    ptrdiff_t      shift_x,
                    rocblas_int    incx,
                    rocblas_stride stride_x)
{
    const auto* AP = load_ptr_batch(APa, hipBlockIdx_x, shift_A, stride_A);
    auto*       x  = load_ptr_batch(xa, hipBlockIdx_x, shift_x, stride_x);

    bool is_diag = diag == rocblas_diagonal_unit;

    if(transA == rocblas_operation_none)
    {
        if(uplo == rocblas_fill_upper)
            tpsv_backward_substitution_calc<false, BLK_SIZE>(is_diag, false, n, AP, x, incx);
        else
            tpsv_forward_substitution_calc<false, BLK_SIZE>(is_diag, false, n, AP, x, incx);
    }
    else if(uplo == rocblas_fill_upper)
        tpsv_forward_substitution_calc<CONJ, BLK_SIZE>(is_diag, true, n, AP, x, incx);
    else
        tpsv_backward_substitution_calc<CONJ, BLK_SIZE>(is_diag, true, n, AP, x, incx);
}

template <rocblas_int BLOCK, typename TConstPtr, typename TPtr>
rocblas_status rocblas_tpsv_template(rocblas_handle    handle,
                                     rocblas_fill      uplo,
                                     rocblas_operation transA,
                                     rocblas_diagonal  diag,
                                     rocblas_int       n,
                                     TConstPtr         A,
                                     rocblas_int       offset_A,
                                     rocblas_stride    stride_A,
                                     TPtr              x,
                                     rocblas_int       offset_x,
                                     rocblas_int       incx,
                                     rocblas_stride    stride_x,
                                     rocblas_int       batch_count)
{
    if(batch_count == 0 || n == 0)
        return rocblas_status_success;

    // Temporarily switch to host pointer mode, restoring on return
    // cppcheck-suppress unreadVariable
    auto saved_pointer_mode = handle->push_pointer_mode(rocblas_pointer_mode_host);

    ptrdiff_t shift_x = incx < 0 ? offset_x - ptrdiff_t(incx) * (n - 1) : offset_x;
    ptrdiff_t shift_A = offset_A;

    dim3 grid(batch_count);
    dim3 threads(BLOCK);

    if(rocblas_operation_conjugate_transpose == transA)
    {
        hipLaunchKernelGGL((rocblas_tpsv_kernel<true, BLOCK>),
                           grid,
                           threads,
                           0,
                           handle->get_stream(),
                           uplo,
                           transA,
                           diag,
                           n,
                           A,
                           shift_A,
                           stride_A,
                           x,
                           shift_x,
                           incx,
                           stride_x);
    }
    else
    {
        hipLaunchKernelGGL((rocblas_tpsv_kernel<false, BLOCK>),
                           grid,
                           threads,
                           0,
                           handle->get_stream(),
                           uplo,
                           transA,
                           diag,
                           n,
                           A,
                           shift_A,
                           stride_A,
                           x,
                           shift_x,
                           incx,
                           stride_x);
    }

    return rocblas_status_success;
}

//TODO :-Add rocblas_check_numerics_tp_matrix_template for checking Matrix `AP` which is a Triangular Packed Matrix
template <typename T, typename U>
rocblas_status rocblas_tpsv_check_numerics(const char*    function_name,
                                           rocblas_handle handle,
                                           rocblas_int    n,
                                           T              AP,
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
// template parameters in the files *tpsv*.cpp

// clang-format off

#ifdef INSTANTIATE_TPSV_TEMPLATE
#error INSTANTIATE_TPSV_TEMPLATE already defined
#endif

#define INSTANTIATE_TPSV_TEMPLATE(BLOCK_, TConstPtr_, TPtr_)              \
template rocblas_status rocblas_tpsv_template<BLOCK_, TConstPtr_, TPtr_> \
                                    (rocblas_handle    handle,           \
                                     rocblas_fill      uplo,             \
                                     rocblas_operation transA,           \
                                     rocblas_diagonal  diag,             \
                                     rocblas_int       n,                \
                                     TConstPtr_        A,                \
                                     rocblas_int       offset_A,         \
                                     rocblas_stride    stride_A,         \
                                     TPtr_             x,                \
                                     rocblas_int       offset_x,         \
                                     rocblas_int       incx,             \
                                     rocblas_stride    stride_x,         \
                                     rocblas_int       batch_count);

INSTANTIATE_TPSV_TEMPLATE(512, float const*, float*)
INSTANTIATE_TPSV_TEMPLATE(512, double const*, double*)
INSTANTIATE_TPSV_TEMPLATE(512, rocblas_float_complex const*, rocblas_float_complex*)
INSTANTIATE_TPSV_TEMPLATE(512, rocblas_double_complex const*, rocblas_double_complex*)
INSTANTIATE_TPSV_TEMPLATE(512, float const* const*, float* const*)
INSTANTIATE_TPSV_TEMPLATE(512, double const* const*, double* const*)
INSTANTIATE_TPSV_TEMPLATE(512, rocblas_float_complex const* const*, rocblas_float_complex* const*)
INSTANTIATE_TPSV_TEMPLATE(512, rocblas_double_complex const* const*, rocblas_double_complex* const*)

#undef INSTANTIATE_TPSV_TEMPLATE

#ifdef INSTANTIATE_TPSV_NUMERICS
#error INSTANTIATE_TPSV_NUMERICS already defined
#endif

#define INSTANTIATE_TPSV_NUMERICS(T_, U_)                                 \
template rocblas_status rocblas_tpsv_check_numerics<T_, U_>               \
                                          (const char*    function_name,  \
                                           rocblas_handle handle,         \
                                           rocblas_int    n,              \
                                           T_             AP,             \
                                           rocblas_int    offset_a,       \
                                           rocblas_stride stride_a,       \
                                           U_             x,              \
                                           rocblas_int    offset_x,       \
                                           rocblas_int    inc_x,          \
                                           rocblas_stride stride_x,       \
                                           rocblas_int    batch_count,    \
                                           const int      check_numerics, \
                                           bool           is_input);

INSTANTIATE_TPSV_NUMERICS(float const*, float*)
INSTANTIATE_TPSV_NUMERICS(double const*, double*)
INSTANTIATE_TPSV_NUMERICS(rocblas_float_complex const*, rocblas_float_complex*)
INSTANTIATE_TPSV_NUMERICS(rocblas_double_complex const*, rocblas_double_complex*)
INSTANTIATE_TPSV_NUMERICS(float const* const*, float* const*)
INSTANTIATE_TPSV_NUMERICS(double const* const*, double* const*)
INSTANTIATE_TPSV_NUMERICS(rocblas_float_complex const* const*, rocblas_float_complex* const*)
INSTANTIATE_TPSV_NUMERICS(rocblas_double_complex const* const*, rocblas_double_complex* const*)

#undef INSTANTIATE_TPSV_NUMERICS

// clang-format on
