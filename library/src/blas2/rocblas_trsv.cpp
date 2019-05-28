/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>

#include "rocblas_trsv.hpp"
#include "rocblas.h"
#include "status.h"
#include "definitions.h"
#include "gemv.hpp"
#include "../blas3/trtri_trsm.hpp"
#include "rocblas_unique_ptr.hpp"
#include "handle.h"
#include "logging.h"
#include "utility.h"

namespace {

#define A(ii, jj) (A + (ii) + (jj)*lda)
#define B(ii) (B + (ii))
#define X(ii) (X + (ii))
#define invA(ii) (invA + (ii)*BLOCK)

template <typename T>
__global__ void flip_vector_kernel(T* data, rocblas_int size)
{
    ssize_t tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if(tx < (size / 2))
    {
        T temp              = data[tx];
        data[tx]            = data[size - tx - 1];
        data[size - tx - 1] = temp;
    }
}

template <typename T>
void flip_vector(hipStream_t rocblas_stream, T* data, rocblas_int size)
{
    static constexpr int NB_X = 1024;
    rocblas_int blocksX       = (size / 2) / NB_X + 1;
    dim3 grid(blocksX);
    dim3 threads(NB_X);

    hipLaunchKernelGGL(flip_vector_kernel, grid, threads, 0, rocblas_stream, data, size);
}

template <typename T>
__global__ void strided_vector_copy_kernel(
    T* dst, rocblas_int dst_incx, const T* src, rocblas_int src_incx, rocblas_int size)
{
    ssize_t tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if(tx < size)
    {
        dst[tx * dst_incx] = src[tx * src_incx];
    }
}

template <typename T>
void strided_vector_copy(hipStream_t rocblas_stream,
                         T* dst,
                         rocblas_int dst_incx,
                         T* src,
                         rocblas_int src_incx,
                         rocblas_int size)
{
    static constexpr int NB_X = 1024;
    rocblas_int blocksX       = (size - 1) / NB_X + 1;
    dim3 grid(blocksX);
    dim3 threads(NB_X);

    hipLaunchKernelGGL(strided_vector_copy_kernel,
                       grid,
                       threads,
                       0,
                       rocblas_stream,
                       dst,
                       dst_incx,
                       src,
                       src_incx,
                       size);
}

template <rocblas_int BLOCK, typename T>
rocblas_status rocblas_trsv_left(rocblas_handle handle,
                                 rocblas_fill uplo,
                                 rocblas_operation transA,
                                 rocblas_int m,
                                 const T* A,
                                 rocblas_int lda,
                                 T* B,
                                 rocblas_int incx,
                                 const T* invA,
                                 T* X)
{
    static constexpr T negative_one = -1;
    static constexpr T one          = 1;
    static constexpr T zero         = 0;

    const T* p_one          = &one;
    const T* p_zero         = &zero;
    const T* p_negative_one = &negative_one;

    if(handle->pointer_mode == rocblas_pointer_mode_device)
    {
        p_one          = reinterpret_cast<T*>(handle->get_trsv_one());
        p_zero         = reinterpret_cast<T*>(handle->get_trsv_zero());
        p_negative_one = reinterpret_cast<T*>(handle->get_trsv_negative_one());
    }

    rocblas_int i, jb;

    // transB is always non-transpose
    rocblas_operation transB = rocblas_operation_none;

    if(transA == transB)
    {
        if(uplo == rocblas_fill_lower)
        {
            // left, lower no-transpose
            jb = min(BLOCK, m);
            rocblas_gemv_template<T>(
                handle, transA, jb, jb, p_one, invA, BLOCK, B, incx, p_zero, X, 1);

            if(BLOCK < m)
            {
                rocblas_gemv_template<T>(handle,
                                         transA,
                                         m - BLOCK,
                                         BLOCK,
                                         p_negative_one,
                                         A(BLOCK, 0),
                                         lda,
                                         X,
                                         1,
                                         p_one,
                                         B(BLOCK * incx),
                                         incx);

                // remaining blocks
                for(i = BLOCK; i < m; i += BLOCK)
                {
                    jb = min(m - i, BLOCK);

                    rocblas_gemv_template<T>(handle,
                                             transA,
                                             jb,
                                             jb,
                                             p_one,
                                             invA(i),
                                             BLOCK,
                                             B(i * incx),
                                             incx,
                                             p_zero,
                                             X(i),
                                             1);
                    // this condition is not necessary at all and can be changed as if (i+BLOCK<m)
                    if(i + BLOCK >= m)
                        break;

                    rocblas_gemv_template<T>(handle,
                                             transA,
                                             m - i - BLOCK,
                                             BLOCK,
                                             p_negative_one,
                                             A(i + BLOCK, i),
                                             lda,
                                             X(i),
                                             1,
                                             p_one,
                                             B((i + BLOCK) * incx),
                                             incx);
                }
            }
        }
        else
        {
            // left, upper no-transpose
            jb = (m % BLOCK == 0) ? BLOCK : (m % BLOCK);
            i  = m - jb;

            // if m=n=35=lda=ldb, BLOCK =32, then jb = 3, i = 32; {3, 35, 3, 32, 35, 35}
            rocblas_gemv_template<T>(
                handle, transA, jb, jb, p_one, invA(i), BLOCK, B(i * incx), incx, p_zero, X(i), 1);

            if(i - BLOCK >= 0)
            {
                rocblas_gemv_template<T>(
                    handle, transA, i, jb, p_negative_one, A(0, i), lda, X(i), 1, p_one, B, incx);

                // remaining blocks
                for(i = m - jb - BLOCK; i >= 0; i -= BLOCK)
                {
                    //{32, 35, 32, 32, 35, 35}
                    rocblas_gemv_template<T>(handle,
                                             transA,
                                             BLOCK,
                                             BLOCK,
                                             p_one,
                                             invA(i),
                                             BLOCK,
                                             B(i * incx),
                                             incx,
                                             p_zero,
                                             X(i),
                                             1);

                    if(i - BLOCK < 0)
                        break;

                    rocblas_gemv_template<T>(handle,
                                             transA,
                                             i,
                                             BLOCK,
                                             p_negative_one,
                                             A(0, i),
                                             lda,
                                             X(i),
                                             1,
                                             p_one,
                                             B,
                                             incx);
                }
            }
        }
    }
    else
    { // transA == rocblas_operation_transpose || transA == rocblas_operation_conjugate_transpose
        if(uplo == rocblas_fill_lower)
        {
            // left, lower transpose
            jb = (m % BLOCK == 0) ? BLOCK : (m % BLOCK);
            i  = m - jb;

            rocblas_gemv_template<T>(
                handle, transA, jb, jb, p_one, invA(i), BLOCK, B(i * incx), incx, p_zero, X(i), 1);

            if(i - BLOCK >= 0)
            {
                rocblas_gemv_template<T>(
                    handle, transA, jb, i, p_negative_one, A(i, 0), lda, X(i), 1, p_one, B, incx);

                // remaining blocks
                for(i = m - jb - BLOCK; i >= 0; i -= BLOCK)
                {
                    rocblas_gemv_template<T>(handle,
                                             transA,
                                             BLOCK,
                                             BLOCK,
                                             p_one,
                                             invA(i),
                                             BLOCK,
                                             B(i * incx),
                                             incx,
                                             p_zero,
                                             X(i),
                                             1);

                    if(i - BLOCK < 0)
                        break;

                    rocblas_gemv_template<T>(handle,
                                             transA,
                                             BLOCK,
                                             i,
                                             p_negative_one,
                                             A(i, 0),
                                             lda,
                                             X(i),
                                             1,
                                             p_one,
                                             B,
                                             incx);
                }
            }
        }
        else
        {
            // left, upper transpose
            jb = min(BLOCK, m);
            rocblas_gemv_template<T>(
                handle, transA, jb, jb, p_one, invA, BLOCK, B, incx, p_zero, X, 1);

            if(BLOCK < m)
            {
                rocblas_gemv_template<T>(handle,
                                         transA,
                                         BLOCK,
                                         m - BLOCK,
                                         p_negative_one,
                                         A(0, BLOCK),
                                         lda,
                                         X,
                                         1,
                                         p_one,
                                         B(BLOCK * incx),
                                         incx);

                // remaining blocks
                for(i = BLOCK; i < m; i += BLOCK)
                {
                    jb = min(m - i, BLOCK);
                    rocblas_gemv_template<T>(handle,
                                             transA,
                                             jb,
                                             jb,
                                             p_one,
                                             invA(i),
                                             BLOCK,
                                             B(i * incx),
                                             incx,
                                             p_zero,
                                             X(i),
                                             1);

                    if(i + BLOCK >= m)
                        break;

                    rocblas_gemv_template<T>(handle,
                                             transA,
                                             BLOCK,
                                             m - i - BLOCK,
                                             p_negative_one,
                                             A(i, i + BLOCK),
                                             lda,
                                             X(i),
                                             1,
                                             p_one,
                                             B((i + BLOCK) * incx),
                                             incx);
                }
            }
        }
    } // transpose

    return rocblas_status_success;
}

template <rocblas_int BLOCK, typename T>
rocblas_status special_trsv_template(rocblas_handle handle,
                                     rocblas_fill uplo,
                                     rocblas_operation transA,
                                     rocblas_diagonal diag,
                                     rocblas_int m,
                                     const T* A,
                                     rocblas_int lda,
                                     T* B,
                                     rocblas_int incx,
                                     const T* supplied_invA,
                                     rocblas_int ldinvA,
                                     const size_t* B_chunk,
                                     T* x)
{
    hipStream_t rocblas_stream;
    RETURN_IF_ROCBLAS_ERROR(rocblas_get_stream(handle, &rocblas_stream));

    if(*B_chunk == 0)
        return rocblas_status_invalid_size;

    if(!x)
    {
        T* invA_temp = reinterpret_cast<T*>(handle->get_trsm_invA());
        T* invA_C    = reinterpret_cast<T*>(handle->get_trsm_invA_C());

        rocblas_trtri_trsm_template<T, BLOCK>(handle, invA_C, uplo, diag, m, A, lda, invA_temp);
    }

    T* x_temp     = x ? x : reinterpret_cast<T*>(handle->get_trsm_Y());
    const T* invA = x ? supplied_invA : reinterpret_cast<T*>(handle->get_trsm_invA());

    int R                = m / BLOCK;
    const T zero         = 0;
    const T one          = 1;
    const T negative_one = -1;

    const T* p_one          = &one;
    const T* p_zero         = &zero;
    const T* p_negative_one = &negative_one;

    if(handle->pointer_mode == rocblas_pointer_mode_device)
    {
        p_one          = reinterpret_cast<T*>(handle->get_trsv_one());
        p_zero         = reinterpret_cast<T*>(handle->get_trsv_zero());
        p_negative_one = reinterpret_cast<T*>(handle->get_trsv_negative_one());
    }

    for(int r = 0; r < R; r++)
    {
        int q = R - 1 - r;

        int j = (((uplo == rocblas_fill_lower) && (transA == rocblas_operation_none)) ||
                 ((uplo == rocblas_fill_upper) && (transA == rocblas_operation_transpose)))
                    ? r
                    : q;

        // copy a BLOCK*n piece we are solving at a time
        strided_vector_copy<T>(rocblas_stream, x_temp, 1, B + j * BLOCK * incx, incx, BLOCK);

        if(r > 0)
        {
            const T* A_current = nullptr;
            T* B_current       = nullptr;

            if((uplo == rocblas_fill_upper) && (transA == rocblas_operation_transpose))
            {
                A_current = A + r * BLOCK * lda;
                B_current = B;
            }
            else if((uplo == rocblas_fill_lower) && (transA == rocblas_operation_none))
            {
                A_current = A + r * BLOCK;
                B_current = B;
            }
            else if((uplo == rocblas_fill_lower) && (transA == rocblas_operation_transpose))
            {
                A_current = A + q * BLOCK * lda + (q + 1) * BLOCK;
                B_current = B + (q + 1) * BLOCK * incx;
            }
            else // ((uplo == rocblas_fill_upper) && (transA == rocblas_operation_none))
            {
                A_current = A + (q + 1) * BLOCK * lda + q * BLOCK;
                B_current = B + (q + 1) * BLOCK * incx;
            }

            rocblas_int M = (transA == rocblas_operation_none) ? BLOCK : r * BLOCK;
            rocblas_int N = (transA == rocblas_operation_none) ? r * BLOCK : BLOCK;
            rocblas_gemv_template(handle,
                                  transA,
                                  M,
                                  N,
                                  p_negative_one,
                                  A_current,
                                  lda,
                                  B_current,
                                  incx,
                                  p_one,
                                  (T*)x_temp,
                                  1);
        }

        rocblas_gemv_template(handle,
                              transA,
                              BLOCK,
                              BLOCK,
                              p_one,
                              ((T*)invA) + j * BLOCK * BLOCK,
                              BLOCK,
                              (T*)x_temp,
                              1,
                              p_zero,
                              (T*)B + j * BLOCK * incx,
                              incx);
    }

    return rocblas_status_success;
}

template <rocblas_int BLOCK, typename T>
rocblas_status rocblas_trsv_impl(rocblas_handle handle,
                                 rocblas_fill uplo,
                                 rocblas_operation transA,
                                 rocblas_diagonal diag,
                                 rocblas_int m,
                                 const T* A,
                                 rocblas_int lda,
                                 T* B,
                                 rocblas_int incx,
                                 const T* invA,
                                 rocblas_int ldInvA,
                                 const size_t* x_temp_size,
                                 T* x_temp)
{
    if(m % BLOCK == 0 && m <= BLOCK * *(handle->get_trsm_A_blks()))
    {
        rocblas_operation trA = transA;
        if(trA == rocblas_operation_conjugate_transpose)
            trA = rocblas_operation_transpose;

        return special_trsv_template<BLOCK>(
            handle, uplo, trA, diag, m, A, lda, B, incx, invA, ldInvA, x_temp_size, x_temp);
    }

    hipStream_t rocblas_stream;
    RETURN_IF_ROCBLAS_ERROR(rocblas_get_stream(handle, &rocblas_stream));

    rocblas_status status =
        rocblas_trsv_left<BLOCK>(handle, uplo, transA, m, A, lda, B, incx, invA, x_temp);

    // copy solution X into B
    strided_vector_copy(rocblas_stream, B, incx, x_temp, 1, m);

    return status;
}

} // namespace

template <typename>
static constexpr char rocblas_trsv_name[] = "unknown";
template <>
static constexpr char rocblas_trsv_name<float>[] = "rocblas_strsv";
template <>
static constexpr char rocblas_trsv_name<double>[] = "rocblas_dtrsv";

template <rocblas_int BLOCK, typename T>
rocblas_status rocblas_trsv_ex_template(rocblas_handle handle,
                                        rocblas_fill uplo,
                                        rocblas_operation transA,
                                        rocblas_diagonal diag,
                                        rocblas_int m,
                                        const T* A,
                                        rocblas_int lda,
                                        T* B,
                                        rocblas_int incx,
                                        const T* invA,
                                        rocblas_int ldInvA,
                                        const size_t* x_temp_size,
                                        T* x_temp)
{
    if(!m)
        return rocblas_status_success;

    if(!invA)
        return rocblas_status_memory_error;

    if(!x_temp_size || (*x_temp_size) < m)
        return rocblas_status_invalid_size;

    if(handle->pointer_mode == rocblas_pointer_mode_device)
    {
        static const T one{1};
        static const T zero{0};
        static const T negative_one{-1};

        T* one_d = (T*)handle->get_trsv_one();
        RETURN_IF_HIP_ERROR(hipMemcpy(one_d, &one, sizeof(T), hipMemcpyHostToDevice));
        T* zero_d = (T*)handle->get_trsv_zero();
        RETURN_IF_HIP_ERROR(hipMemcpy(zero_d, &zero, sizeof(T), hipMemcpyHostToDevice));
        T* negative_one_d = (T*)handle->get_trsv_negative_one();
        RETURN_IF_HIP_ERROR(
            hipMemcpy(negative_one_d, &negative_one, sizeof(T), hipMemcpyHostToDevice));
    }

    hipStream_t rocblas_stream;
    RETURN_IF_ROCBLAS_ERROR(rocblas_get_stream(handle, &rocblas_stream));

    // TODO: workaround to fix negative incx issue
    rocblas_int abs_incx = (incx > 0) ? incx : -incx;

    if(incx < 0)
    {
        flip_vector(rocblas_stream, B, (m - 1) * abs_incx + 1);
    }

    rocblas_status status = rocblas_trsv_impl<BLOCK>(
        handle, uplo, transA, diag, m, A, lda, B, abs_incx, invA, ldInvA, x_temp_size, x_temp);

    if(incx < 0)
    {
        flip_vector(rocblas_stream, B, (m - 1) * abs_incx + 1);
    }

    return status;
}

template <rocblas_int BLOCK, typename T>
rocblas_status rocblas_trsv_template(rocblas_handle handle,
                                     rocblas_fill uplo,
                                     rocblas_operation transA,
                                     rocblas_diagonal diag,
                                     rocblas_int m,
                                     const T* A,
                                     rocblas_int lda,
                                     T* B,
                                     rocblas_int incx)
{
    if(!handle)
        return rocblas_status_invalid_handle;

    auto pointer_mode = handle->pointer_mode;
    auto layer_mode   = handle->layer_mode;

    if(layer_mode & rocblas_layer_mode_log_trace)
        log_trace(handle, rocblas_trsv_name<T>, uplo, transA, diag, m, A, lda, B, incx);

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
    if(!A || !B)
        return rocblas_status_invalid_pointer;
    if(m < 0 || lda < m || lda < 1 || !incx)
        return rocblas_status_invalid_size;

    // quick return if possible.
    if(!m)
        return rocblas_status_success;

    if(handle->pointer_mode == rocblas_pointer_mode_device)
    {
        static const T one{1};
        static const T zero{0};
        static const T negative_one{-1};

        T* one_d = (T*)handle->get_trsv_one();
        RETURN_IF_HIP_ERROR(hipMemcpy(one_d, &one, sizeof(T), hipMemcpyHostToDevice));
        T* zero_d = (T*)handle->get_trsv_zero();
        RETURN_IF_HIP_ERROR(hipMemcpy(zero_d, &zero, sizeof(T), hipMemcpyHostToDevice));
        T* negative_one_d = (T*)handle->get_trsv_negative_one();
        RETURN_IF_HIP_ERROR(
            hipMemcpy(negative_one_d, &negative_one, sizeof(T), hipMemcpyHostToDevice));
    }

    hipStream_t rocblas_stream;
    RETURN_IF_ROCBLAS_ERROR(rocblas_get_stream(handle, &rocblas_stream));

    // TODO: workaround to fix negative incx issue
    rocblas_int abs_incx = (incx > 0) ? incx : -incx;

    if(m % BLOCK == 0 && m <= BLOCK * *(handle->get_trsm_A_blks()))
    {
        rocblas_operation trA = transA;
        if(trA == rocblas_operation_conjugate_transpose)
            trA = rocblas_operation_transpose;

        if(incx < 0)
        {
            flip_vector(rocblas_stream, B, (m - 1) * abs_incx + 1);
        }

        T* x_temp                 = nullptr;
        const T* invA             = nullptr;
        const size_t* x_temp_size = nullptr;

        rocblas_status status = special_trsv_template<BLOCK>(handle,
                                                             uplo,
                                                             trA,
                                                             diag,
                                                             m,
                                                             A,
                                                             lda,
                                                             B,
                                                             abs_incx,
                                                             invA,
                                                             0,
                                                             (handle->get_trsm_B_chnk()),
                                                             x_temp);
        if(incx < 0)
        {
            flip_vector(rocblas_stream, B, (m - 1) * abs_incx + 1);
        }

        return status;
    }

    // invA is of size BLOCK*k, BLOCK is the blocking size
    // used unique_ptr to avoid memory leak
    auto invA =
        rocblas_unique_ptr{rocblas::device_malloc(BLOCK * m * sizeof(T)), rocblas::device_free};
    if(!invA)
        return rocblas_status_memory_error;

    auto C_tmp = rocblas_unique_ptr{
        rocblas::device_malloc(sizeof(T) * (BLOCK / 2) * (BLOCK / 2) * (m / BLOCK)),
        rocblas::device_free};
    if(!C_tmp && m >= BLOCK)
        return rocblas_status_memory_error;

    // X is size of packed B
    auto X = rocblas_unique_ptr{rocblas::device_malloc(m * sizeof(T)), rocblas::device_free};
    if(!X)
        return rocblas_status_memory_error;

    // batched trtri invert diagonal part (BLOCK*BLOCK) of A into invA
    rocblas_status status = rocblas_trtri_trsm_template<T, BLOCK>(
        handle, (T*)C_tmp.get(), uplo, diag, m, A, lda, (T*)invA.get());
    if(status != rocblas_status_success)
        return status;

    if(incx < 0)
    {
        flip_vector(rocblas_stream, B, (m - 1) * abs_incx + 1);
    }

    status = rocblas_trsv_impl<BLOCK>(handle,
                                      uplo,
                                      transA,
                                      diag,
                                      m,
                                      A,
                                      lda,
                                      B,
                                      abs_incx,
                                      (T*)invA.get(),
                                      BLOCK,
                                      (handle->get_trsm_B_chnk()),
                                      (T*)X.get());

    if(incx < 0)
    {
        flip_vector(rocblas_stream, B, (m - 1) * abs_incx + 1);
    }

    return status;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_strsv_ex(rocblas_handle handle,
                                rocblas_fill uplo,
                                rocblas_operation transA,
                                rocblas_diagonal diag,
                                rocblas_int m,
                                const float* A,
                                rocblas_int lda,
                                float* x,
                                rocblas_int incx,
                                const float* invA,
                                rocblas_int ldInvA,
                                const size_t* x_temp_size,
                                float* x_temp)
{
    static constexpr rocblas_int STRSV_BLOCK = 128;
    return rocblas_trsv_ex_template<STRSV_BLOCK>(
        handle, uplo, transA, diag, m, A, lda, x, incx, invA, ldInvA, x_temp_size, x_temp);
}

rocblas_status rocblas_dtrsv_ex(rocblas_handle handle,
                                rocblas_fill uplo,
                                rocblas_operation transA,
                                rocblas_diagonal diag,
                                rocblas_int m,
                                const double* A,
                                rocblas_int lda,
                                double* x,
                                rocblas_int incx,
                                const double* invA,
                                rocblas_int ldInvA,
                                const size_t* x_temp_size,
                                double* x_temp)
{
    static constexpr rocblas_int DTRSV_BLOCK = 128;
    return rocblas_trsv_ex_template<DTRSV_BLOCK>(
        handle, uplo, transA, diag, m, A, lda, x, incx, invA, ldInvA, x_temp_size, x_temp);
}

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
    return rocblas_trsv_template<STRSV_BLOCK>(handle, uplo, transA, diag, m, A, lda, x, incx);
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
    return rocblas_trsv_template<DTRSV_BLOCK>(handle, uplo, transA, diag, m, A, lda, x, incx);
}

} // extern "C"
