/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "../blas3/trtri_trsm.hpp"
#include "definitions.h"
#include "gemv.hpp"
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "utility.h"
#include <algorithm>
#include <hip/hip_runtime_api.h>
#include <tuple>

namespace
{
    using std::max;
    using std::min;

    constexpr rocblas_int NB_X        = 1024;
    constexpr rocblas_int STRSV_BLOCK = 128;
    constexpr rocblas_int DTRSV_BLOCK = 128;
    constexpr rocblas_int NB          = 16;

    template <typename T>
    constexpr T negative_one = -1;

    template <typename T>
    constexpr T zero = 0;

    template <typename T>
    constexpr T one = 1;

    template <typename T>
    __global__ void flip_vector_kernel(T* __restrict__ data,
                                       T* __restrict__ data_end,
                                       rocblas_int size,
                                       rocblas_int abs_incx)
    {
        ptrdiff_t tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
        if(tx < size)
        {
            auto offset       = tx * abs_incx;
            auto temp         = data[offset];
            data[offset]      = data_end[-offset];
            data_end[-offset] = temp;
        }
    }

    template <typename T>
    void flip_vector(rocblas_handle handle, T* data, rocblas_int m, rocblas_int abs_incx)
    {
        T*          data_end = data + (m - 1) * abs_incx;
        rocblas_int size     = (m + 1) / 2;
        rocblas_int blocksX  = (size - 1) / NB_X + 1;
        dim3        grid     = blocksX;
        dim3        threads  = NB_X;

        hipLaunchKernelGGL(flip_vector_kernel,
                           grid,
                           threads,
                           0,
                           handle->rocblas_stream,
                           data,
                           data_end,
                           size,
                           abs_incx);
    }

    template <typename T>
    __global__ void strided_vector_copy_kernel(T* __restrict__ dst,
                                               rocblas_int dst_incx,
                                               const T* __restrict__ src,
                                               rocblas_int src_incx,
                                               rocblas_int size)
    {
        ptrdiff_t tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
        if(tx < size)
            dst[tx * dst_incx] = src[tx * src_incx];
    }

    template <typename T>
    void strided_vector_copy(rocblas_handle handle,
                             T*             dst,
                             rocblas_int    dst_incx,
                             T*             src,
                             rocblas_int    src_incx,
                             rocblas_int    size)
    {
        rocblas_int blocksX = (size - 1) / NB_X + 1;
        dim3        grid    = blocksX;
        dim3        threads = NB_X;

        hipLaunchKernelGGL(strided_vector_copy_kernel,
                           grid,
                           threads,
                           0,
                           handle->rocblas_stream,
                           dst,
                           dst_incx,
                           src,
                           src_incx,
                           size);
    }

    template <rocblas_int BLOCK, typename T>
    rocblas_status rocblas_trsv_left(rocblas_handle    handle,
                                     rocblas_fill      uplo,
                                     rocblas_operation transA,
                                     rocblas_int       m,
                                     const T*          A,
                                     rocblas_int       lda,
                                     T*                B,
                                     rocblas_int       incx,
                                     const T*          invA,
                                     T*                X)
    {
        rocblas_int i, jb;

        if(transA == rocblas_operation_none)
        {
            if(uplo == rocblas_fill_lower)
            {
                // left, lower no-transpose
                jb = min(BLOCK, m);
                rocblas_gemv_template(
                    handle, transA, jb, jb, &one<T>, invA, BLOCK, B, incx, &zero<T>, X, 1);

                if(BLOCK < m)
                {
                    rocblas_gemv_template(handle,
                                          transA,
                                          m - BLOCK,
                                          BLOCK,
                                          &negative_one<T>,
                                          A + BLOCK,
                                          lda,
                                          X,
                                          1,
                                          &one<T>,
                                          B + BLOCK * incx,
                                          incx);

                    // remaining blocks
                    for(i = BLOCK; i < m; i += BLOCK)
                    {
                        jb = min(m - i, BLOCK);

                        rocblas_gemv_template(handle,
                                              transA,
                                              jb,
                                              jb,
                                              &one<T>,
                                              invA + i * BLOCK,
                                              BLOCK,
                                              B + i * incx,
                                              incx,
                                              &zero<T>,
                                              X + i,
                                              1);
                        if(i + BLOCK < m)
                            rocblas_gemv_template(handle,
                                                  transA,
                                                  m - i - BLOCK,
                                                  BLOCK,
                                                  &negative_one<T>,
                                                  A + i + BLOCK + i * lda,
                                                  lda,
                                                  X + i,
                                                  1,
                                                  &one<T>,
                                                  B + (i + BLOCK) * incx,
                                                  incx);
                    }
                }
            }
            else
            {
                // left, upper no-transpose
                jb = m % BLOCK == 0 ? BLOCK : m % BLOCK;
                i  = m - jb;

                // if m=n=35=lda=ldb, BLOCK =32, then jb = 3, i = 32; {3, 35, 3, 32, 35, 35}
                rocblas_gemv_template(handle,
                                      transA,
                                      jb,
                                      jb,
                                      &one<T>,
                                      invA + i * BLOCK,
                                      BLOCK,
                                      B + i * incx,
                                      incx,
                                      &zero<T>,
                                      X + i,
                                      1);

                if(i >= BLOCK)
                {
                    rocblas_gemv_template(handle,
                                          transA,
                                          i,
                                          jb,
                                          &negative_one<T>,
                                          A + i * lda,
                                          lda,
                                          X + i,
                                          1,
                                          &one<T>,
                                          B,
                                          incx);

                    // remaining blocks
                    for(i = m - jb - BLOCK; i >= 0; i -= BLOCK)
                    {
                        //{32, 35, 32, 32, 35, 35}
                        rocblas_gemv_template(handle,
                                              transA,
                                              BLOCK,
                                              BLOCK,
                                              &one<T>,
                                              invA + i * BLOCK,
                                              BLOCK,
                                              B + i * incx,
                                              incx,
                                              &zero<T>,
                                              X + i,
                                              1);

                        if(i >= BLOCK)
                            rocblas_gemv_template(handle,
                                                  transA,
                                                  i,
                                                  BLOCK,
                                                  &negative_one<T>,
                                                  A + i * lda,
                                                  lda,
                                                  X + i,
                                                  1,
                                                  &one<T>,
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
                jb = m % BLOCK == 0 ? BLOCK : m % BLOCK;
                i  = m - jb;

                rocblas_gemv_template(handle,
                                      transA,
                                      jb,
                                      jb,
                                      &one<T>,
                                      invA + i * BLOCK,
                                      BLOCK,
                                      B + i * incx,
                                      incx,
                                      &zero<T>,
                                      X + i,
                                      1);

                if(i - BLOCK >= 0)
                {
                    rocblas_gemv_template(handle,
                                          transA,
                                          jb,
                                          i,
                                          &negative_one<T>,
                                          A + i,
                                          lda,
                                          X + i,
                                          1,
                                          &one<T>,
                                          B,
                                          incx);

                    // remaining blocks
                    for(i = m - jb - BLOCK; i >= 0; i -= BLOCK)
                    {
                        rocblas_gemv_template(handle,
                                              transA,
                                              BLOCK,
                                              BLOCK,
                                              &one<T>,
                                              invA + i * BLOCK,
                                              BLOCK,
                                              B + i * incx,
                                              incx,
                                              &zero<T>,
                                              X + i,
                                              1);

                        if(i >= BLOCK)
                            rocblas_gemv_template(handle,
                                                  transA,
                                                  BLOCK,
                                                  i,
                                                  &negative_one<T>,
                                                  A + i,
                                                  lda,
                                                  X + i,
                                                  1,
                                                  &one<T>,
                                                  B,
                                                  incx);
                    }
                }
            }
            else
            {
                // left, upper transpose
                jb = min(BLOCK, m);
                rocblas_gemv_template(
                    handle, transA, jb, jb, &one<T>, invA, BLOCK, B, incx, &zero<T>, X, 1);

                if(BLOCK < m)
                {
                    rocblas_gemv_template(handle,
                                          transA,
                                          BLOCK,
                                          m - BLOCK,
                                          &negative_one<T>,
                                          A + BLOCK * lda,
                                          lda,
                                          X,
                                          1,
                                          &one<T>,
                                          B + BLOCK * incx,
                                          incx);

                    // remaining blocks
                    for(i = BLOCK; i < m; i += BLOCK)
                    {
                        jb = min(m - i, BLOCK);
                        rocblas_gemv_template(handle,
                                              transA,
                                              jb,
                                              jb,
                                              &one<T>,
                                              invA + i * BLOCK,
                                              BLOCK,
                                              B + i * incx,
                                              incx,
                                              &zero<T>,
                                              X + i,
                                              1);

                        if(i + BLOCK < m)
                            rocblas_gemv_template(handle,
                                                  transA,
                                                  BLOCK,
                                                  m - i - BLOCK,
                                                  &negative_one<T>,
                                                  A + i + (i + BLOCK) * lda,
                                                  lda,
                                                  X + i,
                                                  1,
                                                  &one<T>,
                                                  B + (i + BLOCK) * incx,
                                                  incx);
                    }
                }
            }
        } // transpose

        return rocblas_status_success;
    }

    template <rocblas_int BLOCK, typename T>
    rocblas_status special_trsv_template(rocblas_handle    handle,
                                         rocblas_fill      uplo,
                                         rocblas_operation transA,
                                         rocblas_diagonal  diag,
                                         rocblas_int       m,
                                         const T*          A,
                                         rocblas_int       lda,
                                         T*                B,
                                         ptrdiff_t         incx,
                                         const T*          invA,
                                         T*                x_temp)
    {
        rocblas_int R = m / BLOCK;
        for(rocblas_int r = 0; r < R; r++)
        {
            rocblas_int q = R - 1 - r;
            rocblas_int j
                = (transA == rocblas_operation_none ? rocblas_fill_lower : rocblas_fill_upper)
                          == uplo
                      ? r
                      : q;

            // copy a BLOCK*n piece we are solving at a time
            strided_vector_copy(handle, x_temp, 1, B + j * BLOCK * incx, incx, BLOCK);

            if(r > 0)
            {
                const T*    A_current;
                T*          B_current;
                rocblas_int M, N;

                if(transA == rocblas_operation_none)
                {
                    M = BLOCK;
                    N = r * BLOCK;
                    if(uplo == rocblas_fill_upper)
                    {
                        A_current = A + (q + 1) * BLOCK * lda + q * BLOCK;
                        B_current = B + (q + 1) * BLOCK * incx;
                    }
                    else
                    {
                        A_current = A + N;
                        B_current = B;
                    }
                }
                else
                {
                    M = r * BLOCK;
                    N = BLOCK;
                    if(transA == rocblas_operation_none)
                    {
                        A_current = A + M * lda;
                        B_current = B;
                    }
                    else
                    {
                        A_current = A + q * BLOCK * lda + (q + 1) * BLOCK;
                        B_current = B + (q + 1) * BLOCK * incx;
                    }
                }

                rocblas_gemv_template(handle,
                                      transA,
                                      M,
                                      N,
                                      &negative_one<T>,
                                      A_current,
                                      lda,
                                      B_current,
                                      incx,
                                      &one<T>,
                                      (T*)x_temp,
                                      1);
            }

            rocblas_gemv_template(handle,
                                  transA,
                                  BLOCK,
                                  BLOCK,
                                  &one<T>,
                                  (T*)invA + j * BLOCK * BLOCK,
                                  BLOCK,
                                  (T*)x_temp,
                                  1,
                                  &zero<T>,
                                  (T*)B + j * BLOCK * incx,
                                  incx);
        }

        return rocblas_status_success;
    }

    template <typename>
    constexpr char rocblas_trsv_name[] = "unknown";
    template <>
    constexpr char rocblas_trsv_name<float>[] = "rocblas_strsv";
    template <>
    constexpr char rocblas_trsv_name<double>[] = "rocblas_dtrsv";

    template <rocblas_int BLOCK, typename T>
    rocblas_status rocblas_trsv_ex_template(rocblas_handle    handle,
                                            rocblas_fill      uplo,
                                            rocblas_operation transA,
                                            rocblas_diagonal  diag,
                                            rocblas_int       m,
                                            const T*          A,
                                            rocblas_int       lda,
                                            T*                B,
                                            rocblas_int       incx,
                                            const T*          supplied_invA      = nullptr,
                                            rocblas_int       supplied_invA_size = 0)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        auto layer_mode = handle->layer_mode;
        if(layer_mode & rocblas_layer_mode_log_trace)
            log_trace(handle, rocblas_trsv_name<T>, uplo, transA, diag, m, A, lda, B, incx);

        if(layer_mode & (rocblas_layer_mode_log_bench | rocblas_layer_mode_log_profile))
        {
            auto uplo_letter   = rocblas_fill_letter(uplo);
            auto transA_letter = rocblas_transpose_letter(transA);
            auto diag_letter   = rocblas_diag_letter(diag);

            if(layer_mode & rocblas_layer_mode_log_bench)
            {
                if(handle->pointer_mode == rocblas_pointer_mode_host)
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
        // return rocblas_status_size_unchanged if device memory size query
        if(!m)
            return handle->is_device_memory_size_query() ? rocblas_status_size_unchanged
                                                         : rocblas_status_success;

        // perf_status indicates whether optimal performance is obtainable with available memory
        rocblas_status perf_status = rocblas_status_success;

        // For user-supplied invA, check to make sure size is large enough
        // If not large enough, indicate degraded performance and ignore supplied invA
        if(supplied_invA && supplied_invA_size / BLOCK < m)
        {
            perf_status   = rocblas_status_perf_degraded;
            supplied_invA = nullptr;
        }

        // Only allocate bytes for invA if supplied_invA == nullptr or supplied_invA_size is too small
        size_t invA_bytes = supplied_invA ? 0 : sizeof(T) * BLOCK * m;

        // Temporary solution vector
        size_t x_temp_bytes   = sizeof(T) * m;

        // When m < BLOCK, C is unnecessary for trtri
        size_t c_temp_bytes   = m < BLOCK ? 0 : sizeof(T) * BLOCK / 4 * m;

        // For the remainder of size BLOCK, we need C_temp in case m % BLOCK != 0
        size_t remainder_bytes = sizeof(T) * NB * BLOCK * 2;
        c_temp_bytes = max(c_temp_bytes, remainder_bytes);

        // X and C temporaries can share space, so the maximum size of the two is allocated
        size_t x_c_temp_bytes = max(x_temp_bytes, c_temp_bytes);

        // If this is a device memory size query, set optimal size and return changed status
        if(handle->is_device_memory_size_query())
            return handle->set_optimal_device_memory_size(x_c_temp_bytes, invA_bytes);

        // Attempt to allocate optimal memory size, returning error if failure
        auto mem = handle->device_malloc(x_c_temp_bytes, invA_bytes);
        if(!mem)
            return rocblas_status_memory_error;

        // Get pointers to allocated device memory
        // Note: Order of pointers in std::tie(...) must match order of sizes in handle->device_malloc(...)
        void* x_temp;
        void* invA;
        std::tie(x_temp, invA) = mem;

        // Temporarily switch to host pointer mode, restoring on return
        auto saved_pointer_mode = handle->push_pointer_mode(rocblas_pointer_mode_host);

        rocblas_status status;
        if(supplied_invA)
        {
            invA = const_cast<T*>(supplied_invA);
        }
        else
        {
            // batched trtri invert diagonal part (BLOCK*BLOCK) of A into invA
            auto c_temp = x_temp; // Uses same memory as x_temp
            status      = rocblas_trtri_trsm_template<BLOCK>(
                handle, (T*)c_temp, uplo, diag, m, A, lda, (T*)invA);
            if(status != rocblas_status_success)
                return status;
        }


        if(transA == rocblas_operation_conjugate_transpose)
            transA = rocblas_operation_transpose;

        // TODO: workaround to fix negative incx issue
        rocblas_int abs_incx = incx < 0 ? -incx : incx;
        if(incx < 0)
            flip_vector(handle, B, m, abs_incx);

        if(m % BLOCK == 0 && 0)
        {
            status = special_trsv_template<BLOCK>(
                handle, uplo, transA, diag, m, A, lda, B, abs_incx, (const T*)invA, (T*)x_temp);
            if(status != rocblas_status_success)
                return status;

            // TODO: workaround to fix negative incx issue
            if(incx < 0)
                flip_vector(handle, B, m, abs_incx);
        }
        else
        {
            status = rocblas_trsv_left<BLOCK>(
                handle, uplo, transA, m, A, lda, B, abs_incx, (const T*)invA, (T*)x_temp);
            if(status != rocblas_status_success)
                return status;

            // copy solution X into B
            // TODO: workaround to fix negative incx issue
            strided_vector_copy(handle,
                                B,
                                abs_incx,
                                incx < 0 ? (T*)x_temp + m - 1 : (T*)x_temp,
                                incx < 0 ? -1 : 1,
                                m);
        }

        return perf_status;
    }

} // namespace

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_strsv(rocblas_handle    handle,
                             rocblas_fill      uplo,
                             rocblas_operation transA,
                             rocblas_diagonal  diag,
                             rocblas_int       m,
                             const float*      A,
                             rocblas_int       lda,
                             float*            x,
                             rocblas_int       incx)
{
    return rocblas_trsv_ex_template<STRSV_BLOCK>(handle, uplo, transA, diag, m, A, lda, x, incx);
}

rocblas_status rocblas_dtrsv(rocblas_handle    handle,
                             rocblas_fill      uplo,
                             rocblas_operation transA,
                             rocblas_diagonal  diag,
                             rocblas_int       m,
                             const double*     A,
                             rocblas_int       lda,
                             double*           x,
                             rocblas_int       incx)
{
    return rocblas_trsv_ex_template<DTRSV_BLOCK>(handle, uplo, transA, diag, m, A, lda, x, incx);
}

rocblas_status rocblas_trsv_ex(rocblas_handle    handle,
                               rocblas_fill      uplo,
                               rocblas_operation transA,
                               rocblas_diagonal  diag,
                               rocblas_int       m,
                               const void*       A,
                               rocblas_int       lda,
                               void*             x,
                               rocblas_int       incx,
                               const void*       invA,
                               rocblas_int       invA_size,
                               rocblas_datatype  compute_type)

{
    switch(compute_type)
    {
    case rocblas_datatype_f64_r:
        return rocblas_trsv_ex_template<DTRSV_BLOCK>(handle,
                                                     uplo,
                                                     transA,
                                                     diag,
                                                     m,
                                                     static_cast<const double*>(A),
                                                     lda,
                                                     static_cast<double*>(x),
                                                     incx,
                                                     static_cast<const double*>(invA),
                                                     invA_size);

    case rocblas_datatype_f32_r:
        return rocblas_trsv_ex_template<STRSV_BLOCK>(handle,
                                                     uplo,
                                                     transA,
                                                     diag,
                                                     m,
                                                     static_cast<const float*>(A),
                                                     lda,
                                                     static_cast<float*>(x),
                                                     incx,
                                                     static_cast<const float*>(invA),
                                                     invA_size);

    default:
        return rocblas_status_not_implemented;
    }
}

} // extern "C"
