/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>

#include "rocblas.h"
#include "status.h"
#include "definitions.h"
#include "gemm.hpp"
#include "trtri_trsm.hpp"
#include "rocblas_unique_ptr.hpp"
#include "handle.h"
#include "logging.h"
#include "utility.h"

namespace {

#define A(ii, jj) (A + (ii) + (jj)*lda)
#define B(ii, jj) (B + (ii) + (jj)*ldb)
#define X(ii, jj) (X + (ii) + (jj)*m)
#define invA(ii) (invA + (ii)*BLOCK)

static constexpr size_t WORKBUF_TRSM_A_BLKS     = 10;
static constexpr size_t WORKBUF_TRSM_B_MIN_CHNK = 1024;
static constexpr size_t WORKBUF_TRSM_INVA_SZ    = 128 * 128 * 10;
static constexpr size_t WORKBUF_TRSM_INVA_C_SZ  = 128 * 128 * 10 / 2;
static constexpr size_t WORKBUF_TRSV_X_SZ       = 131072;
static constexpr size_t WORKBUF_TRSV_ALPHA_SZ   = 1;

static size_t getenv_value(const char* name, size_t min)
{
    const char* env = getenv(name);
    size_t val;
    return env && sscanf(env, "%zu", &val) == 1 && val > min ? val : min;
}

static size_t get_trsm_B_chnk()
{
    static size_t val = getenv_value("WORKBUF_TRSM_B_CHNK", WORKBUF_TRSM_B_MIN_CHNK);
    return val;
}

template <typename>
static constexpr char rocblas_trsm_name[] = "unknown";
template <>
static constexpr char rocblas_trsm_name<float>[] = "rocblas_strsm";
template <>
static constexpr char rocblas_trsm_name<double>[] = "rocblas_dtrsm";

/* ===============left==================================================== */

template <rocblas_int BLOCK, typename T>
rocblas_status rocblas_trsm_left(rocblas_handle handle,
                                 rocblas_fill uplo,
                                 rocblas_operation transA,
                                 rocblas_int m,
                                 rocblas_int n,
                                 const T* alpha,
                                 const T* A,
                                 rocblas_int lda,
                                 T* B,
                                 rocblas_int ldb,
                                 const T* invA,
                                 T* X)
{

    static constexpr T negative_one = -1;
    static constexpr T one          = 1;
    static constexpr T zero         = 0;

    rocblas_int i, jb;

    // transB is always non-transpose
    static constexpr rocblas_operation transB = rocblas_operation_none;

    if(transA == transB)
    {
        if(uplo == rocblas_fill_lower)
        {
            // left, lower no-transpose
            jb = min(BLOCK, m);
            rocblas_gemm_template<T>(
                handle, transA, transB, jb, n, jb, alpha, invA, BLOCK, B, ldb, &zero, X, m);
            if(BLOCK < m)
            {
                rocblas_gemm_template<T>(handle,
                                         transA,
                                         transB,
                                         m - BLOCK,
                                         n,
                                         BLOCK,
                                         &negative_one,
                                         A(BLOCK, 0),
                                         lda,
                                         X,
                                         m,
                                         alpha,
                                         B(BLOCK, 0),
                                         ldb);
                // remaining blocks
                for(i = BLOCK; i < m; i += BLOCK)
                {
                    jb = min(m - i, BLOCK);

                    rocblas_gemm_template<T>(handle,
                                             transA,
                                             transB,
                                             jb,
                                             n,
                                             jb,
                                             &one,
                                             invA(i),
                                             BLOCK,
                                             B(i, 0),
                                             ldb,
                                             &zero,
                                             X(i, 0),
                                             m);
                    if(i + BLOCK >= m) // this condition is not necessary at all and can be changed
                                       // as if (i+BLOCK<m)
                        break;

                    rocblas_gemm_template<T>(handle,
                                             transA,
                                             transB,
                                             m - i - BLOCK,
                                             n,
                                             BLOCK,
                                             &negative_one,
                                             A(i + BLOCK, i),
                                             lda,
                                             X(i, 0),
                                             m,
                                             &one,
                                             B(i + BLOCK, 0),
                                             ldb);
                }
            }

#if 0
            for(i = 0; i < m; i += BLOCK)
            {
                jb     = min(m - i, BLOCK);
                T* tmp = (i == 0) ? alpha : one;
                rocblas_gemm_template<T>(handle,
                                         transA,
                                         transB,
                                         jb,
                                         n,
                                         jb,
                                         tmp,
                                         invA(i),
                                         BLOCK,
                                         B(i, 0),
                                         ldb,
                                         &zero,
                                         X(i, 0),
                                         ldb);
                if(i + BLOCK < m)
                {
                    rocblas_gemm_template<T>(handle,
                                             transA,
                                             transB,
                                             m - i - BLOCK,
                                             n,
                                             BLOCK,
                                             &negative_one,
                                             A(i + BLOCK, i),
                                             lda,
                                             X(i, 0),
                                             ldb,
                                             tmp,
                                             B(i + BLOCK, 0),
                                             ldb);
                }
            }

#endif
        }
        else
        {
            // left, upper no-transpose
            jb = (m % BLOCK == 0) ? BLOCK : (m % BLOCK);
            i  = m - jb;

            // if m=n=35=lda=ldb, BLOCK =32, then jb = 3, i = 32; {3, 35, 3, 32, 35, 35}
            rocblas_gemm_template<T>(handle,
                                     transA,
                                     transB,
                                     jb,
                                     n,
                                     jb,
                                     alpha,
                                     invA(i),
                                     BLOCK,
                                     B(i, 0),
                                     ldb,
                                     &zero,
                                     X(i, 0),
                                     m);
            if(i - BLOCK >= 0)
            {
                rocblas_gemm_template<T>(handle,
                                         transA,
                                         transB,
                                         i,
                                         n,
                                         jb,
                                         &negative_one,
                                         A(0, i),
                                         lda,
                                         X(i, 0),
                                         m,
                                         alpha,
                                         B,
                                         ldb);

                // remaining blocks
                for(i = m - jb - BLOCK; i >= 0; i -= BLOCK)
                {
                    //{32, 35, 32, 32, 35, 35}
                    rocblas_gemm_template<T>(handle,
                                             transA,
                                             transB,
                                             BLOCK,
                                             n,
                                             BLOCK,
                                             &one,
                                             invA(i),
                                             BLOCK,
                                             B(i, 0),
                                             ldb,
                                             &zero,
                                             X(i, 0),
                                             m);
                    if(i - BLOCK < 0)
                        break;
                    rocblas_gemm_template<T>(handle,
                                             transA,
                                             transB,
                                             i,
                                             n,
                                             BLOCK,
                                             &negative_one,
                                             A(0, i),
                                             lda,
                                             X(i, 0),
                                             m,
                                             &one,
                                             B,
                                             ldb);
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
            rocblas_gemm_template<T>(handle,
                                     transA,
                                     transB,
                                     jb,
                                     n,
                                     jb,
                                     alpha,
                                     invA(i),
                                     BLOCK,
                                     B(i, 0),
                                     ldb,
                                     &zero,
                                     X(i, 0),
                                     m);
            if(i - BLOCK >= 0)
            {
                rocblas_gemm_template<T>(handle,
                                         transA,
                                         transB,
                                         i,
                                         n,
                                         jb,
                                         &negative_one,
                                         A(i, 0),
                                         lda,
                                         X(i, 0),
                                         m,
                                         alpha,
                                         B,
                                         ldb);

                // remaining blocks
                for(i = m - jb - BLOCK; i >= 0; i -= BLOCK)
                {
                    rocblas_gemm_template<T>(handle,
                                             transA,
                                             transB,
                                             BLOCK,
                                             n,
                                             BLOCK,
                                             &one,
                                             invA(i),
                                             BLOCK,
                                             B(i, 0),
                                             ldb,
                                             &zero,
                                             X(i, 0),
                                             m);
                    if(i - BLOCK < 0)
                        break;
                    rocblas_gemm_template<T>(handle,
                                             transA,
                                             transB,
                                             i,
                                             n,
                                             BLOCK,
                                             &negative_one,
                                             A(i, 0),
                                             lda,
                                             X(i, 0),
                                             m,
                                             &one,
                                             B,
                                             ldb);
                }
            }
        }
        else
        {
            // left, upper transpose
            jb = min(BLOCK, m);
            rocblas_gemm_template<T>(
                handle, transA, transB, jb, n, jb, alpha, invA, BLOCK, B, ldb, &zero, X, m);
            if(BLOCK < m)
            {
                rocblas_gemm_template<T>(handle,
                                         transA,
                                         transB,
                                         m - BLOCK,
                                         n,
                                         BLOCK,
                                         &negative_one,
                                         A(0, BLOCK),
                                         lda,
                                         X,
                                         m,
                                         alpha,
                                         B(BLOCK, 0),
                                         ldb);

                // remaining blocks
                for(i = BLOCK; i < m; i += BLOCK)
                {
                    jb = min(m - i, BLOCK);
                    rocblas_gemm_template<T>(handle,
                                             transA,
                                             transB,
                                             jb,
                                             n,
                                             jb,
                                             &one,
                                             invA(i),
                                             BLOCK,
                                             B(i, 0),
                                             ldb,
                                             &zero,
                                             X(i, 0),
                                             m);
                    if(i + BLOCK >= m)
                        break;
                    rocblas_gemm_template<T>(handle,
                                             transA,
                                             transB,
                                             m - i - BLOCK,
                                             n,
                                             BLOCK,
                                             &negative_one,
                                             A(i, i + BLOCK),
                                             lda,
                                             X(i, 0),
                                             m,
                                             &one,
                                             B(i + BLOCK, 0),
                                             ldb);
                }
            }
        }
    } // transpose

    return rocblas_status_success;
}

/* ===============right==================================================== */

template <rocblas_int BLOCK, typename T>
rocblas_status rocblas_trsm_right(rocblas_handle handle,
                                  rocblas_fill uplo,
                                  rocblas_operation transA,
                                  rocblas_int m,
                                  rocblas_int n,
                                  const T* alpha,
                                  const T* A,
                                  rocblas_int lda,
                                  T* B,
                                  rocblas_int ldb,
                                  const T* invA,
                                  T* X)
{
    static constexpr T negative_one = -1;
    static constexpr T one          = 1;
    static constexpr T zero         = 0;

    rocblas_int i, jb;

    // transB is always non-transpose
    static constexpr rocblas_operation transB = rocblas_operation_none;

    if(transA == transB)
    {
        if(uplo == rocblas_fill_lower)
        {
            // right, lower no-transpose
            jb = (n % BLOCK == 0) ? BLOCK : (n % BLOCK);
            i  = n - jb;
            rocblas_gemm_template<T>(handle,
                                     transB,
                                     transA,
                                     m,
                                     jb,
                                     jb,
                                     alpha,
                                     B(0, i),
                                     ldb,
                                     invA(i),
                                     BLOCK,
                                     &zero,
                                     X(0, i),
                                     m);
            if(i - BLOCK >= 0)
            {
                rocblas_gemm_template<T>(handle,
                                         transB,
                                         transA,
                                         m,
                                         i,
                                         jb,
                                         &negative_one,
                                         X(0, i),
                                         m,
                                         A(i, 0),
                                         lda,
                                         alpha,
                                         B,
                                         ldb);

                // remaining blocks
                for(i = n - jb - BLOCK; i >= 0; i -= BLOCK)
                {
                    rocblas_gemm_template<T>(handle,
                                             transB,
                                             transA,
                                             m,
                                             BLOCK,
                                             BLOCK,
                                             &one,
                                             B(0, i),
                                             ldb,
                                             invA(i),
                                             BLOCK,
                                             &zero,
                                             X(0, i),
                                             m);
                    if(i - BLOCK < 0)
                        break;
                    rocblas_gemm_template<T>(handle,
                                             transB,
                                             transA,
                                             m,
                                             i,
                                             BLOCK,
                                             &negative_one,
                                             X(0, i),
                                             m,
                                             A(i, 0),
                                             lda,
                                             &one,
                                             B,
                                             ldb);
                }
            }
        }
        else
        {
            // right, upper no-transpose
            jb = min(BLOCK, n);
            rocblas_gemm_template<T>(
                handle, transB, transA, m, jb, jb, alpha, B, ldb, invA, BLOCK, &zero, X, m);
            if(BLOCK < n)
            {
                rocblas_gemm_template<T>(handle,
                                         transB,
                                         transA,
                                         m,
                                         n - BLOCK,
                                         BLOCK,
                                         &negative_one,
                                         X,
                                         m,
                                         A(0, BLOCK),
                                         lda,
                                         alpha,
                                         B(0, BLOCK),
                                         ldb);

                // remaining blocks
                for(i = BLOCK; i < n; i += BLOCK)
                {
                    jb = min(BLOCK, n - i);
                    rocblas_gemm_template<T>(handle,
                                             transB,
                                             transA,
                                             m,
                                             jb,
                                             jb,
                                             &one,
                                             B(0, i),
                                             ldb,
                                             invA(i),
                                             BLOCK,
                                             &zero,
                                             X(0, i),
                                             m);
                    if(i + BLOCK >= n)
                        break;
                    rocblas_gemm_template<T>(handle,
                                             transB,
                                             transA,
                                             m,
                                             n - i - BLOCK,
                                             BLOCK,
                                             &negative_one,
                                             X(0, i),
                                             m,
                                             A(i, i + BLOCK),
                                             lda,
                                             &one,
                                             B(0, i + BLOCK),
                                             ldb);
                }
            }
        }
    }
    else
    { // transA == rocblas_operation_transpose || transA == rocblas_operation_conjugate_transpose
        if(uplo == rocblas_fill_lower)
        {
            // right, lower transpose
            jb = min(BLOCK, n);
            rocblas_gemm_template<T>(
                handle, transB, transA, m, jb, jb, alpha, B, ldb, invA, BLOCK, &zero, X, m);
            if(BLOCK < n)
            {
                rocblas_gemm_template<T>(handle,
                                         transB,
                                         transA,
                                         m,
                                         n - BLOCK,
                                         BLOCK,
                                         &negative_one,
                                         X,
                                         m,
                                         A(BLOCK, 0),
                                         lda,
                                         alpha,
                                         B(0, BLOCK),
                                         ldb);

                // remaining blocks
                for(i = BLOCK; i < n; i += BLOCK)
                {
                    jb = min(BLOCK, n - i);
                    rocblas_gemm_template<T>(handle,
                                             transB,
                                             transA,
                                             m,
                                             jb,
                                             jb,
                                             &one,
                                             B(0, i),
                                             ldb,
                                             invA(i),
                                             BLOCK,
                                             &zero,
                                             X(0, i),
                                             m);
                    if(i + BLOCK >= n)
                        break;
                    rocblas_gemm_template<T>(handle,
                                             transB,
                                             transA,
                                             m,
                                             n - i - BLOCK,
                                             BLOCK,
                                             &negative_one,
                                             X(0, i),
                                             m,
                                             A(BLOCK + i, i),
                                             lda,
                                             &one,
                                             B(0, i + BLOCK),
                                             ldb);
                }
            }
        }
        else
        {
            // right, upper transpose
            jb = n % BLOCK == 0 ? BLOCK : n % BLOCK;
            i  = n - jb;
            rocblas_gemm_template<T>(handle,
                                     transB,
                                     transA,
                                     m,
                                     jb,
                                     jb,
                                     alpha,
                                     B(0, i),
                                     ldb,
                                     invA(i),
                                     BLOCK,
                                     &zero,
                                     X(0, i),
                                     m);
            if(i - BLOCK >= 0)
            {
                rocblas_gemm_template<T>(handle,
                                         transB,
                                         transA,
                                         m,
                                         i,
                                         jb,
                                         &negative_one,
                                         X(0, i),
                                         m,
                                         A(0, i),
                                         lda,
                                         alpha,
                                         B,
                                         ldb);

                // remaining blocks
                for(i = n - jb - BLOCK; i >= 0; i -= BLOCK)
                {
                    rocblas_gemm_template<T>(handle,
                                             transB,
                                             transA,
                                             m,
                                             BLOCK,
                                             BLOCK,
                                             &one,
                                             B(0, i),
                                             ldb,
                                             invA(i),
                                             BLOCK,
                                             &zero,
                                             X(0, i),
                                             m);
                    if(i - BLOCK < 0)
                        break;
                    rocblas_gemm_template<T>(handle,
                                             transB,
                                             transA,
                                             m,
                                             i,
                                             BLOCK,
                                             &negative_one,
                                             X(0, i),
                                             m,
                                             A(0, i),
                                             lda,
                                             &one,
                                             B,
                                             ldb);
                }
            }
        }
    } // tranpsose

    return rocblas_status_success;
}

__global__ void copy_void_ptr_matrix_trsm(rocblas_int rows,
                                          rocblas_int cols,
                                          rocblas_int elem_size,
                                          const void* a,
                                          rocblas_int lda,
                                          void* b,
                                          rocblas_int ldb)
{
    size_t tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    size_t ty = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if(tx < rows && ty < cols)
        memcpy((char*)b + (tx + ldb * ty) * elem_size,
               (char*)a + (tx + lda * ty) * elem_size,
               elem_size);
}

template <typename T>
void copy_block_unit(hipStream_t rocblas_stream,
                     rocblas_int m,
                     rocblas_int n,
                     const void* src,
                     rocblas_int src_ld,
                     void* dst,
                     rocblas_int dst_ld)
{
    rocblas_int blocksX = (m - 1) / 128 + 1; // parameters for device kernel
    rocblas_int blocksY = (n - 1) / 8 + 1;
    dim3 grid(blocksX, blocksY);
    dim3 threads(128, 8);

    hipLaunchKernelGGL(copy_void_ptr_matrix_trsm,
                       grid,
                       threads,
                       0,
                       rocblas_stream,
                       m,
                       n,
                       sizeof(T),
                       src,
                       src_ld,
                       dst,
                       dst_ld);
}

template <
    rocblas_int BLOCK,
    typename T,
    typename = typename std::enable_if<std::is_same<T, float>{} || std::is_same<T, double>{}>::type>
rocblas_status special_trsm_template(rocblas_handle handle,
                                     rocblas_side side,
                                     rocblas_fill uplo,
                                     rocblas_operation transA,
                                     rocblas_diagonal diag,
                                     rocblas_int m,
                                     rocblas_int n,
                                     const T* alpha,
                                     const T* A,
                                     rocblas_int lda,
                                     T* B,
                                     rocblas_int ldb,
                                     const T* invA,
                                     size_t B_chunk,
                                     T* x_temp)
{
    hipStream_t rocblas_stream;
    RETURN_IF_ROCBLAS_ERROR(rocblas_get_stream(handle, &rocblas_stream));

    const rocblas_int k = side == rocblas_side_left ? m : n;

    int R                           = k / BLOCK;
    static constexpr T zero         = 0;
    static constexpr T one          = 1;
    static constexpr T negative_one = -1;

    rocblas_int bsize            = side == rocblas_side_left ? n : m;
    int W                        = 1 + (bsize - 1) / B_chunk;
    static const bool arch_lt906 = handle->device_arch_id() < 906;

    for(int w = 0; w < W; w++)
    {
        if(side == rocblas_side_left)
        {
            T* Bw     = B + (size_t)w * B_chunk * (size_t)ldb;
            int width = bsize > (w + 1) * B_chunk ? B_chunk : bsize - w * B_chunk;

            for(int r = 0; r < R; r++)
            {
                int q = R - 1 - r;

                int j =
                    (uplo == rocblas_fill_lower && transA == rocblas_operation_none) ||
                            (uplo == rocblas_fill_upper && transA == rocblas_operation_transpose)
                        ? r
                        : q;

                // copy a BLOCK*n piece we are solving at a time
                if(r == 0 || arch_lt906)
                    copy_block_unit<T>(
                        rocblas_stream, BLOCK, width, Bw + j * BLOCK, ldb, x_temp, BLOCK);

                if(r > 0)
                {
                    const T* A_current = nullptr;
                    T* B_current       = nullptr;

                    if(uplo == rocblas_fill_upper && transA == rocblas_operation_transpose)
                    {
                        A_current = A + r * BLOCK * lda;
                        B_current = Bw;
                    }
                    else if(uplo == rocblas_fill_lower && transA == rocblas_operation_none)
                    {
                        A_current = A + r * BLOCK;
                        B_current = Bw;
                    }
                    else if(uplo == rocblas_fill_lower && transA == rocblas_operation_transpose)
                    {
                        A_current = A + q * BLOCK * lda + (q + 1) * BLOCK;
                        B_current = Bw + (q + 1) * BLOCK;
                    }
                    else // uplo == rocblas_fill_upper && transA == rocblas_operation_none
                    {
                        A_current = A + (q + 1) * BLOCK * lda + q * BLOCK;
                        B_current = Bw + (q + 1) * BLOCK;
                    }

                    if(arch_lt906)
                    {
                        rocblas_gemm_template<T>(handle,
                                                 transA,
                                                 rocblas_operation_none,
                                                 BLOCK,
                                                 width,
                                                 r * BLOCK,
                                                 &negative_one,
                                                 A_current,
                                                 lda,
                                                 B_current,
                                                 ldb,
                                                 alpha,
                                                 (T*)x_temp,
                                                 BLOCK);
                    }
                    else
                    {
                        rocblas_datatype compute_type;
                        rocblas_gemm_algo algo = rocblas_gemm_algo_standard;
                        int32_t solution_index = 0;
                        uint32_t flags         = 0;

                        if(std::is_same<T, float>::value)
                            compute_type = rocblas_datatype_f32_r;
                        else
                            compute_type = rocblas_datatype_f64_r;

                        rocblas_gemm_ex(handle,
                                        transA,
                                        rocblas_operation_none,
                                        BLOCK,
                                        width,
                                        r * BLOCK,
                                        &negative_one,
                                        A_current,
                                        compute_type,
                                        lda,
                                        B_current,
                                        compute_type,
                                        ldb,
                                        alpha,
                                        Bw + j * BLOCK,
                                        compute_type,
                                        ldb,
                                        (T*)x_temp,
                                        compute_type,
                                        BLOCK,
                                        compute_type,
                                        algo,
                                        solution_index,
                                        flags);
                    }
                }

                const T* theta = r == 0 ? alpha : &one;
                rocblas_gemm_template<T>(handle,
                                         transA,
                                         rocblas_operation_none,
                                         BLOCK,
                                         width,
                                         BLOCK,
                                         theta,
                                         (T*)invA + j * BLOCK * BLOCK,
                                         BLOCK,
                                         (T*)x_temp,
                                         BLOCK,
                                         &zero,
                                         Bw + j * BLOCK,
                                         ldb);
            }
        }
        else
        {
            T* Bw     = B + (size_t)w * B_chunk;
            int width = bsize > (w + 1) * B_chunk ? B_chunk : bsize - w * B_chunk;

            for(int r = 0; r < R; r++)
            {
                int q = R - 1 - r;

                int j = (uplo == rocblas_fill_lower && transA == rocblas_operation_transpose) ||
                                (uplo == rocblas_fill_upper && transA == rocblas_operation_none)
                            ? r
                            : q;

                // copy a m*BLOCK piece we are solving at a time
                if(r == 0 || arch_lt906)
                    copy_block_unit<T>(
                        rocblas_stream, width, BLOCK, Bw + j * BLOCK * ldb, ldb, x_temp, width);

                if(r > 0)
                {
                    const T* A_current = nullptr;
                    T* B_current       = nullptr;

                    if(uplo == rocblas_fill_lower && transA == rocblas_operation_transpose)
                    {
                        A_current = A + r * BLOCK;
                        B_current = Bw;
                    }
                    else if(uplo == rocblas_fill_upper && transA == rocblas_operation_none)
                    {
                        A_current = A + r * BLOCK * lda;
                        B_current = Bw;
                    }
                    else if(uplo == rocblas_fill_upper && transA == rocblas_operation_transpose)
                    {
                        A_current = A + (q + 1) * BLOCK * lda + q * BLOCK;
                        B_current = Bw + (size_t)(q + 1) * BLOCK * (size_t)ldb;
                    }
                    else // uplo == rocblas_fill_lower && transA == rocblas_operation_none
                    {
                        A_current = A + q * BLOCK * lda + (q + 1) * BLOCK;
                        B_current = Bw + (size_t)(q + 1) * BLOCK * (size_t)ldb;
                    }

                    if(arch_lt906)
                    {
                        rocblas_gemm_template<T>(handle,
                                                 rocblas_operation_none,
                                                 transA,
                                                 width,
                                                 BLOCK,
                                                 r * BLOCK,
                                                 &negative_one,
                                                 B_current,
                                                 ldb,
                                                 A_current,
                                                 lda,
                                                 alpha,
                                                 (T*)x_temp,
                                                 width);
                    }
                    else
                    {
                        rocblas_datatype compute_type;
                        rocblas_gemm_algo algo = rocblas_gemm_algo_standard;
                        int32_t solution_index = 0;
                        uint32_t flags         = 0;

                        if(std::is_same<T, float>::value)
                            compute_type = rocblas_datatype_f32_r;
                        else
                            compute_type = rocblas_datatype_f64_r;

                        rocblas_gemm_ex(handle,
                                        rocblas_operation_none,
                                        transA,
                                        width,
                                        BLOCK,
                                        r * BLOCK,
                                        &negative_one,
                                        B_current,
                                        compute_type,
                                        ldb,
                                        A_current,
                                        compute_type,
                                        lda,
                                        alpha,
                                        Bw + j * BLOCK * ldb,
                                        compute_type,
                                        ldb,
                                        (T*)x_temp,
                                        compute_type,
                                        width,
                                        compute_type,
                                        algo,
                                        solution_index,
                                        flags);
                    }
                }

                const T* theta = r == 0 ? alpha : &one;

                rocblas_gemm_template<T>(handle,
                                         rocblas_operation_none,
                                         transA,
                                         width,
                                         BLOCK,
                                         BLOCK,
                                         theta,
                                         (T*)x_temp,
                                         width,
                                         (T*)invA + j * BLOCK * BLOCK,
                                         BLOCK,
                                         &zero,
                                         Bw + j * BLOCK * ldb,
                                         ldb);
            }
        }
    }

    return rocblas_status_success;
}

} // namespace

/* ============================================================================================ */

template <rocblas_int BLOCK, typename T>
rocblas_status rocblas_trsm_template(rocblas_handle handle,
                                     rocblas_side side,
                                     rocblas_fill uplo,
                                     rocblas_operation transA,
                                     rocblas_diagonal diag,
                                     rocblas_int m,
                                     rocblas_int n,
                                     const T* alpha,
                                     const T* A,
                                     rocblas_int lda,
                                     T* B,
                                     rocblas_int ldb)
{
    if(!handle)
        return rocblas_status_invalid_handle;

    if(!handle->is_device_memory_size_query())
    {
        if(!alpha || !A || !B)
            return rocblas_status_invalid_pointer;

        auto layer_mode = handle->layer_mode;
        if(layer_mode & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench |
                         rocblas_layer_mode_log_profile))
        {
            auto side_letter   = rocblas_side_letter(side);
            auto uplo_letter   = rocblas_fill_letter(uplo);
            auto transA_letter = rocblas_transpose_letter(transA);
            auto diag_letter   = rocblas_diag_letter(diag);

            if(handle->pointer_mode == rocblas_pointer_mode_host)
            {
                if(layer_mode & rocblas_layer_mode_log_trace)
                    log_trace(handle,
                              rocblas_trsm_name<T>,
                              side,
                              uplo,
                              transA,
                              diag,
                              m,
                              n,
                              *alpha,
                              A,
                              lda,
                              B,
                              ldb);

                if(layer_mode & rocblas_layer_mode_log_bench)
                {
                    log_bench(handle,
                              "./rocblas-bench -f trsm -r",
                              rocblas_precision_string<T>,
                              "--side",
                              side_letter,
                              "--uplo",
                              uplo_letter,
                              "--transposeA",
                              transA_letter,
                              "--diag",
                              diag_letter,
                              "-m",
                              m,
                              "-n",
                              n,
                              "--alpha",
                              *alpha,
                              "--lda",
                              lda,
                              "--ldb",
                              ldb);
                }
            }
            else
            {
                if(layer_mode & rocblas_layer_mode_log_trace)
                    log_trace(handle,
                              rocblas_trsm_name<T>,
                              side,
                              uplo,
                              transA,
                              diag,
                              m,
                              n,
                              alpha,
                              A,
                              lda,
                              B,
                              ldb);
            }

            if(layer_mode & rocblas_layer_mode_log_profile)
            {
                log_profile(handle,
                            rocblas_trsm_name<T>,
                            "side",
                            side_letter,
                            "uplo",
                            uplo_letter,
                            "transA",
                            transA_letter,
                            "diag",
                            diag_letter,
                            "m",
                            m,
                            "n",
                            n,
                            "lda",
                            lda,
                            "ldb",
                            ldb);
            }
        }
    }

    if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
        return rocblas_status_not_implemented;

    // A is of size lda*k
    rocblas_int k = side == rocblas_side_left ? m : n;

    if(m < 0 || n < 0 || lda < k || ldb < m)
        return rocblas_status_invalid_size;

    // quick return if possible.
    if(!m || !n)
        return handle->is_device_memory_size_query() ? rocblas_status_size_unchanged
                                                     : rocblas_status_success;

    if(transA == rocblas_operation_conjugate_transpose)
        transA = rocblas_operation_transpose;

    static constexpr size_t ldInvA = BLOCK;
    size_t invA_size               = sizeof(T) * ldInvA * k;

    if(k % BLOCK == 0 && k <= BLOCK * WORKBUF_TRSM_A_BLKS)
    {
        size_t B_chnk_size = get_trsm_B_chnk();
        size_t invA_C_size = WORKBUF_TRSM_INVA_C_SZ * sizeof(T);
        size_t Y_size      = B_chnk_size * BLOCK * sizeof(T);
        size_t max_size    = invA_C_size > Y_size ? invA_C_size : Y_size;

        if(handle->is_device_memory_size_query())
            return handle->set_optimal_device_memory_size(invA_size, max_size);

        auto device_mem = handle->device_memory_alloc(invA_size, max_size);
        if(!device_mem)
            return rocblas_status_memory_error;

        void *invA, *x_temp;
        std::tie(invA, x_temp) = device_mem;

        void* invA_C          = x_temp;
        rocblas_status status = rocblas_trtri_trsm_template<T, BLOCK>(
            handle, (T*)invA_C, uplo, diag, k, A, lda, (T*)invA);
        if(status != rocblas_status_success)
            return status;

        return special_trsm_template<BLOCK>(handle,
                                            side,
                                            uplo,
                                            transA,
                                            diag,
                                            m,
                                            n,
                                            alpha,
                                            A,
                                            lda,
                                            B,
                                            ldb,
                                            (T*)invA,
                                            B_chnk_size,
                                            (T*)x_temp);
    }
    else
    {
        size_t invA_C_size = k < BLOCK ? 0 : (sizeof(T) * (BLOCK / 2) * (BLOCK / 2)) * (k / BLOCK);
        size_t X_size      = sizeof(T) * m * n;
        size_t max_size    = invA_C_size > X_size ? invA_C_size : X_size;

        if(handle->is_device_memory_size_query())
            return handle->set_optimal_device_memory_size(invA_size, max_size);

        auto device_mem = handle->device_memory_alloc(invA_size, max_size);
        if(!device_mem)
            return rocblas_status_memory_error;

        void *invA, *X;
        std::tie(invA, X) = device_mem;

        // batched trtri invert diagonal part (BLOCK*BLOCK) of A into invA
        rocblas_status status =
            rocblas_trtri_trsm_template<T, BLOCK>(handle, (T*)X, uplo, diag, k, A, lda, (T*)invA);

        if(status != rocblas_status_success)
            return status;

        if(side == rocblas_side_left)
        {
            status = rocblas_trsm_left<BLOCK>(
                handle, uplo, transA, m, n, alpha, A, lda, B, ldb, (T*)invA, (T*)X);
        }
        else
        { // side == rocblas_side_right
            status = rocblas_trsm_right<BLOCK>(
                handle, uplo, transA, m, n, alpha, A, lda, B, ldb, (T*)invA, (T*)X);
        }

        if(status != rocblas_status_success)
            return status;

        hipStream_t rocblas_stream;
        RETURN_IF_ROCBLAS_ERROR(rocblas_get_stream(handle, &rocblas_stream));

        // copy solution X into B
        rocblas_int blocksX = (m - 1) / 128 + 1; // parameters for device kernel
        rocblas_int blocksY = (n - 1) / 8 + 1;

        hipLaunchKernelGGL(copy_void_ptr_matrix_trsm,
                           dim3(blocksX, blocksY),
                           dim3(128, 8),
                           0,
                           rocblas_stream,
                           m,
                           n,
                           sizeof(T),
                           (T*)X,
                           m,
                           B,
                           ldb);

        return rocblas_status_success;
    }
}

/* ============================================================================================ */

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_strsm(rocblas_handle handle,
                             rocblas_side side,
                             rocblas_fill uplo,
                             rocblas_operation transA,
                             rocblas_diagonal diag,
                             rocblas_int m,
                             rocblas_int n,
                             const float* alpha,
                             const float* A,
                             rocblas_int lda,
                             float* B,
                             rocblas_int ldb)
{
    // shared memory usuage is (192/2)^2 * sizeof(float) = 36K. LDS is 64K per CU. Theoretically
    // you can use all 64K, but in practice no.
    static constexpr rocblas_int STRSM_BLOCK = 128;
    return rocblas_trsm_template<STRSM_BLOCK>(
        handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb);
}

rocblas_status rocblas_dtrsm(rocblas_handle handle,
                             rocblas_side side,
                             rocblas_fill uplo,
                             rocblas_operation transA,
                             rocblas_diagonal diag,
                             rocblas_int m,
                             rocblas_int n,
                             const double* alpha,
                             const double* A,
                             rocblas_int lda,
                             double* B,
                             rocblas_int ldb)
{
    // shared memory usuage is (128/2)^2 * sizeof(float) = 32K. LDS is 64K per CU. Theoretically
    // you can use all 64K, but in practice no.
    static constexpr rocblas_int DTRSM_BLOCK = 128;
    return rocblas_trsm_template<DTRSM_BLOCK>(
        handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb);
}

} // extern "C"
