/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <algorithm>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include "rocblas_test.hpp"
#include "rocblas_math.hpp"
#include "rocblas_vector.hpp"
#include <sys/param.h>
#include "utility.hpp"
#include "rocblas.hpp"
#include "cblas_interface.hpp"
#include "../../library/src/include/handle.h"
#include "../../library/src/include/utility.h"

template <typename T>
static constexpr auto precision_letter = "*";
template <>
static constexpr auto precision_letter<rocblas_half> = "h";
template <>
static constexpr auto precision_letter<float> = "s";
template <>
static constexpr auto precision_letter<double> = "d";
template <>
static constexpr auto precision_letter<rocblas_float_complex> = "c";
template <>
static constexpr auto precision_letter<rocblas_double_complex> = "z";

// replaces X in string with s, d, c, z or h depending on typename T
template <typename T>
std::string replaceX(std::string input_string)
{
    std::replace(input_string.begin(), input_string.end(), 'X', precision_letter<T>[0]);
    return input_string;
}

template <typename T>
void testing_logging()
{
    rocblas_pointer_mode test_pointer_mode = rocblas_pointer_mode_host;

    // set environment variable ROCBLAS_LAYER to turn on logging. Note that setenv
    // only has scope for this executable, so it is not necessary to save and restore
    // this environment variable
    //
    // ROCBLAS_LAYER is a bit mask:
    // ROCBLAS_LAYER = 1 turns on log_trace
    // ROCBLAS_LAYER = 2 turns on log_bench
    // ROCBLAS_LAYER = 4 turns on log_profile
    int setenv_status;

    setenv_status = setenv("ROCBLAS_LAYER", "3", true);

#ifdef GOOGLE_TEST
    EXPECT_EQ(setenv_status, 0);
#endif

    auto trace_name1 = "stream_trace_" + std::string(precision_letter<T>) + ".csv";
    // set environment variable to give pathname of for log_trace file
    setenv_status = setenv("ROCBLAS_LOG_TRACE_PATH", trace_name1.c_str(), true);

#ifdef GOOGLE_TEST
    EXPECT_EQ(setenv_status, 0);
#endif

    // set environment variable to give pathname of for log_bench file
    auto bench_name1 = "stream_bench_" + std::string(precision_letter<T>) + ".txt";
    setenv_status    = setenv("ROCBLAS_LOG_BENCH_PATH", bench_name1.c_str(), true);

#ifdef GOOGLE_TEST
    EXPECT_EQ(setenv_status, 0);
#endif

    // set environment variable to give pathname of for log_profile file
    auto profile_name1 = "stream_profile_" + std::string(precision_letter<T>) + ".yaml";
    setenv_status      = setenv("ROCBLAS_LOG_PROFILE_PATH", profile_name1.c_str(), true);

#ifdef GOOGLE_TEST
    EXPECT_EQ(setenv_status, 0);
#endif

    rocblas::reinit_logs(); // reinitialize logging with newly set environment

    //
    // call rocBLAS functions with log_trace and log_bench to output log_trace and log_bench files
    //

    rocblas_int m            = 1;
    rocblas_int n            = 1;
    rocblas_int k            = 1;
    rocblas_int incx         = 1;
    rocblas_int incy         = 1;
    rocblas_int lda          = 1;
    rocblas_int stride_a     = 1;
    rocblas_int ldb          = 1;
    rocblas_int stride_b     = 1;
    rocblas_int ldc          = 1;
    rocblas_int stride_c     = 1;
    rocblas_int ldd          = 1;
    rocblas_int stride_d     = 1;
    rocblas_int batch_count  = 1;
    T alpha                  = 1.0;
    T beta                   = 1.0;
    rocblas_operation transA = rocblas_operation_none;
    rocblas_operation transB = rocblas_operation_transpose;
    rocblas_fill uplo        = rocblas_fill_upper;
    rocblas_diagonal diag    = rocblas_diagonal_unit;
    rocblas_side side        = rocblas_side_left;

    rocblas_int safe_dim = ((m > n ? m : n) > k ? (m > n ? m : n) : k);
    rocblas_int size_x   = n * incx;
    rocblas_int size_y   = n * incy;
    rocblas_int size_a   = (lda > stride_a ? lda : stride_a) * safe_dim * batch_count;
    rocblas_int size_b   = (ldb > stride_b ? ldb : stride_b) * safe_dim * batch_count;
    rocblas_int size_c   = (ldc > stride_c ? ldc : stride_c) * safe_dim * batch_count;
    rocblas_int size_d   = (ldd > stride_d ? ldd : stride_d) * safe_dim * batch_count;

    // allocate memory on device
    device_vector<T> dx(size_x);
    device_vector<T> dy(size_y);
    device_vector<T> da(size_a);
    device_vector<T> db(size_b);
    device_vector<T> dc(size_c);
    device_vector<T> dd(size_d);
    if(!dx || !dy || !da || !db || !dc || !dd)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // enclose in {} so rocblas_local_handle destructor called as it goes out of scope
    {
        int i_result;
        T result;
        rocblas_pointer_mode mode;

        // Auxiliary functions
        rocblas_local_handle handle;

        rocblas_set_pointer_mode(handle, test_pointer_mode);
        rocblas_get_pointer_mode(handle, &mode);

        // BLAS1
        rocblas_iamax<T>(handle, n, dx, incx, &i_result);

        rocblas_iamin<T>(handle, n, dx, incx, &i_result);

        rocblas_asum<T, T>(handle, n, dx, incx, &result);

        rocblas_axpy<T>(handle, n, &alpha, dx, incx, dy, incy);

        rocblas_copy<T>(handle, n, dx, incx, dy, incy);

        rocblas_dot<T>(handle, n, dx, incx, dy, incy, &result);

        rocblas_nrm2<T, T>(handle, n, dx, incx, &result);

        rocblas_scal<T>(handle, n, &alpha, dx, incx);

        rocblas_swap<T>(handle, n, dx, incx, dy, incy);

        // BLAS2
        rocblas_ger<T>(handle, m, n, &alpha, dx, incx, dy, incy, da, lda);

        rocblas_syr<T>(handle, uplo, n, &alpha, dx, incx, da, lda);

        rocblas_gemv<T>(handle, transA, m, n, &alpha, da, lda, dx, incx, &beta, dy, incy);

        // BLAS3
        rocblas_geam<T>(handle, transA, transB, m, n, &alpha, da, lda, &beta, db, ldb, dc, ldc);

        if(BUILD_WITH_TENSILE)
        {
            /* trsm calls rocblas_get_stream and rocblas_dgemm, so test it by comparing files
               rocblas_trsm<T>(handle, side, uplo, transA, diag, m, n, &alpha, da, lda, db,
               ldb);
            */
            rocblas_gemm<T>(
                handle, transA, transB, m, n, k, &alpha, da, lda, db, ldb, &beta, dc, ldc);

            rocblas_gemm_strided_batched<T>(handle,
                                            transA,
                                            transB,
                                            m,
                                            n,
                                            k,
                                            &alpha,
                                            da,
                                            lda,
                                            stride_a,
                                            db,
                                            ldb,
                                            stride_b,
                                            &beta,
                                            dc,
                                            ldc,
                                            stride_c,
                                            batch_count);
        }

        // exclude trtri as it is an internal function
        //      rocblas_trtri<T>(handle, uplo, diag, n, da, lda, db, ldb);

        // trmm
        // tritri

        // BLAS_EX
        if(BUILD_WITH_TENSILE)
        {
            void* alpha             = 0;
            void* beta              = 0;
            float alpha_float       = 1.0;
            float beta_float        = 1.0;
            rocblas_half alpha_half = float_to_half(alpha_float);
            rocblas_half beta_half  = float_to_half(beta_float);
            double alpha_double     = static_cast<double>(alpha_float);
            double beta_double      = static_cast<double>(beta_float);
            rocblas_gemm_algo algo  = rocblas_gemm_algo_standard;
            int32_t solution_index  = 0;
            uint32_t flags          = 0;
            size_t* workspace_size  = 0;
            void* workspace         = 0;
            rocblas_datatype a_type;
            rocblas_datatype b_type;
            rocblas_datatype c_type;
            rocblas_datatype d_type;
            rocblas_datatype compute_type;

            if(std::is_same<T, rocblas_half>{})
            {
                a_type       = rocblas_datatype_f16_r;
                b_type       = rocblas_datatype_f16_r;
                c_type       = rocblas_datatype_f16_r;
                d_type       = rocblas_datatype_f16_r;
                compute_type = rocblas_datatype_f16_r;
                alpha        = static_cast<void*>(&alpha_half);
                beta         = static_cast<void*>(&beta_half);
            }
            else if(std::is_same<T, float>{})
            {
                a_type       = rocblas_datatype_f32_r;
                b_type       = rocblas_datatype_f32_r;
                c_type       = rocblas_datatype_f32_r;
                d_type       = rocblas_datatype_f32_r;
                compute_type = rocblas_datatype_f32_r;
                alpha        = static_cast<void*>(&alpha_float);
                beta         = static_cast<void*>(&beta_float);
            }
            else if(std::is_same<T, double>{})
            {
                a_type       = rocblas_datatype_f64_r;
                b_type       = rocblas_datatype_f64_r;
                c_type       = rocblas_datatype_f64_r;
                d_type       = rocblas_datatype_f64_r;
                compute_type = rocblas_datatype_f64_r;
                alpha        = static_cast<void*>(&alpha_double);
                beta         = static_cast<void*>(&beta_double);
            }

            rocblas_gemm_ex(handle,
                            transA,
                            transB,
                            m,
                            n,
                            k,
                            alpha,
                            da,
                            a_type,
                            lda,
                            db,
                            b_type,
                            ldb,
                            beta,
                            dc,
                            c_type,
                            ldc,
                            dd,
                            d_type,
                            ldd,
                            compute_type,
                            algo,
                            solution_index,
                            flags,
                            workspace_size,
                            workspace);

            rocblas_gemm_strided_batched_ex(handle,
                                            transA,
                                            transB,
                                            m,
                                            n,
                                            k,
                                            alpha,
                                            da,
                                            a_type,
                                            lda,
                                            stride_a,
                                            db,
                                            b_type,
                                            ldb,
                                            stride_b,
                                            beta,
                                            dc,
                                            c_type,
                                            ldc,
                                            stride_c,
                                            dd,
                                            d_type,
                                            ldd,
                                            stride_d,
                                            batch_count,
                                            compute_type,
                                            algo,
                                            solution_index,
                                            flags,
                                            workspace_size,
                                            workspace);
        }
    }

    setenv_status = setenv("ROCBLAS_LAYER", "0", true);

#ifdef GOOGLE_TEST
    EXPECT_EQ(setenv_status, 0);
#endif

    rocblas::reinit_logs(); // reinitialize logging, flushing old data to files

    //
    // write "golden file"
    //

    // find cwd string
    char temp[MAXPATHLEN];
    std::string cwd_str = getcwd(temp, MAXPATHLEN) ? temp : "";

    // open files
    auto trace_name2        = "rocblas_log_trace_gold_" + std::string(precision_letter<T>) + ".csv";
    std::string trace_path1 = cwd_str + "/" + trace_name1;
    std::string trace_path2 = cwd_str + "/" + trace_name2;

    std::string bench_name2 = "rocblas_log_bench_gold_" + std::string(precision_letter<T>) + ".txt";
    std::string bench_path1 = cwd_str + "/" + bench_name1;
    std::string bench_path2 = cwd_str + "/" + bench_name2;

    std::ofstream trace_ofs2;
    std::ofstream bench_ofs2;

    trace_ofs2.open(trace_path2);
    bench_ofs2.open(bench_path2);

    // Auxiliary function
    trace_ofs2 << "rocblas_create_handle\n";
    trace_ofs2 << "rocblas_set_pointer_mode,0\n";
    trace_ofs2 << "rocblas_get_pointer_mode,0\n";

    // BLAS1
    trace_ofs2 << replaceX<T>("rocblas_iXamax") << "," << n << "," << (void*)dx << "," << incx
               << '\n';
    bench_ofs2 << "./rocblas-bench -f iamax -r " << rocblas_precision_string<T> << " -n " << n
               << " --incx " << incx << '\n';

    trace_ofs2 << replaceX<T>("rocblas_iXamin") << "," << n << "," << (void*)dx << "," << incx
               << '\n';
    bench_ofs2 << "./rocblas-bench -f iamin -r " << rocblas_precision_string<T> << " -n " << n
               << " --incx " << incx << '\n';

    trace_ofs2 << replaceX<T>("rocblas_Xasum") << "," << n << "," << (void*)dx << "," << incx
               << '\n';
    bench_ofs2 << "./rocblas-bench -f asum -r " << rocblas_precision_string<T> << " -n " << n
               << " --incx " << incx << '\n';

    if(test_pointer_mode == rocblas_pointer_mode_host)
    {
        trace_ofs2 << replaceX<T>("rocblas_Xaxpy") << "," << n << "," << alpha << "," << (void*)dx
                   << "," << incx << "," << (void*)dy << "," << incy << '\n';
        bench_ofs2 << "./rocblas-bench -f axpy -r " << rocblas_precision_string<T> << " -n " << n
                   << " --alpha " << alpha << " --incx " << incx << " --incy " << incy << '\n';
    }
    else
    {
        trace_ofs2 << replaceX<T>("rocblas_Xaxpy") << "," << n << "," << (void*)&alpha << ","
                   << (void*)dx << "," << incx << "," << (void*)dy << "," << incy << '\n';
    }

    trace_ofs2 << replaceX<T>("rocblas_Xcopy") << "," << n << "," << (void*)dx << "," << incx << ","
               << (void*)dy << "," << incy << '\n';
    bench_ofs2 << "./rocblas-bench -f copy -r " << rocblas_precision_string<T> << " -n " << n
               << " --incx " << incx << " --incy " << incy << '\n';

    trace_ofs2 << replaceX<T>("rocblas_Xdot") << "," << n << "," << (void*)dx << "," << incx << ","
               << (void*)dy << "," << incy << '\n';
    bench_ofs2 << "./rocblas-bench -f dot -r " << rocblas_precision_string<T> << " -n " << n
               << " --incx " << incx << " --incy " << incy << '\n';

    trace_ofs2 << replaceX<T>("rocblas_Xnrm2") << "," << n << "," << (void*)dx << "," << incx
               << '\n';
    bench_ofs2 << "./rocblas-bench -f nrm2 -r " << rocblas_precision_string<T> << " -n " << n
               << " --incx " << incx << '\n';

    if(test_pointer_mode == rocblas_pointer_mode_host)
    {
        trace_ofs2 << replaceX<T>("rocblas_Xscal") << "," << n << "," << alpha << "," << (void*)dx
                   << "," << incx << '\n';
        bench_ofs2 << "./rocblas-bench -f scal -r " << rocblas_precision_string<T> << " -n " << n
                   << " --incx " << incx << " --alpha " << alpha << '\n';
    }
    else
    {
        trace_ofs2 << replaceX<T>("rocblas_Xscal") << "," << n << "," << (void*)&alpha << ","
                   << (void*)dx << "," << incx << '\n';
    }
    trace_ofs2 << replaceX<T>("rocblas_Xswap") << "," << n << "," << (void*)dx << "," << incx << ","
               << (void*)dy << "," << incy << '\n';

    bench_ofs2 << "./rocblas-bench -f swap -r " << rocblas_precision_string<T> << " -n " << n
               << " --incx " << incx << " --incy " << incy << '\n';

    // BLAS2
    auto transA_letter = rocblas2char_operation(transA);
    auto transB_letter = rocblas2char_operation(transB);
    auto side_letter   = rocblas2char_side(side);
    auto uplo_letter   = rocblas2char_fill(uplo);
    auto diag_letter   = rocblas2char_diagonal(diag);

    if(test_pointer_mode == rocblas_pointer_mode_host)
    {
        trace_ofs2 << replaceX<T>("rocblas_Xger") << "," << m << "," << n << "," << alpha << ","
                   << (void*)dx << "," << incx << "," << (void*)dy << "," << incy << ","
                   << (void*)da << "," << lda << '\n';
        bench_ofs2 << "./rocblas-bench -f ger -r " << rocblas_precision_string<T> << " -m " << m
                   << " -n " << n << " --alpha " << alpha << " --incx " << incx << " --incy "
                   << incy << " --lda " << lda << '\n';
    }
    else
    {
        trace_ofs2 << replaceX<T>("rocblas_Xger") << "," << m << "," << n << "," << (void*)&alpha
                   << "," << (void*)dx << "," << incx << "," << (void*)dy << "," << incy << ","
                   << (void*)da << "," << lda << '\n';
    }

    if(test_pointer_mode == rocblas_pointer_mode_host)
    {
        trace_ofs2 << replaceX<T>("rocblas_Xsyr") << "," << uplo << "," << n << "," << alpha << ","
                   << (void*)dx << "," << incx << "," << (void*)da << "," << lda << '\n';
        bench_ofs2 << "./rocblas-bench -f syr -r " << rocblas_precision_string<T> << " --uplo "
                   << uplo_letter << " -n " << n << " --alpha " << alpha << " --incx " << incx
                   << " --lda " << lda << '\n';
    }
    else
    {
        trace_ofs2 << replaceX<T>("rocblas_Xsyr") << "," << uplo << "," << n << "," << (void*)&alpha
                   << "," << (void*)dx << "," << incx << "," << (void*)da << "," << lda << '\n';
    }

    if(test_pointer_mode == rocblas_pointer_mode_host)
    {
        trace_ofs2 << replaceX<T>("rocblas_Xgemv") << "," << transA << "," << m << "," << n << ","
                   << alpha << "," << (void*)da << "," << lda << "," << (void*)dx << "," << incx
                   << "," << beta << "," << (void*)dy << "," << incy << '\n';

        bench_ofs2 << "./rocblas-bench -f gemv -r "
                   << rocblas_precision_string<T> << " --transposeA " << transA_letter << " -m "
                   << m << " -n " << n << " --alpha " << alpha << " --lda " << lda << " --incx "
                   << incx << " --beta " << beta << " --incy " << incy << '\n';
    }
    else
    {
        trace_ofs2 << replaceX<T>("rocblas_Xgemv") << "," << transA << "," << m << "," << n << ","
                   << (void*)&alpha << "," << (void*)da << "," << lda << "," << (void*)dx << ","
                   << incx << "," << (void*)&beta << "," << (void*)dy << "," << incy << '\n';
    }

    // BLAS3

    if(test_pointer_mode == rocblas_pointer_mode_host)
    {
        trace_ofs2 << replaceX<T>("rocblas_Xgeam") << "," << transA << "," << transB << "," << m
                   << "," << n << "," << alpha << "," << (void*)da << "," << lda << "," << beta
                   << "," << (void*)db << "," << ldb << "," << (void*)dc << "," << ldc << '\n';

        bench_ofs2 << "./rocblas-bench -f geam -r "
                   << rocblas_precision_string<T> << " --transposeA " << transA_letter
                   << " --transposeB " << transB_letter << " -m " << m << " -n " << n << " --alpha "
                   << alpha << " --lda " << lda << " --beta " << beta << " --ldb " << ldb
                   << " --ldc " << ldc << '\n';
    }
    else
    {
        trace_ofs2 << replaceX<T>("rocblas_Xgeam") << "," << transA << "," << transB << "," << m
                   << "," << n << "," << (void*)&alpha << "," << (void*)da << "," << lda << ","
                   << (void*)&beta << "," << (void*)db << "," << ldb << "," << (void*)dc << ","
                   << ldc << '\n';
    }

    if(BUILD_WITH_TENSILE)
    {
        /* trsm calls rocblas_get_stream and rocblas_dgemm, so test it by comparing files
                if(test_pointer_mode == rocblas_pointer_mode_host)
                {
                    trace_ofs2 << "\n"
                               << replaceX<T>("rocblas_Xtrsm") << "," << side << "," << uplo
                               << "," << transA << "," << diag << "," << m
                               << "," << n << "," << alpha << "," << (void*)da << "," << lda
                               << "," << (void*)db << "," << ldb;

                    bench_ofs2 << "\n"
                               << "./rocblas-bench -f trsm -r " << rocblas_precision_string<T>
                               << " --side " << side_letter << " --uplo " << uplo_letter
                               << " --transposeA " << transA_letter << " --diag " << diag_letter
                               << " -m " << m << " -n " << n << " --alpha " << alpha
                               << " --lda " << lda << " --ldb " << ldb;
                }
                else
                {
                    trace_ofs2 << "\n"
                               << replaceX<T>("rocblas_Xtrsm") << "," << side << "," << uplo
                               << "," << transA << "," << diag << "," << m
                               << "," << n << "," << (void*)&alpha << "," << (void*)da << ","
           << lda
                               << "," << (void*)db << "," << ldb;
                }
        */
        if(test_pointer_mode == rocblas_pointer_mode_host)
        {
            trace_ofs2 << replaceX<T>("rocblas_Xgemm") << "," << transA << "," << transB << "," << m
                       << "," << n << "," << k << "," << alpha << "," << (void*)da << "," << lda
                       << "," << (void*)db << "," << ldb << "," << beta << "," << (void*)dc << ","
                       << ldc << '\n';

            bench_ofs2 << "./rocblas-bench -f gemm -r "
                       << rocblas_precision_string<T> << " --transposeA " << transA_letter
                       << " --transposeB " << transB_letter << " -m " << m << " -n " << n << " -k "
                       << k << " --alpha " << alpha << " --lda " << lda << " --ldb " << ldb
                       << " --beta " << beta << " --ldc " << ldc << '\n';
        }
        else
        {
            trace_ofs2 << replaceX<T>("rocblas_Xgemm") << "," << transA << "," << transB << "," << m
                       << "," << n << "," << k << "," << (void*)&alpha << "," << (void*)da << ","
                       << lda << "," << (void*)db << "," << ldb << "," << (void*)&beta << ","
                       << (void*)dc << "," << ldc << '\n';
        }

        if(test_pointer_mode == rocblas_pointer_mode_host)
        {
            trace_ofs2 << replaceX<T>("rocblas_Xgemm_strided_batched") << "," << transA << ","
                       << transB << "," << m << "," << n << "," << k << "," << alpha << ","
                       << (void*)da << "," << lda << "," << stride_a << "," << (void*)db << ","
                       << ldb << "," << stride_b << "," << beta << "," << (void*)dc << "," << ldc
                       << "," << stride_c << "," << batch_count << '\n';

            bench_ofs2 << "./rocblas-bench -f gemm_strided_batched -r "
                       << rocblas_precision_string<T> << " --transposeA " << transA_letter
                       << " --transposeB " << transB_letter << " -m " << m << " -n " << n << " -k "
                       << k << " --alpha " << alpha << " --lda " << lda << " --stride_a "
                       << stride_a << " --ldb " << ldb << " --stride_b " << stride_b << " --beta "
                       << beta << " --ldc " << ldc << " --stride_c " << stride_c << " --batch "
                       << batch_count << '\n';
        }
        else
        {
            trace_ofs2 << replaceX<T>("rocblas_Xgemm_strided_batched") << "," << transA << ","
                       << transB << "," << m << "," << n << "," << k << "," << (void*)&alpha << ","
                       << (void*)da << "," << lda << "," << stride_a << "," << (void*)db << ","
                       << ldb << "," << stride_b << "," << (void*)&beta << "," << (void*)dc << ","
                       << ldc << "," << stride_c << "," << batch_count << '\n';
        }

        if(test_pointer_mode == rocblas_pointer_mode_host)
        {
            rocblas_datatype a_type, b_type, c_type, d_type, compute_type;

            if(std::is_same<T, rocblas_half>{})
            {
                a_type       = rocblas_datatype_f16_r;
                b_type       = rocblas_datatype_f16_r;
                c_type       = rocblas_datatype_f16_r;
                d_type       = rocblas_datatype_f16_r;
                compute_type = rocblas_datatype_f16_r;
            }
            else if(std::is_same<T, float>{})
            {
                a_type       = rocblas_datatype_f32_r;
                b_type       = rocblas_datatype_f32_r;
                c_type       = rocblas_datatype_f32_r;
                d_type       = rocblas_datatype_f32_r;
                compute_type = rocblas_datatype_f32_r;
            }
            if(std::is_same<T, double>{})
            {
                a_type       = rocblas_datatype_f64_r;
                b_type       = rocblas_datatype_f64_r;
                c_type       = rocblas_datatype_f64_r;
                d_type       = rocblas_datatype_f64_r;
                compute_type = rocblas_datatype_f64_r;
            }

            rocblas_gemm_algo algo = rocblas_gemm_algo_standard;
            int32_t solution_index = 0;
            uint32_t flags         = 0;
            size_t* workspace_size = 0;
            void* workspace        = 0;

            trace_ofs2 << "rocblas_gemm_ex"
                       << "," << transA << "," << transB << "," << m << "," << n << "," << k << ","
                       << alpha << "," << (void*)da << "," << rocblas_datatype_string(a_type) << ","
                       << lda << "," << (void*)db << "," << rocblas_datatype_string(b_type) << ","
                       << ldb << "," << beta << "," << (void*)dc << ","
                       << rocblas_datatype_string(c_type) << "," << ldc << "," << (void*)dd << ","
                       << rocblas_datatype_string(d_type) << "," << ldd << ","
                       << rocblas_datatype_string(compute_type) << "," << algo << ","
                       << solution_index << "," << flags << "," << workspace_size << ","
                       << (void*)workspace << '\n';

            bench_ofs2 << "./rocblas-bench -f gemm_ex"
                       << " --transposeA " << transA_letter << " --transposeB " << transB_letter
                       << " -m " << m << " -n " << n << " -k " << k << " --alpha " << alpha
                       << " --a_type " << rocblas_datatype_string(a_type) << " --lda " << lda
                       << " --b_type " << rocblas_datatype_string(b_type) << " --ldb " << ldb
                       << " --beta " << beta << " --c_type " << rocblas_datatype_string(c_type)
                       << " --ldc " << ldc << " --d_type " << rocblas_datatype_string(d_type)
                       << " --ldd " << ldd << " --compute_type "
                       << rocblas_datatype_string(compute_type) << " --algo " << algo
                       << " --solution_index " << solution_index << " --flags " << flags
                       << " --workspace_size " << workspace_size << '\n';

            trace_ofs2 << "rocblas_gemm_strided_batched_ex"
                       << "," << transA << "," << transB << "," << m << "," << n << "," << k << ","
                       << alpha << "," << (void*)da << "," << rocblas_datatype_string(a_type) << ","
                       << lda << "," << stride_a << "," << (void*)db << ","
                       << rocblas_datatype_string(b_type) << "," << ldb << "," << stride_b << ","
                       << beta << "," << (void*)dc << "," << rocblas_datatype_string(c_type) << ","
                       << ldc << "," << stride_c << "," << (void*)dd << ","
                       << rocblas_datatype_string(d_type) << "," << ldd << "," << stride_d << ","
                       << batch_count << "," << rocblas_datatype_string(compute_type) << "," << algo
                       << "," << solution_index << "," << flags << "," << workspace_size << ","
                       << (void*)workspace << '\n';

            bench_ofs2 << "./rocblas-bench -f gemm_strided_batched_ex"
                       << " --transposeA " << transA_letter << " --transposeB " << transB_letter
                       << " -m " << m << " -n " << n << " -k " << k << " --alpha " << alpha
                       << " --a_type " << rocblas_datatype_string(a_type) << " --lda " << lda
                       << " --stride_a " << stride_a << " --b_type "
                       << rocblas_datatype_string(b_type) << " --ldb " << ldb << " --stride_b "
                       << stride_b << " --beta " << beta << " --c_type "
                       << rocblas_datatype_string(c_type) << " --ldc " << ldc << " --stride_c "
                       << stride_c << " --d_type " << rocblas_datatype_string(d_type) << " --ldd "
                       << ldd << " --stride_d " << stride_d << " --batch " << batch_count
                       << " --compute_type " << rocblas_datatype_string(compute_type) << " --algo "
                       << algo << " --solution_index " << solution_index << " --flags " << flags
                       << " --workspace_size " << workspace_size << '\n';
        }
        else
        {
            trace_ofs2 << replaceX<T>("rocblas_Xgemm") << "," << transA << "," << transB << "," << m
                       << "," << n << "," << k << "," << (void*)&alpha << "," << (void*)da << ","
                       << lda << "," << (void*)db << "," << ldb << "," << (void*)&beta << ","
                       << (void*)dc << "," << ldc << '\n';
        }
    }
    // exclude trtri as it is an internal function
    //  trace_ofs2 << "\n" << replaceX<T>("rocblas_Xtrtri")  << "," << uplo << "," << diag << "," <<
    //  n
    //  << "," << (void*)da << "," << lda << "," << (void*)db << "," << ldb;

    // Auxiliary function
    trace_ofs2 << "rocblas_destroy_handle\n";

    trace_ofs2.close();
    bench_ofs2.close();

    //
    // check if rocBLAS output files same as "golden files"
    //
    int trace_cmp = system(("cmp -s " + trace_path1 + " " + trace_path2).c_str());

    if(!trace_cmp)
    {
        remove(trace_path1.c_str());
        remove(trace_path2.c_str());
    }

#ifdef GOOGLE_TEST
    EXPECT_EQ(trace_cmp, 0);
#endif

    if(test_pointer_mode == rocblas_pointer_mode_host)
    {
        int bench_cmp = system(("cmp -s " + bench_path1 + " " + bench_path2).c_str());

#ifdef GOOGLE_TEST
        EXPECT_EQ(bench_cmp, 0);
#endif

        if(!bench_cmp)
        {
            remove(bench_path1.c_str());
            remove(bench_path2.c_str());
        }
    }
}
