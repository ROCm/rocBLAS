/* ************************************************************************
 * Copyright 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas_data.hpp"
#include "rocblas_test.hpp"
#include "type_dispatch.hpp"

namespace
{
    // Mostly copied from samples/example_sgemm.cpp
    template <typename T>
    void mat_mat_mult(T              alpha,
                      T              beta,
                      size_t         M,
                      size_t         N,
                      size_t         K,
                      T*             A,
                      rocblas_stride As1,
                      rocblas_stride As2,
                      T*             B,
                      rocblas_stride Bs1,
                      rocblas_stride Bs2,
                      T*             C,
                      rocblas_stride Cs1,
                      rocblas_stride Cs2)
    {
        for(size_t i1 = 0; i1 < M; i1++)
            for(size_t i2 = 0; i2 < N; i2++)
            {
                T t{0};
                if(alpha)
                {
                    for(size_t i3 = 0; i3 < K; i3++)
                        t += A[i1 * As1 + i3 * As2] * B[i3 * Bs1 + i2 * Bs2];
                    t *= alpha;
                }
                if(beta)
                    t += beta * C[i1 * Cs1 + i2 * Cs2];
                C[i1 * Cs1 + i2 * Cs2] = t;
            }
    }

    void thread_function(int id, const Arguments& arg)
    {
        CHECK_HIP_ERROR(hipSetDevice(id));

        rocblas_operation transa = rocblas_operation_none, transb = rocblas_operation_transpose;
        float             alpha = 1.1, beta = 0.9;
        rocblas_int       m = 1023, n = 1024, k = 1025;
        size_t            lda, ldb, ldc, size_a, size_b, size_c;
        rocblas_stride    a_stride_1, a_stride_2, b_stride_1, b_stride_2;
        if(transa == rocblas_operation_none)
        {
            lda        = m;
            size_a     = k * lda;
            a_stride_1 = 1;
            a_stride_2 = lda;
        }
        else
        {
            lda        = k;
            size_a     = m * lda;
            a_stride_1 = lda;
            a_stride_2 = 1;
        }
        if(transb == rocblas_operation_none)
        {
            ldb        = k;
            size_b     = n * ldb;
            b_stride_1 = 1;
            b_stride_2 = ldb;
        }
        else
        {
            ldb        = n;
            size_b     = k * ldb;
            b_stride_1 = ldb;
            b_stride_2 = 1;
        }
        ldc    = m;
        size_c = n * ldc;

        // Naming: da is in GPU (device) memory. ha is in CPU (host) memory
        std::vector<float> ha(size_a);
        std::vector<float> hb(size_b);
        std::vector<float> hc(size_c);
        std::vector<float> hc_gold(size_c);

        // initial data on host
        srand(1);
        for(int i = 0; i < size_a; ++i)
            ha[i] = rand() % 17;
        for(int i = 0; i < size_b; ++i)
            hb[i] = rand() % 17;
        for(int i = 0; i < size_c; ++i)
            hc[i] = rand() % 17;
        hc_gold = hc;

        // allocate memory on device
        float *da, *db, *dc;
        CHECK_HIP_ERROR(hipMalloc(&da, size_a * sizeof(float)));
        CHECK_HIP_ERROR(hipMalloc(&db, size_b * sizeof(float)));
        CHECK_HIP_ERROR(hipMalloc(&dc, size_c * sizeof(float)));

        // copy matrices from host to device
        CHECK_HIP_ERROR(hipMemcpy(da, ha.data(), sizeof(float) * size_a, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(db, hb.data(), sizeof(float) * size_b, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dc, hc.data(), sizeof(float) * size_c, hipMemcpyHostToDevice));

        rocblas_handle handle;
        CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));

        CHECK_ROCBLAS_ERROR(rocblas_sgemm(
            handle, transa, transb, m, n, k, &alpha, da, lda, db, ldb, &beta, dc, ldc));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hc.data(), dc, sizeof(float) * size_c, hipMemcpyDeviceToHost));

        float max_relative_error = std::numeric_limits<float>::min();

        // calculate golden or correct result
        mat_mat_mult<float>(alpha,
                            beta,
                            m,
                            n,
                            k,
                            ha.data(),
                            a_stride_1,
                            a_stride_2,
                            hb.data(),
                            b_stride_1,
                            b_stride_2,
                            hc_gold.data(),
                            1,
                            ldc);

        for(int i = 0; i < size_c; i++)
        {
            float relative_error = (hc_gold[i] - hc[i]) / hc_gold[i];
            relative_error       = relative_error > 0 ? relative_error : -relative_error;
            max_relative_error
                = relative_error < max_relative_error ? max_relative_error : relative_error;
        }

        float eps       = std::numeric_limits<float>::epsilon();
        float tolerance = 10;
        if(max_relative_error != max_relative_error || max_relative_error > eps * tolerance)
            FAIL() << "FAIL: max_relative_error = " << max_relative_error;

        CHECK_HIP_ERROR(hipFree(da));
        CHECK_HIP_ERROR(hipFree(db));
        CHECK_HIP_ERROR(hipFree(dc));
        CHECK_ROCBLAS_ERROR(rocblas_destroy_handle(handle));
    }

    void testing_multiheaded(const Arguments& arg)
    {
        int count;
        CHECK_HIP_ERROR(hipGetDeviceCount(&count));

        auto thread = std::make_unique<std::thread[]>(count);

        for(int id = 0; id < count; ++id)
            thread[id] = std::thread(thread_function, id, arg);

        for(int id = 0; id < count; ++id)
            thread[id].join();
    }

    template <typename...>
    struct multiheaded_testing : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "multiheaded"))
                testing_multiheaded(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    struct multiheaded : RocBLAS_Test<multiheaded, multiheaded_testing>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return true;
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            return !strcmp(arg.function, "multiheaded");
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            return RocBLAS_TestName<multiheaded>{};
        }
    };

    TEST_P(multiheaded, auxilliary)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            rocblas_simple_dispatch<multiheaded_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(multiheaded);

} // namespace
