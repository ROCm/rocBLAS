/* ************************************************************************
 * Copyright 2018-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "cblas_interface.hpp"
#include "norm.hpp"
#include "rocblas.hpp"
#include "rocblas_init.hpp"
#include "rocblas_math.hpp"
#include "rocblas_random.hpp"
#include "rocblas_test.hpp"
#include "rocblas_vector.hpp"
#include "unit.hpp"
#include "utility.hpp"

template <typename T, typename U = T>
void testing_rotg_strided_batched_bad_arg(const Arguments& arg)
{
    static const size_t safe_size   = 1;
    rocblas_int         batch_count = 5;
    rocblas_stride      stride_a    = 10;
    rocblas_stride      stride_b    = 10;
    rocblas_stride      stride_c    = 10;
    rocblas_stride      stride_s    = 10;

    rocblas_local_handle handle;
    device_vector<T>     da(batch_count * stride_a);
    device_vector<T>     db(batch_count * stride_b);
    device_vector<U>     dc(batch_count * stride_c);
    device_vector<T>     ds(batch_count * stride_s);
    if(!da || !db || !dc || !ds)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    EXPECT_ROCBLAS_STATUS(
        (rocblas_rotg_strided_batched<T, U>(
            nullptr, da, stride_a, db, stride_b, dc, stride_c, ds, stride_s, batch_count)),
        rocblas_status_invalid_handle);
    EXPECT_ROCBLAS_STATUS(
        (rocblas_rotg_strided_batched<T, U>(
            handle, nullptr, stride_a, db, stride_b, dc, stride_c, ds, stride_s, batch_count)),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        (rocblas_rotg_strided_batched<T, U>(
            handle, da, stride_a, nullptr, stride_b, dc, stride_c, ds, stride_s, batch_count)),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        (rocblas_rotg_strided_batched<T, U>(
            handle, da, stride_a, db, stride_b, nullptr, stride_c, ds, stride_s, batch_count)),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        (rocblas_rotg_strided_batched<T, U>(
            handle, da, stride_a, db, stride_b, dc, stride_c, nullptr, stride_s, batch_count)),
        rocblas_status_invalid_pointer);
}

template <typename T, typename U = T>
void testing_rotg_strided_batched(const Arguments& arg)
{
    const int   TEST_COUNT  = 100;
    rocblas_int stride_a    = arg.stride_a;
    rocblas_int stride_b    = arg.stride_b;
    rocblas_int stride_c    = arg.stride_c;
    rocblas_int stride_s    = arg.stride_d;
    rocblas_int batch_count = arg.batch_count;

    rocblas_local_handle handle;
    double               gpu_time_used, cpu_time_used;
    double               norm_error_host = 0.0, norm_error_device = 0.0;

    // check to prevent undefined memory allocation error
    if(batch_count <= 0)
    {
        static const size_t safe_size = 1; // arbitrarily set to 100
        device_vector<T>    da(safe_size);
        device_vector<T>    db(safe_size);
        device_vector<U>    dc(safe_size);
        device_vector<T>    ds(safe_size);
        if(!da || !db || !dc || !ds)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        if(batch_count < 0)
            EXPECT_ROCBLAS_STATUS((rocblas_rotg_strided_batched<T, U>)(handle,
                                                                       da,
                                                                       stride_a,
                                                                       db,
                                                                       stride_b,
                                                                       dc,
                                                                       stride_c,
                                                                       ds,
                                                                       stride_s,
                                                                       batch_count),
                                  rocblas_status_invalid_size);
        else
            CHECK_ROCBLAS_ERROR((rocblas_rotg_strided_batched<T, U>(
                handle, da, stride_a, db, stride_b, dc, stride_c, ds, stride_s, batch_count)));
        return;
    }

    size_t size_a = size_t(stride_a) * size_t(batch_count);
    size_t size_b = size_t(stride_b) * size_t(batch_count);
    size_t size_c = size_t(stride_c) * size_t(batch_count);
    size_t size_s = size_t(stride_s) * size_t(batch_count);

    host_vector<T> ha(size_a);
    host_vector<T> hb(size_b);
    host_vector<U> hc(size_c);
    host_vector<T> hs(size_s);

    for(int i = 0; i < TEST_COUNT; i++)
    {
        // Initial data on CPU
        rocblas_seedrand();
        rocblas_init<T>(ha, 1, 1, 1, stride_a, batch_count);
        rocblas_init<T>(hb, 1, 1, 1, stride_b, batch_count);
        rocblas_init<U>(hc, 1, 1, 1, stride_c, batch_count);
        rocblas_init<T>(hs, 1, 1, 1, stride_s, batch_count);

        // CPU_BLAS
        host_vector<T> ca = ha;
        host_vector<T> cb = hb;
        host_vector<U> cc = hc;
        host_vector<T> cs = hs;
        cpu_time_used     = get_time_us();
        for(int b = 0; b < batch_count; b++)
        {
            cblas_rotg<T, U>(
                ca + b * stride_a, cb + b * stride_b, cc + b * stride_c, cs + b * stride_s);
        }
        cpu_time_used = get_time_us() - cpu_time_used;

        // Test rocblas_pointer_mode_host
        {
            host_vector<T> ra = ha;
            host_vector<T> rb = hb;
            host_vector<U> rc = hc;
            host_vector<T> rs = hs;
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            CHECK_ROCBLAS_ERROR((rocblas_rotg_strided_batched<T, U>(
                handle, ra, stride_a, rb, stride_b, rc, stride_c, rs, stride_s, batch_count)));

            if(arg.unit_check)
            {
                double rel_error = std::numeric_limits<U>::epsilon() * 100;
                near_check_general<T>(1, 1, batch_count, 1, stride_a, ca, ra, rel_error);
                near_check_general<T>(1, 1, batch_count, 1, stride_b, cb, rb, rel_error);
                near_check_general<U>(1, 1, batch_count, 1, stride_c, cc, rc, rel_error);
                near_check_general<T>(1, 1, batch_count, 1, stride_s, cs, rs, rel_error);
            }

            if(arg.norm_check)
            {
                norm_error_host
                    = norm_check_general<T>('F', 1, 1, 1, stride_a, batch_count, ca, ra);
                norm_error_host
                    += norm_check_general<T>('F', 1, 1, 1, stride_b, batch_count, cb, rb);
                norm_error_host
                    += norm_check_general<U>('F', 1, 1, 1, stride_c, batch_count, cc, rc);
                norm_error_host
                    += norm_check_general<T>('F', 1, 1, 1, stride_s, batch_count, cs, rs);
            }
        }

        // Test rocblas_pointer_mode_device
        {
            device_vector<T> da(size_a);
            device_vector<T> db(size_b);
            device_vector<U> dc(size_c);
            device_vector<T> ds(size_s);
            CHECK_HIP_ERROR(hipMemcpy(da, ha, sizeof(T) * size_a, hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(db, hb, sizeof(T) * size_b, hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(dc, hc, sizeof(U) * size_c, hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(ds, hs, sizeof(T) * size_s, hipMemcpyHostToDevice));
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            CHECK_ROCBLAS_ERROR((rocblas_rotg_strided_batched<T, U>(
                handle, da, stride_a, db, stride_b, dc, stride_c, ds, stride_s, batch_count)));
            host_vector<T> ra(size_a);
            host_vector<T> rb(size_b);
            host_vector<U> rc(size_c);
            host_vector<T> rs(size_s);
            CHECK_HIP_ERROR(hipMemcpy(ra, da, sizeof(T) * size_a, hipMemcpyDeviceToHost));
            CHECK_HIP_ERROR(hipMemcpy(rb, db, sizeof(T) * size_b, hipMemcpyDeviceToHost));
            CHECK_HIP_ERROR(hipMemcpy(rc, dc, sizeof(U) * size_c, hipMemcpyDeviceToHost));
            CHECK_HIP_ERROR(hipMemcpy(rs, ds, sizeof(T) * size_s, hipMemcpyDeviceToHost));

            if(arg.unit_check)
            {
                double rel_error = std::numeric_limits<U>::epsilon() * 100;
                near_check_general<T>(1, 1, batch_count, 1, stride_a, ca, ra, rel_error);
                near_check_general<T>(1, 1, batch_count, 1, stride_b, cb, rb, rel_error);
                near_check_general<U>(1, 1, batch_count, 1, stride_c, cc, rc, rel_error);
                near_check_general<T>(1, 1, batch_count, 1, stride_s, cs, rs, rel_error);
            }

            if(arg.norm_check)
            {
                norm_error_host
                    = norm_check_general<T>('F', 1, 1, 1, stride_a, batch_count, ca, ra);
                norm_error_host
                    += norm_check_general<T>('F', 1, 1, 1, stride_b, batch_count, cb, rb);
                norm_error_host
                    += norm_check_general<U>('F', 1, 1, 1, stride_c, batch_count, cc, rc);
                norm_error_host
                    += norm_check_general<T>('F', 1, 1, 1, stride_s, batch_count, cs, rs);
            }
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = 100;
        // Device mode will be quicker
        // (TODO: or is there another reason we are typically using host_mode for timing?)
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

        device_vector<T> da(size_a);
        device_vector<T> db(size_b);
        device_vector<U> dc(size_c);
        device_vector<T> ds(size_s);
        CHECK_HIP_ERROR(hipMemcpy(da, ha, sizeof(T) * size_a, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(db, hb, sizeof(T) * size_b, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dc, hc, sizeof(U) * size_c, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(ds, hs, sizeof(T) * size_s, hipMemcpyHostToDevice));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_rotg_strided_batched<T, U>(
                handle, da, stride_a, db, stride_b, dc, stride_c, ds, stride_s, batch_count);
        }
        gpu_time_used = get_time_us(); // in microseconds
        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_rotg_strided_batched<T, U>(
                handle, da, stride_a, db, stride_b, dc, stride_c, ds, stride_s, batch_count);
        }
        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        std::cout << "rocblas-us,CPU-us";
        if(arg.norm_check)
            std::cout << ",norm_error_host_ptr,norm_error_device";
        std::cout << std::endl;

        std::cout << gpu_time_used << "," << cpu_time_used;
        if(arg.norm_check)
            std::cout << ',' << norm_error_host << ',' << norm_error_device;
        std::cout << std::endl;
    }
}
