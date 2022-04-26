/* ************************************************************************
 * Copyright 2018-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

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
    // clang-format off
    const bool FORTRAN                         = arg.fortran;
    auto       rocblas_rotg_strided_batched_fn = FORTRAN ? rocblas_rotg_strided_batched<T, U, true>
                                                         : rocblas_rotg_strided_batched<T, U, false>;
    // clang-format on

    rocblas_int    batch_count = 5;
    rocblas_stride stride_a    = 10;
    rocblas_stride stride_b    = 10;
    rocblas_stride stride_c    = 10;
    rocblas_stride stride_s    = 10;

    rocblas_local_handle handle{arg};

    // Allocate device memory
    device_strided_batch_vector<T> da(1, 1, stride_a, batch_count);
    device_strided_batch_vector<T> db(1, 1, stride_b, batch_count);
    device_strided_batch_vector<U> dc(1, 1, stride_c, batch_count);
    device_strided_batch_vector<T> ds(1, 1, stride_s, batch_count);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(da.memcheck());
    CHECK_DEVICE_ALLOCATION(db.memcheck());
    CHECK_DEVICE_ALLOCATION(dc.memcheck());
    CHECK_DEVICE_ALLOCATION(ds.memcheck());

    EXPECT_ROCBLAS_STATUS(
        (rocblas_rotg_strided_batched_fn(
            nullptr, da, stride_a, db, stride_b, dc, stride_c, ds, stride_s, batch_count)),
        rocblas_status_invalid_handle);
    EXPECT_ROCBLAS_STATUS(
        (rocblas_rotg_strided_batched_fn(
            handle, nullptr, stride_a, db, stride_b, dc, stride_c, ds, stride_s, batch_count)),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        (rocblas_rotg_strided_batched_fn(
            handle, da, stride_a, nullptr, stride_b, dc, stride_c, ds, stride_s, batch_count)),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        (rocblas_rotg_strided_batched_fn(
            handle, da, stride_a, db, stride_b, nullptr, stride_c, ds, stride_s, batch_count)),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        (rocblas_rotg_strided_batched_fn(
            handle, da, stride_a, db, stride_b, dc, stride_c, nullptr, stride_s, batch_count)),
        rocblas_status_invalid_pointer);
}

template <typename T, typename U = T>
void testing_rotg_strided_batched(const Arguments& arg)
{
    // clang-format off
    const bool FORTRAN                         = arg.fortran;
    auto       rocblas_rotg_strided_batched_fn = FORTRAN ? rocblas_rotg_strided_batched<T, U, true>
                                                         : rocblas_rotg_strided_batched<T, U, false>;
    // clang-format on

    const int   TEST_COUNT  = 100;
    rocblas_int stride_a    = arg.stride_a;
    rocblas_int stride_b    = arg.stride_b;
    rocblas_int stride_c    = arg.stride_c;
    rocblas_int stride_s    = arg.stride_d;
    rocblas_int batch_count = arg.batch_count;

    rocblas_local_handle handle{arg};
    double               gpu_time_used, cpu_time_used;
    double               norm_error_host = 0.0, norm_error_device = 0.0;
    const U              rel_error = std::numeric_limits<U>::epsilon() * 1000;

    // check to prevent undefined memory allocation error
    if(batch_count <= 0)
    {
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        EXPECT_ROCBLAS_STATUS((rocblas_rotg_strided_batched_fn(handle,
                                                               nullptr,
                                                               stride_a,
                                                               nullptr,
                                                               stride_b,
                                                               nullptr,
                                                               stride_c,
                                                               nullptr,
                                                               stride_s,
                                                               batch_count)),
                              rocblas_status_success);
        return;
    }

    host_strided_batch_vector<T> ha(1, 1, stride_a, batch_count);
    host_strided_batch_vector<T> hb(1, 1, stride_b, batch_count);
    host_strided_batch_vector<U> hc(1, 1, stride_c, batch_count);
    host_strided_batch_vector<T> hs(1, 1, stride_s, batch_count);

    bool enable_near_check_general = true;

#ifdef WIN32
    // During explicit NaN initialization (i.e., when arg.alpha=NaN), the host side computation results of OpenBLAS differs from the result of kernel computation in rocBLAS.
    // The output value of `hb_gold` is NaN in OpenBLAS and, the output value of `hb_gold` is 1.000 in rocBLAS. There was no difference observed when comparing the rocBLAS results with BLIS.
    // Therefore, using the bool enable_near_check_general to skip unit check for WIN32 during NaN initialization.

    enable_near_check_general = !rocblas_isnan(arg.alpha);
#endif

    for(int i = 0; i < TEST_COUNT; i++)
    {
        // Initialize data on host memory
        rocblas_init_vector(ha, arg, rocblas_client_alpha_sets_nan, true);
        rocblas_init_vector(hb, arg, rocblas_client_alpha_sets_nan, false);
        rocblas_init_vector(hc, arg, rocblas_client_alpha_sets_nan, false);
        rocblas_init_vector(hs, arg, rocblas_client_alpha_sets_nan, false);

        // CPU_BLAS
        host_strided_batch_vector<T> ha_gold(1, 1, stride_a, batch_count);
        host_strided_batch_vector<T> hb_gold(1, 1, stride_b, batch_count);
        host_strided_batch_vector<U> hc_gold(1, 1, stride_c, batch_count);
        host_strided_batch_vector<T> hs_gold(1, 1, stride_s, batch_count);

        ha_gold.copy_from(ha);
        hb_gold.copy_from(hb);
        hc_gold.copy_from(hc);
        hs_gold.copy_from(hs);

        cpu_time_used = get_time_us_no_sync();
        for(int b = 0; b < batch_count; b++)
        {
            cblas_rotg<T, U>(ha_gold[b], hb_gold[b], hc_gold[b], hs_gold[b]);
        }
        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        // Test rocblas_pointer_mode_host
        {
            host_strided_batch_vector<T> ra(1, 1, stride_a, batch_count);
            host_strided_batch_vector<T> rb(1, 1, stride_b, batch_count);
            host_strided_batch_vector<U> rc(1, 1, stride_c, batch_count);
            host_strided_batch_vector<T> rs(1, 1, stride_s, batch_count);

            ra.copy_from(ha);
            rb.copy_from(hb);
            rc.copy_from(hc);
            rs.copy_from(hs);

            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            CHECK_ROCBLAS_ERROR((rocblas_rotg_strided_batched_fn(
                handle, ra, stride_a, rb, stride_b, rc, stride_c, rs, stride_s, batch_count)));

            if(arg.unit_check)
            {
                if(enable_near_check_general)
                {
                    near_check_general<T>(1, 1, 1, stride_a, ha_gold, ra, batch_count, rel_error);
                    near_check_general<T>(1, 1, 1, stride_b, hb_gold, rb, batch_count, rel_error);
                    near_check_general<U>(1, 1, 1, stride_c, hc_gold, rc, batch_count, rel_error);
                    near_check_general<T>(1, 1, 1, stride_s, hs_gold, rs, batch_count, rel_error);
                }
            }

            if(arg.norm_check)
            {
                norm_error_host
                    = norm_check_general<T>('F', 1, 1, 1, stride_a, ha_gold, ra, batch_count);
                norm_error_host
                    += norm_check_general<T>('F', 1, 1, 1, stride_b, hb_gold, rb, batch_count);
                norm_error_host
                    += norm_check_general<U>('F', 1, 1, 1, stride_c, hc_gold, rc, batch_count);
                norm_error_host
                    += norm_check_general<T>('F', 1, 1, 1, stride_s, hs_gold, rs, batch_count);
            }
        }

        // Test rocblas_pointer_mode_device
        {
            // Allocate device memory
            device_strided_batch_vector<T> da(1, 1, stride_a, batch_count);
            device_strided_batch_vector<T> db(1, 1, stride_b, batch_count);
            device_strided_batch_vector<U> dc(1, 1, stride_c, batch_count);
            device_strided_batch_vector<T> ds(1, 1, stride_s, batch_count);

            // Check device memory allocation
            CHECK_DEVICE_ALLOCATION(da.memcheck());
            CHECK_DEVICE_ALLOCATION(db.memcheck());
            CHECK_DEVICE_ALLOCATION(dc.memcheck());
            CHECK_DEVICE_ALLOCATION(ds.memcheck());

            // Transfer from CPU to GPU
            CHECK_HIP_ERROR(da.transfer_from(ha));
            CHECK_HIP_ERROR(db.transfer_from(hb));
            CHECK_HIP_ERROR(dc.transfer_from(hc));
            CHECK_HIP_ERROR(ds.transfer_from(hs));

            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            CHECK_ROCBLAS_ERROR((rocblas_rotg_strided_batched_fn(
                handle, da, stride_a, db, stride_b, dc, stride_c, ds, stride_s, batch_count)));

            host_strided_batch_vector<T> ra(1, 1, stride_a, batch_count);
            host_strided_batch_vector<T> rb(1, 1, stride_b, batch_count);
            host_strided_batch_vector<U> rc(1, 1, stride_c, batch_count);
            host_strided_batch_vector<T> rs(1, 1, stride_s, batch_count);

            // Transfer from GPU to CPU
            CHECK_HIP_ERROR(ra.transfer_from(da));
            CHECK_HIP_ERROR(rb.transfer_from(db));
            CHECK_HIP_ERROR(rc.transfer_from(dc));
            CHECK_HIP_ERROR(rs.transfer_from(ds));

            if(arg.unit_check)
            {
                if(enable_near_check_general)
                {
                    near_check_general<T>(1, 1, 1, stride_a, ha_gold, ra, batch_count, rel_error);
                    near_check_general<T>(1, 1, 1, stride_b, hb_gold, rb, batch_count, rel_error);
                    near_check_general<U>(1, 1, 1, stride_c, hc_gold, rc, batch_count, rel_error);
                    near_check_general<T>(1, 1, 1, stride_s, hs_gold, rs, batch_count, rel_error);
                }
            }

            if(arg.norm_check)
            {
                norm_error_device
                    = norm_check_general<T>('F', 1, 1, 1, stride_a, ha_gold, ra, batch_count);
                norm_error_device
                    += norm_check_general<T>('F', 1, 1, 1, stride_b, hb_gold, rb, batch_count);
                norm_error_device
                    += norm_check_general<U>('F', 1, 1, 1, stride_c, hc_gold, rc, batch_count);
                norm_error_device
                    += norm_check_general<T>('F', 1, 1, 1, stride_s, hs_gold, rs, batch_count);
            }
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;
        // Device mode will be quicker
        // (TODO: or is there another reason we are typically using host_mode for timing?)
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

        // Allocate device memory
        device_strided_batch_vector<T> da(1, 1, stride_a, batch_count);
        device_strided_batch_vector<T> db(1, 1, stride_b, batch_count);
        device_strided_batch_vector<U> dc(1, 1, stride_c, batch_count);
        device_strided_batch_vector<T> ds(1, 1, stride_s, batch_count);

        // Check device memory allocation
        CHECK_DEVICE_ALLOCATION(da.memcheck());
        CHECK_DEVICE_ALLOCATION(db.memcheck());
        CHECK_DEVICE_ALLOCATION(dc.memcheck());
        CHECK_DEVICE_ALLOCATION(ds.memcheck());

        // Transfer from CPU to GPU
        CHECK_HIP_ERROR(da.transfer_from(ha));
        CHECK_HIP_ERROR(db.transfer_from(hb));
        CHECK_HIP_ERROR(dc.transfer_from(hc));
        CHECK_HIP_ERROR(ds.transfer_from(hs));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_rotg_strided_batched_fn(
                handle, da, stride_a, db, stride_b, dc, stride_c, ds, stride_s, batch_count);
        }
        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds
        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_rotg_strided_batched_fn(
                handle, da, stride_a, db, stride_b, dc, stride_c, ds, stride_s, batch_count);
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_stride_a, e_stride_b, e_stride_c, e_stride_d, e_batch_count>{}.log_args<T>(
            rocblas_cout,
            arg,
            gpu_time_used,
            ArgumentLogging::NA_value,
            ArgumentLogging::NA_value,
            cpu_time_used,
            norm_error_host,
            norm_error_device);
    }
}
