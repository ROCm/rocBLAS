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
void testing_rotg_batched_bad_arg(const Arguments& arg)
{
    rocblas_int         batch_count = 5;
    static const size_t safe_size   = 1;

    rocblas_local_handle    handle;
    device_vector<T*, 0, T> da(batch_count);
    device_vector<T*, 0, T> db(batch_count);
    device_vector<U*, 0, U> dc(batch_count);
    device_vector<T*, 0, T> ds(batch_count);

    if(!da || !db || !dc || !ds)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    EXPECT_ROCBLAS_STATUS((rocblas_rotg_batched<T, U>(nullptr, da, db, dc, ds, batch_count)),
                          rocblas_status_invalid_handle);
    EXPECT_ROCBLAS_STATUS((rocblas_rotg_batched<T, U>(handle, nullptr, db, dc, ds, batch_count)),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS((rocblas_rotg_batched<T, U>(handle, da, nullptr, dc, ds, batch_count)),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS((rocblas_rotg_batched<T, U>(handle, da, db, nullptr, ds, batch_count)),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS((rocblas_rotg_batched<T, U>(handle, da, db, dc, nullptr, batch_count)),
                          rocblas_status_invalid_pointer);
}

template <typename T, typename U = T>
void testing_rotg_batched(const Arguments& arg)
{
    const int            TEST_COUNT  = 100;
    rocblas_int          batch_count = arg.batch_count;
    rocblas_local_handle handle;

    double gpu_time_used, cpu_time_used;
    double norm_error_host = 0.0, norm_error_device = 0.0;

    // check to prevent undefined memory allocation error
    if(batch_count <= 0)
    {
        size_t                  safe_size = 1;
        device_vector<T*, 0, T> da(safe_size);
        device_vector<T*, 0, T> db(safe_size);
        device_vector<U*, 0, U> dc(safe_size);
        device_vector<T*, 0, T> ds(safe_size);

        if(!da || !db || !dc || !ds)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        if(batch_count < 0)
            EXPECT_ROCBLAS_STATUS((rocblas_rotg_batched<T, U>(handle, da, db, dc, ds, batch_count)),
                                  rocblas_status_invalid_size);
        else
            CHECK_ROCBLAS_ERROR((rocblas_rotg_batched<T, U>(handle, da, db, dc, ds, batch_count)));
        return;
    }

    // Initial Data on CPU
    host_vector<T> ha[batch_count];
    host_vector<T> hb[batch_count];
    host_vector<U> hc[batch_count];
    host_vector<T> hs[batch_count];

    device_batch_vector<T> ba(batch_count, 1);
    device_batch_vector<T> bb(batch_count, 1);

    for(int b = 0; b < batch_count; b++)
    {
        ha[b] = host_vector<T>(1);
        hb[b] = host_vector<T>(1);
        hc[b] = host_vector<U>(1);
        hs[b] = host_vector<T>(1);
    }

    for(int i = 0; i < TEST_COUNT; i++)
    {
        host_vector<T> ca[batch_count];
        host_vector<T> cb[batch_count];
        host_vector<U> cc[batch_count];
        host_vector<T> cs[batch_count];

        rocblas_seedrand();
        for(int b = 0; b < batch_count; b++)
        {
            rocblas_init<T>(ha[b], 1, 1, 1);
            rocblas_init<T>(hb[b], 1, 1, 1);
            rocblas_init<U>(hc[b], 1, 1, 1);
            rocblas_init<T>(hs[b], 1, 1, 1);
            ca[b] = ha[b];
            cb[b] = hb[b];
            cc[b] = hc[b];
            cs[b] = hs[b];
        }

        cpu_time_used = get_time_us();
        for(int b = 0; b < batch_count; b++)
        {
            cblas_rotg<T, U>(ca[b], cb[b], cc[b], cs[b]);
        }
        cpu_time_used = get_time_us() - cpu_time_used;

        // Test rocblas_pointer_mode_host
        {
            host_vector<T> ra[batch_count];
            host_vector<T> rb[batch_count];
            host_vector<U> rc[batch_count];
            host_vector<T> rs[batch_count];
            T*             ra_in[batch_count];
            T*             rb_in[batch_count];
            U*             rc_in[batch_count];
            T*             rs_in[batch_count];
            for(int b = 0; b < batch_count; b++)
            {
                ra_in[b] = ra[b] = ha[b];
                rb_in[b] = rb[b] = hb[b];
                rc_in[b] = rc[b] = hc[b];
                rs_in[b] = rs[b] = hs[b];
            }

            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

            CHECK_ROCBLAS_ERROR(
                (rocblas_rotg_batched<T, U>(handle, ra_in, rb_in, rc_in, rs_in, batch_count)));

            if(arg.unit_check)
            {
                unit_check_general<T>(1, 1, batch_count, 1, ra, ca);
                unit_check_general<T>(1, 1, batch_count, 1, rb, cb);
                unit_check_general<U>(1, 1, batch_count, 1, rc, cc);
                unit_check_general<T>(1, 1, batch_count, 1, rs, cs);
            }

            if(arg.norm_check)
            {
                norm_error_host = norm_check_general<T>('F', 1, 1, batch_count, 1, ra, ca);
                norm_error_host += norm_check_general<T>('F', 1, 1, batch_count, 1, rb, cb);
                norm_error_host += norm_check_general<U>('F', 1, 1, batch_count, 1, rc, cc);
                norm_error_host += norm_check_general<T>('F', 1, 1, batch_count, 1, rs, cs);
            }
        }

        // Test rocblas_pointer_mode_device
        {
            device_vector<T*, 0, T> da(batch_count);
            device_vector<T*, 0, T> db(batch_count);
            device_vector<U*, 0, U> dc(batch_count);
            device_vector<T*, 0, T> ds(batch_count);
            device_batch_vector<T>  ba(batch_count, 1);
            device_batch_vector<T>  bb(batch_count, 1);
            device_batch_vector<U>  bc(batch_count, 1);
            device_batch_vector<T>  bs(batch_count, 1);
            for(int b = 0; b < batch_count; b++)
            {
                CHECK_HIP_ERROR(hipMemcpy(ba[b], ha[b], sizeof(T), hipMemcpyHostToDevice));
                CHECK_HIP_ERROR(hipMemcpy(bb[b], hb[b], sizeof(T), hipMemcpyHostToDevice));
                CHECK_HIP_ERROR(hipMemcpy(bc[b], hc[b], sizeof(U), hipMemcpyHostToDevice));
                CHECK_HIP_ERROR(hipMemcpy(bs[b], hs[b], sizeof(T), hipMemcpyHostToDevice));
            }
            CHECK_HIP_ERROR(hipMemcpy(da, ba, sizeof(T*) * batch_count, hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(db, bb, sizeof(T*) * batch_count, hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(dc, bc, sizeof(U*) * batch_count, hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(ds, bs, sizeof(T*) * batch_count, hipMemcpyHostToDevice));

            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            CHECK_ROCBLAS_ERROR((rocblas_rotg_batched<T, U>(handle, da, db, dc, ds, batch_count)));

            host_vector<T> ra[batch_count];
            host_vector<T> rb[batch_count];
            host_vector<U> rc[batch_count];
            host_vector<T> rs[batch_count];
            for(int b = 0; b < batch_count; b++)
            {
                ra[b] = host_vector<T>(1);
                rb[b] = host_vector<T>(1);
                rc[b] = host_vector<U>(1);
                rs[b] = host_vector<T>(1);
                CHECK_HIP_ERROR(hipMemcpy(ra[b], ba[b], sizeof(T), hipMemcpyDeviceToHost));
                CHECK_HIP_ERROR(hipMemcpy(rb[b], bb[b], sizeof(T), hipMemcpyDeviceToHost));
                CHECK_HIP_ERROR(hipMemcpy(rc[b], bc[b], sizeof(U), hipMemcpyDeviceToHost));
                CHECK_HIP_ERROR(hipMemcpy(rs[b], bs[b], sizeof(T), hipMemcpyDeviceToHost));
            }

            if(arg.unit_check)
            {
                unit_check_general<T>(1, 1, batch_count, 1, ra, ca);
                unit_check_general<T>(1, 1, batch_count, 1, rb, cb);
                unit_check_general<U>(1, 1, batch_count, 1, rc, cc);
                unit_check_general<T>(1, 1, batch_count, 1, rs, cs);
            }

            if(arg.norm_check)
            {
                norm_error_device = norm_check_general<T>('F', 1, 1, batch_count, 1, ra, ca);
                norm_error_device += norm_check_general<T>('F', 1, 1, batch_count, 1, rb, cb);
                norm_error_device += norm_check_general<U>('F', 1, 1, batch_count, 1, rc, cc);
                norm_error_device += norm_check_general<T>('F', 1, 1, batch_count, 1, rs, cs);
            }
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = 100;
        // Device mode will be much quicker
        // (TODO: or is there another reason we are typically using host_mode for timing?)
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

        device_vector<T*, 0, T> da(batch_count);
        device_vector<T*, 0, T> db(batch_count);
        device_vector<U*, 0, U> dc(batch_count);
        device_vector<T*, 0, T> ds(batch_count);
        device_batch_vector<T>  ba(batch_count, 1);
        device_batch_vector<T>  bb(batch_count, 1);
        device_batch_vector<U>  bc(batch_count, 1);
        device_batch_vector<T>  bs(batch_count, 1);
        for(int b = 0; b < batch_count; b++)
        {
            CHECK_HIP_ERROR(hipMemcpy(ba[b], ha[b], sizeof(T), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(bb[b], hb[b], sizeof(T), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(bc[b], hc[b], sizeof(U), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(bs[b], hs[b], sizeof(T), hipMemcpyHostToDevice));
        }
        CHECK_HIP_ERROR(hipMemcpy(da, ba, sizeof(T*) * batch_count, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(db, bb, sizeof(T*) * batch_count, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dc, bc, sizeof(U*) * batch_count, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(ds, bs, sizeof(T*) * batch_count, hipMemcpyHostToDevice));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_rotg_batched<T, U>(handle, da, db, dc, ds, batch_count);
        }
        gpu_time_used = get_time_us(); // in microseconds
        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_rotg_batched<T, U>(handle, da, db, dc, ds, batch_count);
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
