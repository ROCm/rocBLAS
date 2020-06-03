/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
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
    const bool FORTRAN = arg.fortran;
    auto       rocblas_rotg_batched_fn
        = FORTRAN ? rocblas_rotg_batched<T, U, true> : rocblas_rotg_batched<T, U, false>;

    rocblas_int         batch_count = 5;
    static const size_t safe_size   = 1;

    rocblas_local_handle   handle;
    device_batch_vector<T> da(1, 1, batch_count);
    device_batch_vector<T> db(1, 1, batch_count);
    device_batch_vector<U> dc(1, 1, batch_count);
    device_batch_vector<T> ds(1, 1, batch_count);
    CHECK_DEVICE_ALLOCATION(da.memcheck());
    CHECK_DEVICE_ALLOCATION(db.memcheck());
    CHECK_DEVICE_ALLOCATION(dc.memcheck());
    CHECK_DEVICE_ALLOCATION(ds.memcheck());

    EXPECT_ROCBLAS_STATUS((rocblas_rotg_batched_fn(nullptr,
                                                   da.ptr_on_device(),
                                                   db.ptr_on_device(),
                                                   dc.ptr_on_device(),
                                                   ds.ptr_on_device(),
                                                   batch_count)),
                          rocblas_status_invalid_handle);
    EXPECT_ROCBLAS_STATUS((rocblas_rotg_batched_fn(handle,
                                                   nullptr,
                                                   db.ptr_on_device(),
                                                   dc.ptr_on_device(),
                                                   ds.ptr_on_device(),
                                                   batch_count)),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS((rocblas_rotg_batched_fn(handle,
                                                   da.ptr_on_device(),
                                                   nullptr,
                                                   dc.ptr_on_device(),
                                                   ds.ptr_on_device(),
                                                   batch_count)),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS((rocblas_rotg_batched_fn(handle,
                                                   da.ptr_on_device(),
                                                   db.ptr_on_device(),
                                                   nullptr,
                                                   ds.ptr_on_device(),
                                                   batch_count)),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS((rocblas_rotg_batched_fn(handle,
                                                   da.ptr_on_device(),
                                                   db.ptr_on_device(),
                                                   dc.ptr_on_device(),
                                                   nullptr,
                                                   batch_count)),
                          rocblas_status_invalid_pointer);
}

template <typename T, typename U = T>
void testing_rotg_batched(const Arguments& arg)
{
    const bool FORTRAN = arg.fortran;
    auto       rocblas_rotg_batched_fn
        = FORTRAN ? rocblas_rotg_batched<T, U, true> : rocblas_rotg_batched<T, U, false>;

    const int            TEST_COUNT  = 100;
    rocblas_int          batch_count = arg.batch_count;
    rocblas_local_handle handle;

    double  gpu_time_used, cpu_time_used;
    double  norm_error_host = 0.0, norm_error_device = 0.0;
    const U rel_error = std::numeric_limits<U>::epsilon() * 1000;

    // check to prevent undefined memory allocation error
    if(batch_count <= 0)
    {
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        EXPECT_ROCBLAS_STATUS(
            (rocblas_rotg_batched_fn)(handle, nullptr, nullptr, nullptr, nullptr, batch_count),
            rocblas_status_success);
        return;
    }

    // Initial Data on CPU
    host_batch_vector<T> ha(1, 1, batch_count);
    host_batch_vector<T> hb(1, 1, batch_count);
    host_batch_vector<U> hc(1, 1, batch_count);
    host_batch_vector<T> hs(1, 1, batch_count);

    for(int i = 0; i < TEST_COUNT; i++)
    {
        host_batch_vector<T> ca(1, 1, batch_count);
        host_batch_vector<T> cb(1, 1, batch_count);
        host_batch_vector<U> cc(1, 1, batch_count);
        host_batch_vector<T> cs(1, 1, batch_count);

        rocblas_init(ha, true);
        rocblas_init(hb, false);
        rocblas_init(hc, false);
        rocblas_init(hs, false);
        ca.copy_from(ha);
        cb.copy_from(hb);
        cc.copy_from(hc);
        cs.copy_from(hs);

        cpu_time_used = get_time_us();
        for(int b = 0; b < batch_count; b++)
        {
            cblas_rotg<T, U>(ca[b], cb[b], cc[b], cs[b]);
        }
        cpu_time_used = get_time_us() - cpu_time_used;

        // Test rocblas_pointer_mode_host
        {
            host_batch_vector<T> ra(1, 1, batch_count);
            host_batch_vector<T> rb(1, 1, batch_count);
            host_batch_vector<U> rc(1, 1, batch_count);
            host_batch_vector<T> rs(1, 1, batch_count);
            ra.copy_from(ha);
            rb.copy_from(hb);
            rc.copy_from(hc);
            rs.copy_from(hs);

            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

            CHECK_ROCBLAS_ERROR((rocblas_rotg_batched_fn(handle, ra, rb, rc, rs, batch_count)));

            if(arg.unit_check)
            {
                near_check_general<T>(1, 1, 1, ra, ca, rel_error, batch_count);
                near_check_general<T>(1, 1, 1, rb, cb, rel_error, batch_count);
                near_check_general<U>(1, 1, 1, rc, cc, rel_error, batch_count);
                near_check_general<T>(1, 1, 1, rs, cs, rel_error, batch_count);
            }

            if(arg.norm_check)
            {
                norm_error_host = norm_check_general<T>('F', 1, 1, 1, ra, ca, batch_count);
                norm_error_host += norm_check_general<T>('F', 1, 1, 1, rb, cb, batch_count);
                norm_error_host += norm_check_general<U>('F', 1, 1, 1, rc, cc, batch_count);
                norm_error_host += norm_check_general<T>('F', 1, 1, 1, rs, cs, batch_count);
            }
        }

        // Test rocblas_pointer_mode_device
        {
            device_batch_vector<T> da(1, 1, batch_count);
            device_batch_vector<T> db(1, 1, batch_count);
            device_batch_vector<U> dc(1, 1, batch_count);
            device_batch_vector<T> ds(1, 1, batch_count);

            CHECK_HIP_ERROR(da.transfer_from(ha));
            CHECK_HIP_ERROR(db.transfer_from(hb));
            CHECK_HIP_ERROR(dc.transfer_from(hc));
            CHECK_HIP_ERROR(ds.transfer_from(hs));

            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            CHECK_ROCBLAS_ERROR((rocblas_rotg_batched_fn(handle,
                                                         da.ptr_on_device(),
                                                         db.ptr_on_device(),
                                                         dc.ptr_on_device(),
                                                         ds.ptr_on_device(),
                                                         batch_count)));

            host_batch_vector<T> ra(1, 1, batch_count);
            host_batch_vector<T> rb(1, 1, batch_count);
            host_batch_vector<U> rc(1, 1, batch_count);
            host_batch_vector<T> rs(1, 1, batch_count);
            CHECK_HIP_ERROR(ra.transfer_from(da));
            CHECK_HIP_ERROR(rb.transfer_from(db));
            CHECK_HIP_ERROR(rc.transfer_from(dc));
            CHECK_HIP_ERROR(rs.transfer_from(ds));

            if(arg.unit_check)
            {
                near_check_general<T>(1, 1, 1, ra, ca, batch_count, rel_error);
                near_check_general<T>(1, 1, 1, rb, cb, batch_count, rel_error);
                near_check_general<U>(1, 1, 1, rc, cc, batch_count, rel_error);
                near_check_general<T>(1, 1, 1, rs, cs, batch_count, rel_error);
            }

            if(arg.norm_check)
            {
                norm_error_device = norm_check_general<T>('F', 1, 1, 1, ra, ca, batch_count);
                norm_error_device += norm_check_general<T>('F', 1, 1, 1, rb, cb, batch_count);
                norm_error_device += norm_check_general<U>('F', 1, 1, 1, rc, cc, batch_count);
                norm_error_device += norm_check_general<T>('F', 1, 1, 1, rs, cs, batch_count);
            }
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;
        // Device mode will be much quicker
        // (TODO: or is there another reason we are typically using host_mode for timing?)
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

        device_batch_vector<T> da(1, 1, batch_count);
        device_batch_vector<T> db(1, 1, batch_count);
        device_batch_vector<U> dc(1, 1, batch_count);
        device_batch_vector<T> ds(1, 1, batch_count);
        CHECK_HIP_ERROR(da.transfer_from(ha));
        CHECK_HIP_ERROR(db.transfer_from(hb));
        CHECK_HIP_ERROR(dc.transfer_from(hc));
        CHECK_HIP_ERROR(ds.transfer_from(hs));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_rotg_batched_fn(handle,
                                    da.ptr_on_device(),
                                    db.ptr_on_device(),
                                    dc.ptr_on_device(),
                                    ds.ptr_on_device(),
                                    batch_count);
        }
        gpu_time_used = get_time_us(); // in microseconds
        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_rotg_batched_fn(handle,
                                    da.ptr_on_device(),
                                    db.ptr_on_device(),
                                    dc.ptr_on_device(),
                                    ds.ptr_on_device(),
                                    batch_count);
        }
        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        rocblas_cout << "rocblas-us,CPU-us";
        if(arg.norm_check)
            rocblas_cout << ",norm_error_host_ptr,norm_error_device";
        rocblas_cout << std::endl;

        rocblas_cout << gpu_time_used << "," << cpu_time_used;
        if(arg.norm_check)
            rocblas_cout << ',' << norm_error_host << ',' << norm_error_device;
        rocblas_cout << std::endl;
    }
}
