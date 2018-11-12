/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <vector>

#include "rocblas.hpp"
#include "arg_check.h"
#include "utility.h"
#include "cblas_interface.h"
#include "norm.h"
#include "unit.h"
#include <complex.h>

using namespace std;

/* ============================================================================================ */

template <typename T>
rocblas_status testing_bandwidth(Arguments argus)
{

    rocblas_int N = 25 * 1e7;

    rocblas_int incx = 1;

    rocblas_status status = rocblas_status_success;

    rocblas_int size_X = N * incx;
    T alpha            = 2.0;

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<T> hx(size_X);
    host_vector<T> hz(size_X);

    if(N > hx.max_size())
    {
        printf("max_size of a std::vector is %lu, please reduce the input size \n", hx.max_size());
        return status;
    }

    double gpu_time_used, gpu_bandwidth;

    rocblas_local_handle handle;

    // allocate memory on device
    device_vector<T> dx(size_X);
    device_vector<T> dy(size_X);
    device_vector<T> d_rocblas_result(1);
    if(!dx || !dy || !d_rocblas_result)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Initial Data on CPU
    rocblas_seedrand();
    rocblas_init<T>(hx, 1, N, incx);

    // hz = hx;

    // copy data from CPU to device,
    CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T) * N * incx, hipMemcpyHostToDevice));

    printf("Bandwidth     MByte    GPU (GB/s)    Time (us) \n");

    /* =====================================================================
         Bandwidth
    =================================================================== */

    for(size_t size = 1e6; size <= N; size *= 2)
    {

        CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T) * size * incx, hipMemcpyHostToDevice));

        gpu_time_used = get_time_us(); // in microseconds

        // scal dx
        status = rocblas_scal<T>(handle, N, &alpha, dx, incx);
        if(status != rocblas_status_success)
        {
            break;
            return status;
        }

        //       hipMemcpy(dy, dx, sizeof(T)*size*incx, hipMemcpyDeviceToDevice);

        //       hipMemset(dx, 0, size*sizeof(T));

        gpu_time_used = get_time_us() - gpu_time_used;

        gpu_bandwidth = 2 * size * sizeof(T) / 1e6 / (gpu_time_used); // in GB/s

// CPU result, before GPU result copy back to CPU
#pragma unroll
        for(size_t i = 0; i < size; i++)
        {
            hz[i] = alpha * hx[i];
        }

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hx, dx, sizeof(T) * size * incx, hipMemcpyDeviceToHost));

        // check error with CPU result
        for(size_t i = 0; i < size; i++)
        {
            T error = fabs(hz[i] - hx[i]);
            if(error > 0)
            {
                printf("error is %f, CPU=%f, GPU=%f, at elment %zu", error, hz[i], hx[i], i);
                break;
            }
        }

        printf("              %6.2f     %8.2f  %8.2f        \n",
               (int)size * sizeof(T) / 1e6,
               gpu_bandwidth,
               gpu_time_used);
    }

    return rocblas_status_success;
}
