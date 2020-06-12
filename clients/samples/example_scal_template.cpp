/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "rocblas.hpp"
#include "rocblas_init.hpp"
#include "utility.hpp"
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>

/* ============================================================================================ */

int main()
{

    rocblas_int    N      = 10240;
    rocblas_status status = rocblas_status_success;
    float          alpha  = 10.0;

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    std::vector<float> hx(N);
    std::vector<float> hz(N);
    float*             dx;

    double gpu_time_used;

    rocblas_handle handle;
    rocblas_create_handle(&handle);

    // allocate memory on device
    hipMalloc(&dx, N * sizeof(float));

    // Initial Data on CPU
    srand(1);
    rocblas_init<float>(hx, 1, N, 1);

    // copy vector is easy in STL; hz = hx: save a copy in hz which will be output of CPU BLAS
    hz = hx;

    hipMemcpy(dx, hx.data(), sizeof(float) * N, hipMemcpyHostToDevice);

    printf("N        rocblas(us)     \n");

    gpu_time_used = get_time_us_sync_device(); // in microseconds

    /* =====================================================================
         ROCBLAS  C++ template interface
    =================================================================== */

    status = rocblas_scal<float>(handle, N, &alpha, dx, 1);
    if(status != rocblas_status_success)
    {
        return status;
    }

    gpu_time_used = get_time_us_sync_device() - gpu_time_used;

    // copy output from device to CPU
    hipMemcpy(hx.data(), dx, sizeof(float) * N, hipMemcpyDeviceToHost);

    // verify rocblas_scal result
    for(rocblas_int i = 0; i < N; i++)
    {
        if(hz[i] * alpha != hx[i])
        {
            printf("error in element %d: CPU=%f, GPU=%f ", i, hz[i] * alpha, hx[i]);
            break;
        }
    }

    printf("%d    %8.2f        \n", (int)N, gpu_time_used);

    hipFree(dx);
    rocblas_destroy_handle(handle);
    return rocblas_status_success;
}
