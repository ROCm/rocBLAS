/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <stdlib.h>
#include <stdio.h>
#include <vector>

#include "rocblas.h"
#include "utility.h"
#include "cblas_interface.h"
#include <typeinfo>

using namespace std;

/* ============================================================================================ */

int main()
{

    rocblas_int N = 10240;
    rocblas_status status = rocblas_success;
    float alpha = 10.0;

    int norm_check = 0;

    //Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    vector<float> hx(N);
    vector<float> hz(N);
    float *dx;

    double gpu_time_used;
    float rocblas_error = 0.0;

    rocblas_handle handle;
    rocblas_create(&handle);

    //allocate memory on device
    hipMalloc(&dx, N * sizeof(float));

    //Initial Data on CPU
    srand(1);
    rocblas_init<float>(hx, 1, N, 1);

    //copy vector is easy in SfloatL; hz = hx: save a copy in hz which will be output of CPU BLAS
    hz = hx;

    hipMemcpy(dx, hx.data(), sizeof(float)*N, hipMemcpyHostToDevice);

    printf("N    rocblas    (us)     \n");

    /* =====================================================================
         ROCBLAS
    =================================================================== */

    gpu_time_used = get_time_us();// in microseconds


    //library interface
    status = rocblas_scal<float>(handle,
                    N,
                    &alpha,
                    dx, 1);
    if (status != rocblas_success) {
        return status;
    }


    gpu_time_used = get_time_us() - gpu_time_used;

    //copy output from device to CPU
    hipMemcpy(hx.data(), dx, sizeof(float)*N, hipMemcpyDeviceToHost);

    for(int i=0;i<10;i++){
        printf("hx[%d]=%f   ", i, hx.data()[i]);
    }

    printf("%d    %8.2f     ---     \n", (int)N, gpu_time_used);


    hipFree(dx);
    rocblas_destroy(handle);
    return rocblas_success;
}
