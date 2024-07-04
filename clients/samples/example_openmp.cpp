/* ************************************************************************
 * Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

/*
  README:  Test multiple OpenMP threads
           The main thread creates NUM_THREADS handles/streams
           Each OpenMP thread calls a rocblas routine asscociated with one handle, one stream
           The main thread finally destroy all handles/streams
           An alternate valid way is each thread creates its own handle/stream and destroy it
  locally.
           But in the second way, the handles/streams can not persist across multiple parallel
  regions.
           In this example, we have two parallel regions

           It is NOT recommended that multiple thread share the same rocblas handle.
           Yet, it is safe that multiple thread shared the same stream.
           If users do not create streams explicitely like what I am doing here,
           all rocblas routine take the NULL (0) stream.
*/
#include "client_utility.hpp"
#include "rocblas.hpp"
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <vector>

#define NUM_THREADS 4

/* ============================================================================================ */

int main()
{
    constexpr rocblas_int N     = 102400;
    float                 alpha = 10.0;

    omp_set_num_threads(NUM_THREADS);
    int thread_id;
    printf("%d OpenMP threads performing rocblas_scal \n", NUM_THREADS);

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    std::vector<float> hx(N * NUM_THREADS);

#if 0
    std::vector<float> hz(N * NUM_THREADS);
#endif

    float *dx, *dy;

    double gpu_time_used;

    rocblas_handle handles[NUM_THREADS];
    hipStream_t    streams[NUM_THREADS];

    // Create handle/stream have overhead
    for(rocblas_int i = 0; i < NUM_THREADS; i++)
    {
        rocblas_create_handle(&handles[i]);
        hipStreamCreate(&streams[i]);
    }

    // allocate memory on device
    hipMalloc(&dx, N * NUM_THREADS * sizeof(float));
    hipMalloc(&dy, N * NUM_THREADS * sizeof(float));

    // Initial Data on CPU
    srand(1);
    rocblas_init<float>(hx, 1, N * NUM_THREADS, 1);

#if 0
    // copy vector is easy in STL; hz = hx: save a copy in hz which will be output of CPU BLAS
    hz = hx;
#endif

    hipMemcpy(dx, hx.data(), sizeof(float) * N * NUM_THREADS, hipMemcpyHostToDevice);

    printf("N        rocblas(us)     \n");

    gpu_time_used = get_time_us_sync_device(); // in microseconds

// 1st parallel rocblas routine call : scal x
// spawn openmp threads
#ifdef _OPENMP
#pragma omp parallel private(thread_id)
#endif
    {

        thread_id = omp_get_thread_num(); // thread_id from 0,...,NUM_THREADS-1
        // associate each handle with a stream
        rocblas_set_stream(handles[thread_id], streams[thread_id]);

        /* =====================================================================
             ROCBLAS  template interface
        =================================================================== */
        rocblas_scal<float>(handles[thread_id], N, &alpha, dx + thread_id * N, 1);

        // Blocks until all stream has completed all operations.
        hipStreamSynchronize(streams[thread_id]);
    }

// 2nd parallel rocblas routine call : copy x to y
// spawn openmp threads
#ifdef _OPENMP
#pragma omp parallel private(thread_id)
#endif
    {

        thread_id = omp_get_thread_num(); // thread_id from 0,...,NUM_THREADS-1
        // associate each handle with a stream
        rocblas_set_stream(handles[thread_id], streams[thread_id]);

        /* =====================================================================
             ROCBLAS  template interface
        =================================================================== */
        rocblas_copy<float>(handles[thread_id], N, dx + thread_id * N, 1, dy + thread_id * N, 1);

        // Blocks until all stream has completed all operations.
        hipStreamSynchronize(streams[thread_id]);
    }

    gpu_time_used = get_time_us_sync_device() - gpu_time_used;

    // copy output from device to CPU
    hipMemcpy(hx.data(), dy, sizeof(float) * N * NUM_THREADS, hipMemcpyDeviceToHost);

#if 0
    //verify rocblas_scal result
    for(rocblas_int i=0;i<N*NUM_THREADS;i++){
        if(hz[i] * alpha != hx[i]){
            printf("error in element %d: CPU=%f, GPU=%f ", i, hz[i]*alpha, hx[i]);
            break;
        }
    }

#endif

    printf("%d    %8.2f         \n", (int)N * NUM_THREADS, gpu_time_used);

    hipFree(dx);
    hipFree(dy);

    // Destroy handle/streams
    for(rocblas_int i = 0; i < NUM_THREADS; i++)
    {
        rocblas_destroy_handle(handles[i]);
        hipStreamDestroy(streams[i]);
    }

    return 0;
}
