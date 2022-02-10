/* ************************************************************************
 * Copyright 2018-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas_random.hpp"

// Random number generator
// Note: We do not use random_device to initialize the RNG, because we want
// repeatability in case of test failure. TODO: Add seed as an optional CLI
// argument, and print the seed on output, to ensure repeatability.
rocblas_rng_t g_rocblas_seed(69069); // A fixed seed to start at

// This records the main thread ID at startup
std::thread::id g_main_thread_id = std::this_thread::get_id();

// For the main thread, we use g_rocblas_seed; for other threads, we start with a different seed but
// deterministically based on the thread id's hash function.
thread_local rocblas_rng_t t_rocblas_rng = get_seed();

thread_local int t_rocblas_rand_idx;

// length to allow use as bitmask to wraparound
#define RANDLEN 1024
#define RANDWIN 256
#define RANDBUF RANDLEN + RANDWIN
static thread_local int    t_rand_init = 0;
static thread_local float  t_rand_f_array[RANDBUF];
static thread_local double t_rand_d_array[RANDBUF];

/* ============================================================================================ */

float rocblas_uniform_int_1_10()
{
    if(!t_rand_init)
    {
        for(int i = 0; i < RANDBUF; i++)
        {
            t_rand_f_array[i]
                = (float)std::uniform_int_distribution<unsigned>(1, 10)(t_rocblas_rng);
            t_rand_d_array[i] = (double)t_rand_f_array[i];
        }
        t_rand_init = 1;
    }
    t_rocblas_rand_idx = (t_rocblas_rand_idx + 1) & (RANDLEN - 1);
    return t_rand_f_array[t_rocblas_rand_idx];
}

inline int pseudo_rand_ptr_offset()
{
    t_rocblas_rand_idx = (t_rocblas_rand_idx + 1) & (RANDWIN - 1);
    return t_rocblas_rand_idx;
}

void rocblas_uniform_int_1_10_run_float(float* ptr, size_t num)
{
    if(!t_rand_init)
        rocblas_uniform_int_1_10();

    for(size_t i = 0; i < num; i += RANDLEN)
    {
        float* rptr = t_rand_f_array + pseudo_rand_ptr_offset();
        size_t n    = i + RANDLEN < num ? RANDLEN : num - i;
        memcpy(ptr, rptr, sizeof(float) * n);
        ptr += RANDLEN;
    }
}

void rocblas_uniform_int_1_10_run_double(double* ptr, size_t num)
{
    if(!t_rand_init)
        rocblas_uniform_int_1_10();

    for(size_t i = 0; i < num; i += RANDLEN)
    {
        double* rptr = t_rand_d_array + pseudo_rand_ptr_offset();
        size_t  n    = i + RANDLEN < num ? RANDLEN : num - i;
        memcpy(ptr, rptr, sizeof(double) * n);
        ptr += RANDLEN;
    }
}

void rocblas_uniform_int_1_10_run_float_complex(rocblas_float_complex* ptr, size_t num)
{
    if(!t_rand_init)
        rocblas_uniform_int_1_10();

    constexpr int rand_len = RANDLEN / 2;
    for(size_t i = 0; i < num; i += rand_len)
    {
        float* rptr = t_rand_f_array + pseudo_rand_ptr_offset();
        size_t n    = i + rand_len < num ? rand_len : num - i;
        memcpy(ptr, rptr, sizeof(float) * 2 * n);
        ptr += rand_len;
    }
}

void rocblas_uniform_int_1_10_run_double_complex(rocblas_double_complex* ptr, size_t num)
{
    if(!t_rand_init)
        rocblas_uniform_int_1_10();

    constexpr int rand_len = RANDLEN / 2;
    for(size_t i = 0; i < num; i += rand_len)
    {
        double* rptr = t_rand_d_array + pseudo_rand_ptr_offset();
        size_t  n    = i + rand_len < num ? rand_len : num - i;
        memcpy(ptr, rptr, sizeof(double) * 2 * n);
        ptr += rand_len;
    }
}

#undef RANDLEN
#undef RANDWIN
#undef RANDBUF
