/* ************************************************************************
 * Copyright 2018-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef ROCBLAS_VECTOR_H_
#define ROCBLAS_VECTOR_H_

#include "rocblas.h"
#include "rocblas_init.hpp"
#include "rocblas_test.hpp"
#include <cinttypes>
#include <cstdio>
#include <locale.h>
#include <vector>

/* ============================================================================================ */
/*! \brief  pseudo-vector class which uses device memory */

template <typename T, size_t PAD = 4096>
class device_vector
{
#ifdef GOOGLE_TEST

    T guard[PAD];

    void device_vector_setup()
    {
        if((hipMalloc)(&data, bytes) != hipSuccess)
        {
            static char* lc = setlocale(LC_NUMERIC, "");
            fprintf(stderr, "Error allocating %'zu bytes (%zu GB)\n", bytes, bytes >> 30);
            data = nullptr;
        }
        else
        {
            // Initialize guard with random data
            rocblas_init_nan(guard, PAD);

            // Copy guard to device memory before allocated memory
            CHECK_HIP_ERROR(hipMemcpy(data, guard, sizeof(guard), hipMemcpyHostToDevice));

            // Point to allocated block
            data += PAD;

            // Copy guard to device memory after allocated memory
            CHECK_HIP_ERROR(hipMemcpy(data + size, guard, sizeof(guard), hipMemcpyHostToDevice));
        }
    }

    void device_vector_teardown()
    {
        if(data != nullptr)
        {
            T host[PAD];

            // Copy device memory after allocated memory to host
            CHECK_HIP_ERROR(hipMemcpy(host, data + size, sizeof(guard), hipMemcpyDeviceToHost));

            // Make sure no corruption has occurred
            EXPECT_EQ(memcmp(host, guard, sizeof(guard)), 0);

            // Point to guard before allocated memory
            data -= PAD;

            // Copy device memory after allocated memory to host
            CHECK_HIP_ERROR(hipMemcpy(host, data, sizeof(guard), hipMemcpyDeviceToHost));

            // Make sure no corruption has occurred
            EXPECT_EQ(memcmp(host, guard, sizeof(guard)), 0);

            // Free device memory
            CHECK_HIP_ERROR((hipFree)(data));
        }
    }

public:
    // Must wrap constructor and destructor in functions to allow Google Test macros to work
    explicit device_vector(size_t size)
        : size(size)
        , bytes((size + PAD * 2) * sizeof(T))
    {
        device_vector_setup();
    }

    ~device_vector()
    {
        device_vector_teardown();
    }

#else // GOOGLE_TEST

    // Code without memory guards

public:
    explicit device_vector(size_t size)
        : size(size)
        , bytes(size ? size * sizeof(T) : sizeof(T))
    {
        if((hipMalloc)(&data, bytes) != hipSuccess)
        {
            static char* lc = setlocale(LC_NUMERIC, "");
            fprintf(stderr, "Error allocating %'zu bytes (%'zu GB)\n", bytes, bytes >> 30);
            data = nullptr;
        }
    }

    ~device_vector()
    {
        if(data != nullptr)
            CHECK_HIP_ERROR((hipFree)(data));
    }

#endif // GOOGLE_TEST

public:
    // Decay into pointer wherever pointer is expected
    operator T*()
    {
        return data;
    }
    operator const T*() const
    {
        return data;
    }

    // Tell whether malloc failed
    explicit operator bool() const
    {
        return data != nullptr;
    }

    // Disallow copying or assigning
    device_vector(const device_vector&) = delete;
    device_vector& operator=(const device_vector&) = delete;

private:
    T*           data;
    const size_t size, bytes;
};

/* ============================================================================================ */
/*! \brief  pseudo-vector class which uses host memory */
template <typename T>
struct host_vector : std::vector<T>
{
    // Inherit constructors
    using std::vector<T>::vector;

    // Decay into pointer wherever pointer is expected
    operator T*()
    {
        return this->data();
    }
    operator const T*() const
    {
        return this->data();
    }
};

#endif
