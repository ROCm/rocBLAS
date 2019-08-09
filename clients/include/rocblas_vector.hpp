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
/*! \brief  base-class to allocate/deallocate device memory */
template <typename T, size_t PAD, typename U>
class d_vector
{
protected:
    size_t size, bytes;

#ifdef GOOGLE_TEST
    U guard[PAD];
    d_vector(size_t s)
        : size(s)
        , bytes((s + PAD * 2) * sizeof(T))
    {
        // Initialize guard with random data
        if(PAD > 0)
        {
            rocblas_init_nan(guard, PAD);
        }
    }
#else
    d_vector(size_t s)
        : size(s)
        , bytes(s ? s * sizeof(T) : sizeof(T))
    {
    }
#endif

    T* device_vector_setup()
    {
        T* d;
        if((hipMalloc)(&d, bytes) != hipSuccess)
        {
            static char* lc = setlocale(LC_NUMERIC, "");
            fprintf(stderr, "Error allocating %'zu bytes (%zu GB)\n", bytes, bytes >> 30);
            d = nullptr;
        }
#ifdef GOOGLE_TEST
        else
        {
            if(PAD > 0)
            {
                // Copy guard to device memory before allocated memory
                hipMemcpy(d, guard, sizeof(guard), hipMemcpyHostToDevice);

                // Point to allocated block
                d += PAD;

                // Copy guard to device memory after allocated memory
                hipMemcpy(d + size, guard, sizeof(guard), hipMemcpyHostToDevice);
            }
        }
#endif
        return d;
    }

    void device_vector_teardown(T* d)
    {
        if(d != nullptr)
        {
#ifdef GOOGLE_TEST
            if(PAD > 0)
            {
                U host[PAD];

                // Copy device memory after allocated memory to host
                hipMemcpy(host, d + size, sizeof(guard), hipMemcpyDeviceToHost);

                // Make sure no corruption has occurred
                EXPECT_EQ(memcmp(host, guard, sizeof(guard)), 0);

                // Point to guard before allocated memory
                d -= PAD;

                // Copy device memory after allocated memory to host
                hipMemcpy(host, d, sizeof(guard), hipMemcpyDeviceToHost);

                // Make sure no corruption has occurred
                EXPECT_EQ(memcmp(host, guard, sizeof(guard)), 0);
            }
#endif
            // Free device memory
            CHECK_HIP_ERROR((hipFree)(d));
        }
    }
};

/* ============================================================================================ */
/*! \brief  pseudo-vector subclass which uses a batch of device memory pointers and 
            an array of pointers in host memory*/
template <typename T, size_t PAD = 4096, typename U = T>
class device_batch_vector : private d_vector<T, PAD, U>
{
public:
    explicit device_batch_vector(size_t b, size_t s)
        : batch(b)
        , d_vector<T, PAD, U>(s)
    {
        data = (T**)malloc(batch * sizeof(T*));
        for(int b = 0; b < batch; ++b)
            data[b] = this->device_vector_setup();
    }

    ~device_batch_vector()
    {
        if(data != nullptr)
        {
            for(int b = 0; b < batch; ++b)
                this->device_vector_teardown(data[b]);
            free(data);
        }
    }

    T* operator[](int n)
    {
        return data[n];
    }

    operator T**()
    {
        return data;
    }

    // Disallow copying or assigning
    device_batch_vector(const device_batch_vector&) = delete;
    device_batch_vector& operator=(const device_batch_vector&) = delete;

private:
    T**    data;
    size_t batch;
};

/* ============================================================================================ */
/*! \brief  pseudo-vector subclass which uses device memory */
template <typename T, size_t PAD = 4096, typename U = T>
class device_vector : private d_vector<T, PAD, U>
{
public:
    // Must wrap constructor and destructor in functions to allow Google Test macros to work
    explicit device_vector(size_t s)
        : d_vector<T, PAD, U>(s)
    {
        data = this->device_vector_setup();
    }

    ~device_vector()
    {
        this->device_vector_teardown(data);
    }

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
    T* data;
};

/* ============================================================================================ */
/*! \brief  pseudo-vector subclass which uses host memory */
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
