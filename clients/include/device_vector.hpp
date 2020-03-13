//
// Copyright 2018-2020 Advanced Micro Devices, Inc.
//
#pragma once

#include "d_vector.hpp"

//
// Local declaration of the host vector.
//
template <typename T>
class host_vector;

//!
//! @brief pseudo-vector subclass which uses device memory
//!
template <typename T, size_t PAD = 4096, typename U = T>
class device_vector : private d_vector<T, PAD, U>
{

public:
    //!
    //! @brief Disallow copying.
    //!
    device_vector(const device_vector&) = delete;

    //!
    //! @brief Disallow assigning
    //!
    device_vector& operator=(const device_vector&) = delete;

    //!
    //! @brief Constructor.
    //! @param n The length of the vector.
    //! @param inc The increment.
    //! @remark Must wrap constructor and destructor in functions to allow Google Test macros to work
    //!
    explicit device_vector(rocblas_int n, rocblas_int inc)
        : d_vector<T, PAD, U>(n * std::abs(inc))
        , m_n(n)
        , m_inc(inc)
    {
        this->m_data = this->device_vector_setup();
    }

    //!
    //! @brief Constructor (kept for backward compatibility)
    //! @param s the size.
    //! @remark Must wrap constructor and destructor in functions to allow Google Test macros to work
    //!
    explicit device_vector(size_t s)
        : d_vector<T, PAD, U>(s)
        , m_n(s)
        , m_inc(1)
    {
        this->m_data = this->device_vector_setup();
    }

    //!
    //! @brief Destructor.
    //!
    ~device_vector()
    {
        this->device_vector_teardown(this->m_data);
        this->m_data = nullptr;
    }

    //!
    //! @brief Returns the length of the vector.
    //!
    rocblas_int n() const
    {
        return this->m_n;
    }

    //!
    //! @brief Returns the increment of the vector.
    //!
    rocblas_int inc() const
    {
        return this->m_inc;
    }

    //!
    //! @brief Returns the batch count (always 1).
    //!
    rocblas_int batch_count() const
    {
        return 1;
    }

    //!
    //! @brief Returns the stride (out of context, always 0)
    //!
    rocblas_stride stride() const
    {
        return 0;
    }

    //!
    //! @brief Decay into pointer wherever pointer is expected.
    //!
    operator T*()
    {
        return this->m_data;
    }

    //!
    //! @brief Decay into constant pointer wherever pointer is expected.
    //!
    operator const T*() const
    {
        return this->m_data;
    }

    //!
    //! @brief Tell whether malloc failed.
    //!
    explicit operator bool() const
    {
        return nullptr != this->m_data;
    }

    //!
    //! @brief Transfer data from a host vector.
    //! @param that The host vector.
    //! @return the hip error.
    //!
    hipError_t transfer_from(const host_vector<T>& that)
    {
        return hipMemcpy(
            this->m_data, (const T*)that, this->nmemb() * sizeof(T), hipMemcpyHostToDevice);
    }

    hipError_t memcheck() const
    {
        if(*this)
            return hipSuccess;
        else
            return hipErrorOutOfMemory;
    }

private:
    size_t      m_size{};
    rocblas_int m_n{};
    rocblas_int m_inc{};
    T*          m_data{};
};
