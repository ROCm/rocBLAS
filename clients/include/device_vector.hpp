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
class device_vector : d_vector<T, PAD, U>
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
    //!
    explicit device_vector(size_t n, rocblas_int inc = 1)
        : d_vector<T, PAD, U>{n * std::abs(inc)}
        , m_n{n}
        , m_inc{inc}
        , m_data{this->device_vector_setup()}
    {
    }

    //!
    //! @brief Destructor.
    //!
    ~device_vector()
    {
        this->device_vector_teardown(m_data);
        m_data = nullptr;
    }

    //!
    //! @brief Returns the length of the vector.
    //!
    size_t n() const
    {
        return m_n;
    }

    //!
    //! @brief Returns the increment of the vector.
    //!
    rocblas_int inc() const
    {
        return m_inc;
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
        return m_data;
    }

    //!
    //! @brief Decay into constant pointer wherever pointer is expected.
    //!
    operator const T*() const
    {
        return m_data;
    }

    //!
    //! @brief Transfer data from a host vector.
    //! @param that The host vector.
    //! @return the hip error.
    //!
    hipError_t transfer_from(const host_vector<T>& that)
    {
        return hipMemcpy(m_data, (const T*)that, this->nmemb() * sizeof(T), hipMemcpyHostToDevice);
    }

    hipError_t memcheck() const
    {
        return m_data ? hipSuccess : hipErrorOutOfMemory;
    }

private:
    size_t      m_n{};
    rocblas_int m_inc{};
    T*          m_data{};
};
