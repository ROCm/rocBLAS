//
// Copyright 2018-2019 Advanced Micro Devices, Inc.
//
#pragma once

#include "d_vector.hpp"

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
    //! @param s the size.
    //! @remark Must wrap constructor and destructor in functions to allow Google Test macros to work
    //!
    explicit device_vector(size_t s)
        : m_size(s)
        , d_vector<T, PAD, U>(s)
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
    //! @brief Returns the size of the vector.
    //!
    size_t size() const
    {
        return this->m_size;
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
            this->m_data, (const T*)that, this->size() * sizeof(T), hipMemcpyHostToDevice);
    }

    hipError_t memcheck() const
    {
        return ((bool)*this) ? hipSuccess : hipErrorOutOfMemory;
    }

private:
    size_t m_size{0};
    T*     m_data{nullptr};
};
