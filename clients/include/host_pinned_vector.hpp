//
// Copyright 2018-2019 Advanced Micro Devices, Inc.
//
#pragma once
#include "pinned_memory_allocator.hpp"
#include <memory>

//!
//! @brief  Pseudo-vector subclass which uses host pinned memory.
//!
template <typename T>
struct host_pinned_vector : std::vector<T, pinned_memory_allocator<T>>
{
    // Inherit constructors
    using std::vector<T, pinned_memory_allocator<T>>::vector;

    //!
    //! @brief Constructor.
    //!

    host_pinned_vector(rocblas_int n, rocblas_int inc)
        : std::vector<T, pinned_memory_allocator<T>>(size_t(n) * inc, pinned_memory_allocator<T>())
        , m_n(n)
        , m_inc(inc)
    {
    }

    //!
    //! @brief Decay into pointer wherever pointer is expected
    //!
    operator T*()
    {
        return this->data();
    }

    //!
    //! @brief Decay into constant pointer wherever constant pointer is expected
    //!
    operator const T*() const
    {
        return this->data();
    }

    //!
    //! @brief Transfer from a device vector.
    //! @param  that That device vector.
    //! @return the hip error.
    //!
    hipError_t transfer_from(const device_vector<T>& that)
    {
        return hipMemcpy(
            this->data(), (const T*)that, sizeof(T) * this->size(), hipMemcpyDeviceToHost);
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
    //! @brief Check if memory exists.
    //! @return hipSuccess if memory exists, hipErrorOutOfMemory otherwise.
    //!
    hipError_t memcheck() const
    {
        return (nullptr != (const T*)this) ? hipSuccess : hipErrorOutOfMemory;
    }

private:
    rocblas_int m_n{};
    rocblas_int m_inc{};
};
