//
// Copyright 2018-2020 Advanced Micro Devices, Inc.
//
#pragma once

#include <cmath>
#include <type_traits>
#include <vector>

//!
//! @brief  Pseudo-vector subclass which uses host memory.
//!
template <typename T>
struct host_vector : std::vector<T>
{
    // Inherit constructors
    using std::vector<T>::vector;

    //!
    //! @brief Constructor.
    //!
    host_vector(size_t n, ptrdiff_t inc)
        : std::vector<T>(n * std::abs(inc))
        , m_n(n)
        , m_inc(inc)
    {
    }

    //!
    //! @brief Copy constructor from host_vector of other types convertible to T
    //!
    template <typename U, std::enable_if_t<std::is_convertible<U, T>{}, int> = 0>
    host_vector(const host_vector<U>& x)
        : std::vector<T>(x.size())
        , m_n(x.size())
        , m_inc(1)
    {
        for(size_t i = 0; i < m_n; ++i)
            (*this)[i] = x[i];
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
        return hipMemcpy(*this, that, sizeof(T) * this->size(), hipMemcpyDeviceToHost);
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
    ptrdiff_t inc() const
    {
        return m_inc;
    }

    //!
    //! @brief Returns the batch count (always 1).
    //!
    static constexpr rocblas_int batch_count()
    {
        return 1;
    }

    //!
    //! @brief Returns the stride (out of context, always 0)
    //!
    static constexpr rocblas_stride stride()
    {
        return 0;
    }

    //!
    //! @brief Check if memory exists (out of context, always hipSuccess)
    //!
    static constexpr hipError_t memcheck()
    {
        return hipSuccess;
    }

private:
    size_t    m_n;
    ptrdiff_t m_inc;
};
