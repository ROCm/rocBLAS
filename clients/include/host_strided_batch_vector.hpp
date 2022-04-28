/* ************************************************************************
 * Copyright (C) 2018-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include "host_alloc.hpp"

//
// Local declaration of the device strided batch vector.
//
template <typename T>
class device_strided_batch_vector;

//!
//! @brief Implementation of a host strided batched vector.
//!
template <typename T>
class host_strided_batch_vector
{
public:
    //!
    //! @brief Disallow copying.
    //!
    host_strided_batch_vector(const host_strided_batch_vector&) = delete;

    //!
    //! @brief Disallow assigning.
    //!
    host_strided_batch_vector& operator=(const host_strided_batch_vector&) = delete;

    //!
    //! @brief Constructor.
    //! @param n   The length of the vector.
    //! @param inc The increment.
    //! @param stride The stride.
    //! @param batch_count The batch count.
    //!
    explicit host_strided_batch_vector(size_t         n,
                                       rocblas_int    inc,
                                       rocblas_stride stride,
                                       rocblas_int    batch_count)
        : m_n(n)
        , m_inc(inc)
        , m_stride(stride)
        , m_batch_count(batch_count)
        , m_nmemb(calculate_nmemb(n, inc, stride, batch_count))
    {
        this->m_data = (T*)host_malloc_throw(this->m_nmemb, sizeof(T));
    }

    //!
    //! @brief Destructor.
    //!
    ~host_strided_batch_vector()
    {
        if(nullptr != this->m_data)
        {
            free(this->m_data);
            this->m_data = nullptr;
        }
    }

    //!
    //! @brief Returns the data pointer.
    //!
    T* data()
    {
        return this->m_data;
    }

    //!
    //! @brief Returns the data pointer.
    //!
    const T* data() const
    {
        return this->m_data;
    }

    //!
    //! @brief Returns the length.
    //!
    size_t n() const
    {
        return this->m_n;
    }

    //!
    //! @brief Returns the increment.
    //!
    rocblas_int inc() const
    {
        return this->m_inc;
    }

    //!
    //! @brief Returns the batch count.
    //!
    rocblas_int batch_count() const
    {
        return this->m_batch_count;
    }

    //!
    //! @brief Returns the stride.
    //!
    rocblas_stride stride() const
    {
        return this->m_stride;
    }

    //!
    //! @brief Returns pointer.
    //! @param batch_index The batch index.
    //! @return A mutable pointer to the batch_index'th vector.
    //!
    T* operator[](rocblas_int batch_index)
    {

        return (this->m_stride >= 0)
                   ? this->m_data + this->m_stride * batch_index
                   : this->m_data + (batch_index + 1 - this->m_batch_count) * this->m_stride;
    }

    //!
    //! @brief Returns non-mutable pointer.
    //! @param batch_index The batch index.
    //! @return A non-mutable mutable pointer to the batch_index'th vector.
    //!
    const T* operator[](rocblas_int batch_index) const
    {

        return (this->m_stride >= 0)
                   ? this->m_data + this->m_stride * batch_index
                   : this->m_data + (batch_index + 1 - this->m_batch_count) * this->m_stride;
    }

    //!
    //! @brief Cast operator.
    //! @remark Returns the pointer of the first vector.
    //!
    operator T*()
    {
        return (*this)[0];
    }

    //!
    //! @brief Non-mutable cast operator.
    //! @remark Returns the non-mutable pointer of the first vector.
    //!
    operator const T*() const
    {
        return (*this)[0];
    }

    //!
    //! @brief Tell whether ressources allocation failed.
    //!
    explicit operator bool() const
    {
        return nullptr != this->m_data;
    }

    //!
    //! @brief Copy data from a strided batched vector on host.
    //! @param that That strided batched vector on host.
    //! @return true if successful, false otherwise.
    //!
    bool copy_from(const host_strided_batch_vector& that)
    {
        if(that.n() == this->m_n && that.inc() == this->m_inc && that.stride() == this->m_stride
           && that.batch_count() == this->m_batch_count)
        {
            memcpy(this->data(), that.data(), sizeof(T) * this->m_nmemb);
            return true;
        }
        else
        {
            return false;
        }
    }

    //!
    //! @brief Transfer data from a strided batched vector on device.
    //! @param that That strided batched vector on device.
    //! @return The hip error.
    //!
    hipError_t transfer_from(const device_strided_batch_vector<T>& that)
    {
        hipError_t hip_err;

        if(that.use_HMM && hipSuccess != (hip_err = hipDeviceSynchronize()))
            return hip_err;

        return hipMemcpy(this->m_data,
                         that.data(),
                         sizeof(T) * this->m_nmemb,
                         that.use_HMM ? hipMemcpyHostToHost : hipMemcpyDeviceToHost);
    }

    //!
    //! @brief Check if memory exists.
    //! @return hipSuccess if memory exists, hipErrorOutOfMemory otherwise.
    //!
    hipError_t memcheck() const
    {
        return ((bool)*this) ? hipSuccess : hipErrorOutOfMemory;
    }

private:
    size_t         m_n{};
    rocblas_int    m_inc{};
    rocblas_stride m_stride{};
    rocblas_int    m_batch_count{};
    size_t         m_nmemb{};
    T*             m_data{};

    static size_t
        calculate_nmemb(size_t n, rocblas_int inc, rocblas_stride stride, rocblas_int batch_count)
    {
        return std::abs(inc) * n + size_t(batch_count - 1) * std::abs(stride);
    }
};

//!
//! @brief Overload output operator.
//! @param os The ostream.
//! @param that That host strided batch vector.
//!
template <typename T>
rocblas_internal_ostream& operator<<(rocblas_internal_ostream&           os,
                                     const host_strided_batch_vector<T>& that)
{
    auto n           = that.n();
    auto inc         = std::abs(that.inc());
    auto batch_count = that.batch_count();

    for(rocblas_int batch_index = 0; batch_index < batch_count; ++batch_index)
    {
        auto batch_data = that[batch_index];
        os << "[" << batch_index << "] = { " << batch_data[0];
        for(size_t i = 1; i < n; ++i)
        {
            os << ", " << batch_data[i * inc];
        }
        os << " }" << std::endl;
    }

    return os;
}
