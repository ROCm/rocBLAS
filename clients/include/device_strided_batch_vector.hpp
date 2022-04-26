/* ************************************************************************
 * Copyright 2018-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

//
// Local declaration of the host strided batch vector.
//
template <typename T>
class host_strided_batch_vector;

//!
//! @brief Implementation of a strided batched vector on device.
//!
template <typename T>
class device_strided_batch_vector : public d_vector<T>
{
public:
    //!
    //! @brief Disallow copying.
    //!
    device_strided_batch_vector(const device_strided_batch_vector&) = delete;

    //!
    //! @brief Disallow assigning.
    //!
    device_strided_batch_vector& operator=(const device_strided_batch_vector&) = delete;

    //!
    //! @brief Constructor.
    //! @param n   The length of the vector.
    //! @param inc The increment.
    //! @param stride The stride.
    //! @param batch_count The batch count.
    //! @param HMM         HipManagedMemory Flag.
    //!
    explicit device_strided_batch_vector(
        size_t n, rocblas_int inc, rocblas_stride stride, rocblas_int batch_count, bool HMM = false)
        : d_vector<T>(calculate_nmemb(n, inc, stride, batch_count), HMM)
        , m_n(n)
        , m_inc(inc)
        , m_stride(stride)
        , m_batch_count(batch_count)
    {
        this->m_data = this->device_vector_setup();
    }

    //!
    //! @brief Destructor.
    //!
    ~device_strided_batch_vector()
    {
        if(nullptr != this->m_data)
        {
            this->device_vector_teardown(this->m_data);
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
    //! @brief Returns the stride value.
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
                   ? this->m_data + batch_index * this->m_stride
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
                   ? this->m_data + batch_index * this->m_stride
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
    //! @brief Transfer data from a strided batched vector on device.
    //! @param that That strided batched vector on device.
    //! @return The hip error.
    //!
    hipError_t transfer_from(const host_strided_batch_vector<T>& that)
    {
        return hipMemcpy(this->data(),
                         that.data(),
                         sizeof(T) * this->nmemb(),
                         this->use_HMM ? hipMemcpyHostToHost : hipMemcpyHostToDevice);
    }

    //!
    //! @brief Check if memory exists.
    //! @return hipSuccess if memory exists, hipErrorOutOfMemory otherwise.
    //!
    hipError_t memcheck() const
    {
        if(*this)
            return hipSuccess;
        else
            return hipErrorOutOfMemory;
    }

private:
    size_t         m_n{};
    rocblas_int    m_inc{};
    rocblas_stride m_stride{};
    rocblas_int    m_batch_count{};
    T*             m_data{};

    static size_t
        calculate_nmemb(size_t n, rocblas_int inc, rocblas_stride stride, rocblas_int batch_count)
    {
        return std::abs(inc) * n + size_t(batch_count - 1) * std::abs(stride);
    }
};
