//
// Copyright 2018-2019 Advanced Micro Devices, Inc.
//
#pragma once

template <typename T>
class host_strided_batch_vector;

//!
//! @brief Implementation of a strided batched vector on device.
//!
template <typename T, size_t PAD = 4096, typename U = T>
class device_strided_batch_vector : public d_vector<T, PAD, U>
{
public:
    //!
    //! @brief The storage type to use.
    //!
    typedef enum class estorage
    {
        block,
        interleave
    } storage;

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
    //! @param stg The storage format to use.
    //!
    explicit device_strided_batch_vector(rocblas_int    n,
                                         rocblas_int    inc,
                                         rocblas_stride stride,
                                         rocblas_int    batch_count,
                                         storage        stg = storage::block)
        : d_vector<T, PAD, U>(calculate_size(n, inc, stride, batch_count, stg))
        , m_storage(stg)
        , m_n(n)
        , m_inc(inc)
        , m_stride(stride)
        , m_batch_count(batch_count)
    {
        bool valid_parameters = true;

        switch(this->m_storage)
        {
        case storage::block:
        {
            if(this->m_stride < this->m_n * this->m_inc)
            {
                valid_parameters = false;
            }
            break;
        }
        case storage::interleave:
        {
            if(this->m_inc < this->m_stride * this->m_batch_count)
            {
                valid_parameters = false;
            }
            break;
        }
        }

        if(valid_parameters)
        {
            this->m_data = this->device_vector_setup();
        }
    };

    //!
    //! @brief Destructor.
    //!
    inline ~device_strided_batch_vector() noexcept
    {
        if(nullptr != this->m_data)
        {
            this->device_vector_teardown(this->m_data);
            this->m_data = nullptr;
        }
    }

    //!
    //! @brief Returns the length.
    //!
    inline rocblas_int n() const noexcept
    {
        return this->m_n;
    };

    //!
    //! @brief Returns the increment.
    //!
    inline rocblas_int inc() const noexcept
    {
        return this->m_inc;
    };

    //!
    //! @brief Returns the batch count.
    //!
    inline rocblas_int batch_count() const noexcept
    {
        return this->m_batch_count;
    };

    //!
    //! @brief Returns the stride value.
    //!
    inline rocblas_stride stride() const noexcept
    {
        return this->m_stride;
    };

    //!
    //! @brief Returns pointer.
    //! @param batch_index The batch index.
    //! @return A mutable pointer to the batch_index'th vector.
    //!
    inline T* operator[](rocblas_int batch_index) noexcept
    {

        assert(0 <= batch_index && this->m_batch_count > batch_index);

        return (this->m_stride >= 0)
                   ? this->m_data + batch_index * this->m_stride
                   : this->m_data + (batch_index + 1 - this->m_batch_count) * this->m_stride;
    };

    //!
    //! @brief Returns non-mutable pointer.
    //! @param batch_index The batch index.
    //! @return A non-mutable mutable pointer to the batch_index'th vector.
    //!
    inline const T* operator[](rocblas_int batch_index) const noexcept
    {

        assert(0 <= batch_index && this->m_batch_count > batch_index);

        return (this->m_stride >= 0)
                   ? this->m_data + batch_index * this->m_stride
                   : this->m_data + (batch_index + 1 - this->m_batch_count) * this->m_stride;
    };

    //!
    //! @brief Cast operator.
    //! @remark Returns the pointer of the first vector.
    //!
    inline operator T*() noexcept
    {
        return (*this)[0];
    };

    //!
    //! @brief Non-mutable cast operator.
    //! @remark Returns the non-mutable pointer of the first vector.
    //!
    inline operator const T*() const noexcept
    {
        return (*this)[0];
    };

    //!
    //! @brief Tell whether ressources allocation failed.
    //!
    inline explicit operator bool() const noexcept
    {
        return nullptr != this->m_data;
    };

    //!
    //! @brief Transfer data from a strided batched vector on device.
    //! @param that That strided batched vector on device.
    //! @return The hip error.
    //!
    inline hipError_t transfer_from(const host_strided_batch_vector<T>& that) noexcept
    {
        if(that.n() == this->m_n && that.inc() == this->m_inc && that.stride() == this->m_stride
           && that.batch_count() == this->m_batch_count)
        {
            auto hip_err
                = hipMemcpy((*this)[0], that[0], sizeof(T) * this->nmemb(), hipMemcpyHostToDevice);
            if(hipSuccess != hip_err)
            {
                return hip_err;
            }

            return hipSuccess;
        }
        else
        {
            return hipErrorInvalidContext;
        }
    };

private:
    storage        m_storage{storage::block};
    rocblas_int    m_n{0};
    rocblas_int    m_inc{0};
    rocblas_stride m_stride{0};
    rocblas_int    m_batch_count{0};
    T*             m_data{nullptr};

    static inline size_t calculate_size(
        rocblas_int n, rocblas_int inc, rocblas_stride stride, rocblas_int batch_count, storage st)
    {
        switch(st)
        {
        case storage::block:
        {
            return stride * batch_count;
        }
        case storage::interleave:
        {
            return n * inc;
        }
        default:
        {
            return 0;
        }
        }
    };
};
