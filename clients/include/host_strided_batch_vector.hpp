//
// Copyright 2018-2019 Advanced Micro Devices, Inc.
//
#pragma once

template <typename T, size_t PAD, typename U>
class device_strided_batch_vector;

//!
//! @brief Implementation of a host strided batched vector.
//!
template <typename T>
class host_strided_batch_vector
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
    //! @param stg The storage format to use.
    //!
    explicit host_strided_batch_vector(rocblas_int    n,
                                       rocblas_int    inc,
                                       rocblas_stride stride,
                                       rocblas_int    batch_count,
                                       storage        stg = storage::block) noexcept
        : m_storage(stg)
        , m_n(n)
        , m_inc(inc)
        , m_stride(stride)
        , m_batch_count(batch_count)
        ,

        m_size(calculate_size(n, inc, stride, batch_count, stg))
    {

        bool valid_parameters = (m_size > 0);
        if(valid_parameters)
        {
            switch(this->m_storage)
            {
            case storage::block:
            {
                if(this->m_stride < this->m_n * std::abs(this->m_inc))
                {
                    valid_parameters = false;
                }
                break;
            }
            case storage::interleave:
            {
                if(std::abs(this->m_inc) < this->m_stride * this->m_batch_count)
                {
                    valid_parameters = false;
                }
                break;
            }
            }

            if(valid_parameters)
            {
                this->m_data = new T[m_size];
            }
        }
    };

    //!
    //! @brief Destructor.
    //!
    inline ~host_strided_batch_vector() noexcept
    {
        if(nullptr != this->m_data)
        {
            delete[] this->m_data;
            this->m_data = nullptr;
        }
    };

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
    //! @brief Returns the stride.
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
                   ? this->m_data + this->m_stride * batch_index
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
                   ? this->m_data + this->m_stride * batch_index
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
    //! @brief Copy data from a strided batched vector on host.
    //! @param that That strided batched vector on host.
    //! @return true if successful, false otherwise.
    //!
    inline bool copy_from(const host_strided_batch_vector& that) noexcept
    {
        if(that.n() == this->m_n && that.inc() == this->m_inc && that.stride() == this->m_stride
           && that.batch_count() == this->m_batch_count)
        {
            memcpy((*this)[0], that[0], sizeof(T) * this->m_size);
            return true;
        }
        else
        {
            return false;
        }
    };

    //!
    //! @brief Transfer data from a strided batched vector on device.
    //! @param that That strided batched vector on device.
    //! @return The hip error.
    //!
    template <size_t PAD, typename U>
    inline hipError_t transfer_from(const device_strided_batch_vector<T, PAD, U>& that) noexcept
    {
        if(that.n() == this->m_n && that.inc() == this->m_inc && that.stride() == this->m_stride
           && that.batch_count() == this->m_batch_count)
        {
            auto hip_err
                = hipMemcpy((*this)[0], that[0], sizeof(T) * this->m_size, hipMemcpyDeviceToHost);
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
    size_t         m_size{0};
    T*             m_data{nullptr};

    static inline size_t calculate_size(
        rocblas_int n, rocblas_int inc, rocblas_stride stride, rocblas_int batch_count, storage st)
    {
        switch(st)
        {
        case storage::block:
        {
            return size_t(stride) * batch_count;
        }
        case storage::interleave:
        {
            return size_t(n) * std::abs(inc);
        }
        default:
        {
            return 0;
        }
        }
    };
};

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const host_strided_batch_vector<T>& that)
{
    auto batch_count = that.batch_count();
    auto n           = that.n();
    auto inc         = std::abs(that.inc());

    for(rocblas_int batch_index = 0; batch_index < batch_count; ++batch_index)
    {
        auto v = that[batch_index];
        os << "[" << batch_index << "] = { " << v[0];
        for(rocblas_int i = 1; i < n; ++i)
        {
            os << ", " << v[i * inc];
        }
        os << " }" << std::endl;
    }

    return os;
};
