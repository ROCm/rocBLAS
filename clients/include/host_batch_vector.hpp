//
// Copyright 2018-2019 Advanced Micro Devices, Inc.
//
#pragma once

#include "rocblas_init.hpp"
#include <string.h>

//!
//! @brief Implementation of the batch vector on host.
//!
template <typename T>
class host_batch_vector
{
public:
    //!
    //! @brief Delete copy constructor.
    //!
    host_batch_vector(const host_batch_vector<T>& that) = delete;

    //!
    //! @brief Delete copy assignement.
    //!
    host_batch_vector& operator=(const host_batch_vector<T>& that) = delete;

    //!
    //! @brief Constructor.
    //! @param n           The length of the vector.
    //! @param inc         The increment.
    //! @param stride      (UNUSED) The stride.
    //! @param batch_count The batch count.
    //!
    explicit host_batch_vector(rocblas_int    n,
                               rocblas_int    inc,
                               rocblas_stride stride,
                               rocblas_int    batch_count)
        : host_batch_vector(n, inc, batch_count)
    {
    }

    //!
    //! @brief Constructor.
    //! @param n           The length of the vector.
    //! @param inc         The increment.
    //! @param batch_count The batch count.
    //!
    explicit host_batch_vector(rocblas_int n, rocblas_int inc, rocblas_int batch_count)
        : m_n(n)
        , m_inc(inc)
        , m_batch_count(batch_count)
        , m_size(n * std::abs(inc))
    {
        if(false == this->try_initialize_memory())
        {
            this->free_memory();
        }
    }

    //!
    //! @brief Constructor.
    //! @param size_vector The size of one of the vectors.
    //! @param batch_count The number of vectors.
    //!
    explicit host_batch_vector(size_t size_vector, rocblas_int batch_count)
        : m_batch_count(batch_count)
        , m_size(size_vector)
    {
        if(false == this->try_initialize_memory())
        {
            this->free_memory();
        }
    }

    //!
    //! @brief Destructor.
    //!
    ~host_batch_vector()
    {
        this->free_memory();
    }

    //!
    //! @brief Returns the size of the vector.
    //!
    size_t size() const
    {
        return this->m_size;
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
        return 0;
    }

    //!
    //! @brief Random access to the vectors.
    //! @param batch_index the batch index.
    //! @return The mutable pointer.
    //!
    T* operator[](rocblas_int batch_index)
    {

        return this->m_data[batch_index];
    }

    //!
    //! @brief Constant random access to the vectors.
    //! @param batch_index the batch index.
    //! @return The non-mutable pointer.
    //!
    const T* operator[](rocblas_int batch_index) const
    {

        return this->m_data[batch_index];
    }

    //!
    //! @brief Cast to a double pointer.
    //!
    operator T**()
    {
        return this->m_data;
    }

    //!
    //! @brief Constant cast to a double pointer.
    //!
    operator const T* const*()
    {
        return this->m_data;
    }

    //!
    //! @brief Copy from a host batched vector.
    //! @param that the vector the data is copied from.
    //! @return true if the copy is done successfully, false otherwise.
    //!
    bool copy_from(const host_batch_vector<T>& that)
    {
        if((this->batch_count() == that.batch_count()) && (this->size() == that.size()))
        {
            size_t num_bytes = this->size() * sizeof(T);
            for(size_t batch_index = 0; batch_index < this->m_batch_count; ++batch_index)
            {
                memcpy((*this)[batch_index], that[batch_index], num_bytes);
            }
            return true;
        }
        else
        {
            return false;
        }
    }

    //!
    //! @brief Transfer from a device batched vector.
    //! @param that the vector the data is copied from.
    //! @return the hip error.
    //!
    hipError_t transfer_from(const device_batch_vector<T>& that)
    {
        auto num_bytes = this->size() * sizeof(T);
        for(size_t batch_index = 0; batch_index < this->m_batch_count; ++batch_index)
        {
            auto err = hipMemcpy(
                (*this)[batch_index], that[batch_index], num_bytes, hipMemcpyDeviceToHost);
            if(hipSuccess != err)
            {
                return err;
            }
        }
        return hipSuccess;
    }

    //!
    //! @brief Check if memory exists.
    //! @return hipSuccess if memory exists, hipErrorOutOfMemory otherwise.
    //!
    hipError_t memcheck() const
    {
        return (nullptr != this->m_data) ? hipSuccess : hipErrorOutOfMemory;
    }

private:
    rocblas_int m_n{};
    rocblas_int m_inc{};
    rocblas_int m_batch_count{};
    size_t      m_size{};
    T**         m_data{};

    bool try_initialize_memory()
    {
        bool success = (nullptr != (this->m_data = (T**)calloc(this->m_batch_count, sizeof(T*))));
        if(success)
        {
            for(rocblas_int batch_index = 0; batch_index < this->m_batch_count; ++batch_index)
            {
                success = (nullptr
                           != (this->m_data[batch_index] = (T*)calloc(this->m_size, sizeof(T))));
                if(false == success)
                {
                    break;
                }
            }
        }
        return success;
    }

    void free_memory()
    {
        if(nullptr != this->m_data)
        {
            for(rocblas_int batch_index = 0; batch_index < this->m_batch_count; ++batch_index)
            {
                if(nullptr != this->m_data[batch_index])
                {
                    free(this->m_data[batch_index]);
                    this->m_data[batch_index] = nullptr;
                }
            }

            free(this->m_data);
            this->m_data = nullptr;
        }
    }
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const host_batch_vector<T>& that)
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
}
