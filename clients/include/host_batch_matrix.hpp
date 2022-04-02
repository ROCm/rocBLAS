/* ************************************************************************
 * Copyright 2018-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include <string.h>

#include "host_alloc.hpp"
#include "rocblas_init.hpp"

//
// Local declaration of the device batch Matrix.
//
template <typename T>
class device_batch_matrix;

//!
//! @brief Implementation of the batch Matrix on host.
//!
template <typename T>
class host_batch_matrix
{
public:
    //!
    //! @brief Delete copy constructor.
    //!
    host_batch_matrix(const host_batch_matrix<T>& that) = delete;

    //!
    //! @brief Delete copy assignement.
    //!
    host_batch_matrix& operator=(const host_batch_matrix<T>& that) = delete;

    //!
    //! @brief Constructor.
    //! @param m           The number of rows of the Matrix.
    //! @param n           The number of cols of the Matrix.
    //! @param lda         The leading dimension of the Matrix.
    //! @param batch_count The batch count.
    //!
    explicit host_batch_matrix(size_t m, size_t n, size_t lda, rocblas_int batch_count)
        : m_m(m)
        , m_n(n)
        , m_lda(lda)
        , m_batch_count(batch_count)
    {
        if(false == this->try_initialize_memory())
        {
            this->free_memory();
        }
    }

    //!
    //! @brief Destructor.
    //!
    ~host_batch_matrix()
    {
        this->free_memory();
    }

    //!
    //! @brief Returns the rows of the Matrix.
    //!
    size_t m() const
    {
        return this->m_m;
    }

    //!
    //! @brief Returns the cols of the Matrix.
    //!
    size_t n() const
    {
        return this->m_n;
    }

    //!
    //! @brief Returns the leading dimension of the Matrix.
    //!
    size_t lda() const
    {
        return this->m_lda;
    }

    //!
    //! @brief Returns the batch count.
    //!
    rocblas_int batch_count() const
    {
        return this->m_batch_count;
    }

    //!
    //! @brief Random access to the Matrices.
    //! @param batch_index the batch index.
    //! @return The mutable pointer.
    //!
    T* operator[](rocblas_int batch_index)
    {

        return this->m_data[batch_index];
    }

    //!
    //! @brief Constant random access to the Matrices.
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
    // clang-format off
    operator T**()
    // clang-format on
    {
        return this->m_data;
    }

    //!
    //! @brief Constant cast to a double pointer.
    //!
    operator const T* const *()
    {
        return this->m_data;
    }

    //!
    //! @brief Copy from a host batched Matrix.
    //! @param that the Matrix the data is copied from.
    //! @return true if the copy is done successfully, false otherwise.
    //!
    bool copy_from(const host_batch_matrix<T>& that)
    {
        if((this->batch_count() == that.batch_count()) && (this->m() == that.m())
           && (this->n() == that.n()) && (this->lda() == that.lda()))
        {
            size_t num_bytes = this->lda() * this->n() * sizeof(T);
            for(rocblas_int batch_index = 0; batch_index < this->m_batch_count; ++batch_index)
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
    //! @brief Transfer from a device batched Matrix.
    //! @param that the Matrix the data is copied from.
    //! @return the hip error.
    //!
    hipError_t transfer_from(const device_batch_matrix<T>& that)
    {
        hipError_t hip_err;
        size_t     num_bytes = this->lda() * this->n() * sizeof(T);
        if(that.use_HMM && hipSuccess != (hip_err = hipDeviceSynchronize()))
            return hip_err;

        hipMemcpyKind kind = that.use_HMM ? hipMemcpyHostToHost : hipMemcpyDeviceToHost;

        for(rocblas_int batch_index = 0; batch_index < this->m_batch_count; ++batch_index)
        {
            if(hipSuccess
               != (hip_err = hipMemcpy((*this)[batch_index], that[batch_index], num_bytes, kind)))
            {
                return hip_err;
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
    size_t      m_m{};
    size_t      m_n{};
    size_t      m_lda{};
    rocblas_int m_batch_count{};
    T**         m_data{};

    bool try_initialize_memory()
    {
        bool success
            = (nullptr != (this->m_data = (T**)host_calloc_throw(this->m_batch_count, sizeof(T*))));
        if(success)
        {
            size_t nmemb = size_t(this->m_n) * size_t(this->m_lda);
            for(rocblas_int batch_index = 0; batch_index < this->m_batch_count; ++batch_index)
            {
                success
                    = (nullptr
                       != (this->m_data[batch_index] = (T*)host_calloc_throw(nmemb, sizeof(T))));
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

//!
//! @brief Overload output operator.
//! @param os The ostream.
//! @param that That host batch Matrix.
//!
template <typename T>
rocblas_internal_ostream& operator<<(rocblas_internal_ostream& os, const host_batch_matrix<T>& that)
{
    auto m           = that.m();
    auto n           = that.n();
    auto lda         = that.lda();
    auto batch_count = that.batch_count();

    for(rocblas_int batch_index = 0; batch_index < batch_count; ++batch_index)
    {
        auto batch_data = that[batch_index];
        os << "[" << batch_index << "]" << std::endl;
        for(size_t i = 0; i < n; ++i)
        {
            for(size_t j = 0; j < m; ++j)
                os << ", " << batch_data[j + i * lda];
        }
        os << std::endl;
    }

    return os;
}
