/* ************************************************************************
 * Copyright 2018-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "d_vector.hpp"

//
// Local declaration of the host batch matrix.
//
template <typename T>
class host_batch_matrix;

//!
//! @brief  pseudo-matrix subclass which uses a batch of device memory pointers and
//!  - an array of pointers in host memory
//!  - an array of pointers in device memory
//!
template <typename T>
class device_batch_matrix : public d_vector<T>
{
public:
    //!
    //! @brief Disallow copying.
    //!
    device_batch_matrix(const device_batch_matrix&) = delete;

    //!
    //! @brief Disallow assigning.
    //!
    device_batch_matrix& operator=(const device_batch_matrix&) = delete;

    //!
    //! @brief Constructor.
    //! @param m           The number of rows of the Matrix.
    //! @param n           The number of cols of the Matrix.
    //! @param lda         The leading dimension of the Matrix.
    //! @param batch_count The batch count.
    //! @param HMM         HipManagedMemory Flag.
    //!
    explicit device_batch_matrix(
        size_t m, size_t n, size_t lda, rocblas_int batch_count, bool HMM = false)
        : m_m(m)
        , m_n(n)
        , m_lda(lda)
        , m_batch_count(batch_count)
        , d_vector<T>(size_t(n) * lda, HMM)
    {
        if(false == this->try_initialize_memory())
        {
            this->free_memory();
        }
    }

    //!
    //! @brief Destructor.
    //!
    ~device_batch_matrix()
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
    //! @brief Returns the value of batch_count.
    //!
    rocblas_int batch_count() const
    {
        return this->m_batch_count;
    }

    //!
    //! @brief Access to device data.
    //! @return Pointer to the device data.
    //!
    T** ptr_on_device()
    {
        return this->m_device_data;
    }

    //!
    //! @brief Const access to device data.
    //! @return Const pointer to the device data.
    //!
    const T* const* ptr_on_device() const
    {
        return this->m_device_data;
    }

    //!
    //! @brief access to device data.
    //! @return Const pointer to the device data.
    //!
    T* const* const_batch_ptr()
    {
        return this->m_device_data;
    }

    //!
    //! @brief Random access.
    //! @param batch_index The batch index.
    //! @return Pointer to the array on device.
    //!
    T* operator[](rocblas_int batch_index)
    {

        return this->m_data[batch_index];
    }

    //!
    //! @brief Constant random access.
    //! @param batch_index The batch index.
    //! @return Constant pointer to the array on device.
    //!
    const T* operator[](rocblas_int batch_index) const
    {

        return this->m_data[batch_index];
    }

    //!
    //! @brief Const cast of the data on host.
    //!
    operator const T* const *() const
    {
        return this->m_data;
    }

    //!
    //! @brief Cast of the data on host.
    //!
    // clang-format off
    operator T**()
    // clang-format on
    {
        return this->m_data;
    }

    //!
    //! @brief Tell whether ressources allocation failed.
    //!
    explicit operator bool() const
    {
        return nullptr != this->m_data;
    }

    //!
    //! @brief Copy from a host batched matrix.
    //! @param that The host_batch_matrix to copy.
    //!
    hipError_t transfer_from(const host_batch_matrix<T>& that)
    {
        hipError_t hip_err;
        //
        // Copy each matrix.
        //
        hipMemcpyKind kind = this->use_HMM ? hipMemcpyHostToHost : hipMemcpyHostToDevice;
        for(rocblas_int batch_index = 0; batch_index < this->m_batch_count; ++batch_index)
        {
            if(hipSuccess
               != (hip_err = hipMemcpy(
                       (*this)[batch_index], that[batch_index], sizeof(T) * this->nmemb(), kind)))
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
        if(*this)
            return hipSuccess;
        else
            return hipErrorOutOfMemory;
    }

private:
    size_t      m_m{};
    size_t      m_n{};
    size_t      m_lda{};
    rocblas_int m_batch_count{};
    T**         m_data{};
    T**         m_device_data{};

    //!
    //! @brief Try to allocate the ressources.
    //! @return true if success false otherwise.
    //!
    bool try_initialize_memory()
    {
        bool success = false;

        success
            = (hipSuccess
               == (!this->use_HMM
                       ? (hipMalloc)(&this->m_device_data, this->m_batch_count * sizeof(T*))
                       : hipMallocManaged(&this->m_device_data, this->m_batch_count * sizeof(T*))));
        if(success)
        {
            success
                = (nullptr
                   != (this->m_data = !this->use_HMM ? (T**)calloc(this->m_batch_count, sizeof(T*))
                                                     : m_device_data));
            if(success)
            {
                for(rocblas_int batch_index = 0; batch_index < this->m_batch_count; ++batch_index)
                {
                    success
                        = (nullptr != (this->m_data[batch_index] = this->device_vector_setup()));
                    if(!success)
                    {
                        break;
                    }
                }

                if(success && !this->use_HMM)
                {
                    success = (hipSuccess
                               == hipMemcpy(this->m_device_data,
                                            this->m_data,
                                            sizeof(T*) * this->m_batch_count,
                                            hipMemcpyHostToDevice));
                }
            }
        }
        return success;
    }

    //!
    //! @brief Free the ressources, as much as we can.
    //!
    void free_memory()
    {
        if(nullptr != this->m_data)
        {
            for(rocblas_int batch_index = 0; batch_index < this->m_batch_count; ++batch_index)
            {
                if(nullptr != this->m_data[batch_index])
                {
                    this->device_vector_teardown(this->m_data[batch_index]);
                    this->m_data[batch_index] = nullptr;
                }
            }

            if(!this->use_HMM)
            {
                free(this->m_data);
            }
            // else this is just a copy of m_device_data

            this->m_data = nullptr;
        }

        if(nullptr != this->m_device_data)
        {
            auto tmp_device_data = this->m_device_data;
            this->m_device_data  = nullptr;
            CHECK_HIP_ERROR((hipFree)(tmp_device_data));
        }
    }
};
