//
// Copyright 2018-2019 Advanced Micro Devices, Inc.
//
#pragma once

#include "d_vector.hpp"

template <typename T>
class host_batch_vector;

//!
//! @brief  pseudo-vector subclass which uses a batch of device memory pointers and
//!  - an array of pointers in host memory
//!  - an array of pointers in device memory
//!
template <typename T, size_t PAD = 4096, typename U = T>
class device_batch_vector : private d_vector<T, PAD, U>
{
public:
    //!
    //! @brief Disallow copying.
    //!
    device_batch_vector(const device_batch_vector&) = delete;

    //!
    //! @brief Disallow or assigning.
    //!
    device_batch_vector& operator=(const device_batch_vector&) = delete;

    //!
    //! @brief Constructor.
    //! @param batch_count The number of vectors.
    //! @param size_vector The size of each vectors.
    //!
    explicit device_batch_vector(rocblas_int batch_count, size_t size_vector)
        : m_batch_count(batch_count)
        , m_size_vector(size_vector)
        , d_vector<T, PAD, U>(size_vector)
    {
        if(false == this->try_initialize_memory())
        {
            this->free_memory();
        }
    };

    //!
    //! @brief Destructor.
    //!
    inline ~device_batch_vector() noexcept
    {
        this->free_memory();
    }

    //!
    //! @brief Returns the size of the vectors.
    //!
    inline size_t size() const
    {
        return this->m_size_vector;
    };

    //!
    //! @brief Returns the value of batch_count.
    //!
    inline rocblas_int batch_count() const noexcept
    {
        return this->m_batch_count;
    };

    //!
    //! @brief Access to device data.
    //! @return Pointer to the device data.
    //!
    inline T** ptr_on_device() noexcept
    {
        return this->m_device_data;
    }

    //!
    //! @brief Const access to device data.
    //! @return Const pointer to the device data.
    //!
    inline const T* const* ptr_on_device() const noexcept
    {
        return this->m_device_data;
    }

    //!
    //! @brief Random access.
    //! @param batch_index The batch index.
    //! @return Pointer to the array on device.
    //!
    inline T* operator[](rocblas_int batch_index) noexcept
    {
        assert(0 <= batch_index && this->m_batch_count > batch_index);

        return this->m_data[batch_index];
    }

    //!
    //! @brief Constant random access.
    //! @param batch_index The batch index.
    //! @return Constant pointer to the array on device.
    //!
    inline const T* operator[](rocblas_int batch_index) const noexcept
    {
        assert(0 <= batch_index && this->m_batch_count > batch_index);

        return this->m_data[batch_index];
    }

    //!
    //! @brief Const cast of the data on host.
    //!
    inline operator const T* const*() const noexcept
    {
        return this->m_data;
    };

    //!
    //! @brief Cast of the data on host.
    //!
    inline operator T**() noexcept
    {
        return this->m_data;
    }

    //!
    //! @brief Tell whether ressources allocation failed.
    //!
    inline explicit operator bool() const noexcept
    {
        return this->m_data != nullptr;
    }

    //!
    //! @brief Copy from a host batched vector.
    //! @param that The host_batch_vector to copy.
    //!
    inline hipError_t transfer_from(const host_batch_vector<T>& that) noexcept
    {
        //
        // Check sizes.
        //
        if((this->batch_count() != that.batch_count()) || (this->size() != that.size()))
        {
            return hipErrorInvalidContext;
        }

        //
        // Copy each vector.
        //
        for(rocblas_int batch_index = 0; batch_index < this->m_batch_count; ++batch_index)
        {
            auto hip_err = hipMemcpy((*this)[batch_index],
                                     that[batch_index],
                                     sizeof(T) * this->m_size_vector,
                                     hipMemcpyHostToDevice);
            if(hipSuccess != hip_err)
            {
                return hip_err;
            }
        }

        return hipSuccess;
    }

private:
    rocblas_int m_batch_count{0};
    size_t      m_size_vector{0};
    T**         m_data{nullptr};
    T**         m_device_data{nullptr};

    //!
    //! @brief Try to allocate the ressources.
    //! @return true if success false otherwise.
    //!
    bool try_initialize_memory()
    {
        bool success = false;
        success
            = (hipSuccess == (hipMalloc)(&this->m_device_data, this->m_batch_count * sizeof(T*)));
        if(success)
        {
            success = (nullptr != (this->m_data = (T**)calloc(this->m_batch_count, sizeof(T*))));
            if(success)
            {
                for(size_t batch_index = 0; batch_index < this->m_batch_count; ++batch_index)
                {
                    success
                        = (nullptr != (this->m_data[batch_index] = this->device_vector_setup()));
                    if(!success)
                    {
                        break;
                    }
                }

                if(success)
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

    //! @brief Free the ressources, as much as we can.
    void free_memory()
    {
        if(nullptr != this->m_data)
        {
            for(size_t batch_index = 0; batch_index < this->m_batch_count; ++batch_index)
            {
                if(nullptr != this->m_data[batch_index])
                {
                    this->device_vector_teardown(this->m_data[batch_index]);
                    this->m_data[batch_index] = nullptr;
                }
            }

            free(this->m_data);
            this->m_data = nullptr;
        }

        if(nullptr != this->m_device_data)
        {
            auto tmp_device_data = this->m_device_data;
            this->m_device_data  = nullptr;
            CHECK_HIP_ERROR((hipFree)(tmp_device_data));
        }
    };
};
