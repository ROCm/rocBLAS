//
// Copyright 2018-2019 Advanced Micro Devices, Inc.
//
#pragma once

//!
//! @brief  Pseudo-vector subclass which uses host memory.
//!
template <typename T>
struct host_vector : std::vector<T>
{
    // Inherit constructors
    using std::vector<T>::vector;

    //!
    //! @brief Decay into pointer wherever pointer is expected
    //!
    inline operator T*() noexcept
    {
        return this->data();
    }

    //!
    //! @brief Decay into constant pointer wherever constant pointer is expected
    //!
    inline operator const T*() const noexcept
    {
        return this->data();
    }

    //!
    //! @brief Transfer from a device vector.
    //! @param  that That device vector.
    //! @return the hip error.
    //!
    hipError_t transfer_from(const device_vector<T>& that) noexcept
    {
        if(that.size() == this->size())
        {
            return hipMemcpy(
                this->data(), (const T*)that, sizeof(T) * this->size(), hipMemcpyDeviceToHost);
        }
        else
        {
            return hipErrorInvalidContext;
        }
    };

    //!
    //! @brief Initialize with the rocblas random number generator.
    //! @param seedReset if true reset the seed.
    //!
    inline void random_init(bool seedReset = true) noexcept
    {
        if(seedReset)
        {
            rocblas_seedrand();
        }

        auto data
            = (this->m_inc >= 0) ? this->data() : this->data() - (this->m_n - 1) * this->m_inc;
        for(rocblas_int i = 0; i < this->m_n; ++i)
        {
            data[i * this->m_inc] = random_generator<T>();
        }
    };

    //!
    //! @brief Check if memory exists.
    //! @return hipSuccess if memory exists, hipErrorOutOfMemory otherwise.
    //!
    inline hipError_t memcheck() const noexcept
    {
        return (nullptr != (const T*)this) ? hipSuccess : hipErrorOutOfMemory;
    };
};
