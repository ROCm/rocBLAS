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
  hipError_t transfer_from(const device_vector<T>&that) noexcept
  {
    if (that.size() == this->size())
      {
	return hipMemcpy(this->data(), (const T*)that, sizeof(T) * this->size(), hipMemcpyDeviceToHost);
      }
    else
      {
	return hipErrorInvalidContext;
      }    
  };
  
};
