//
// Copyright 2018-2019 Advanced Micro Devices, Inc.
//
#pragma once
#include <hip/hip_runtime.h>

//!
//! @brief  Allocator which uses pinned host memory.
//!
// template <typename T>
// class pinned_memory_allocator : public std::allocator<T>
// {
// public:
    
//     T* allocate(size_t n, const void *hint=0)
//     {
//       T* ptr;
//       hipHostMalloc(&ptr, sizeof(T) * n, hipHostMallocDefault);   
//       return ptr; 
//     }

//     void deallocate(T* ptr, size_t n)
//     {
//       hipHostFree(ptr);   
//       return; 
//     }
// };

template <class T>
struct pinned_memory_allocator {
  typedef T value_type;
  pinned_memory_allocator() noexcept {}
  template <class U> pinned_memory_allocator (const pinned_memory_allocator<U>&) noexcept {}
  T* allocate (std::size_t n) {       
      T* ptr;
      hipHostMalloc(&ptr, sizeof(T) * n, hipHostMallocDefault);   
      return ptr;  
      }
  void deallocate (T* ptr, std::size_t n) { hipHostFree(ptr); }
};

template <class T, class U>
constexpr bool operator== (const pinned_memory_allocator<T>&, const pinned_memory_allocator<U>&) noexcept
{return true;}

template <class T, class U>
constexpr bool operator!= (const pinned_memory_allocator<T>&, const pinned_memory_allocator<U>&) noexcept
{return false;}

