/* ************************************************************************
 * Copyright 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

//!
//! @brief Host free memory w/o swap.  Returns kB or -1 if unknown.
//!
ptrdiff_t host_bytes_available();

//!
//! @brief Allocates memory which can be freed with free.  Returns nullptr if swap required.
//!
void* host_malloc(size_t size);

//!
//! @brief Allocates memory which can be freed with free.  Throws exception if swap required.
//!
inline void* host_malloc_throw(size_t nmemb, size_t size)
{
    void* ptr = host_malloc(nmemb * size);
    if(!ptr)
    {
        throw std::bad_alloc{};
    }
    return ptr;
}

//!
//! @brief Allocates cleared memory which can be freed with free.  Returns nullptr if swap required.
//!
void* host_calloc(size_t nmemb, size_t size);

//!
//! @brief Allocates cleared memory which can be freed with free.  Throws exception if swap required.
//!
inline void* host_calloc_throw(size_t nmemb, size_t size)
{
    void* ptr = host_calloc(nmemb, size);
    if(!ptr)
    {
        throw std::bad_alloc{};
    }
    return ptr;
}

//!
//! @brief  Allocator which allocates with host_calloc
//!
template <class T>
struct host_memory_allocator
{
    using value_type = T;

    host_memory_allocator() = default;

    template <class U>
    host_memory_allocator(const host_memory_allocator<U>&)
    {
    }

    T* allocate(std::size_t n)
    {
        return (T*)host_malloc_throw(n, sizeof(T));
    }

    void deallocate(T* ptr, std::size_t n)
    {
        free(ptr);
    }
};

template <class T, class U>
constexpr bool operator==(const host_memory_allocator<T>&, const host_memory_allocator<U>&)
{
    return true;
}

template <class T, class U>
constexpr bool operator!=(const host_memory_allocator<T>&, const host_memory_allocator<U>&)
{
    return false;
}
