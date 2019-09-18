#pragma once
#include "handle.h"
#include "rocblas.h"
#include "utility.h"
#include <type_traits>
#include <utility>
#include "logging.h"

template <typename T>
struct batched_arrays_traits
{

};


template <typename T>
using strided_batched_arrays = T*;

template <typename T>
using const_strided_batched_arrays = const T*;

template <typename T>
using batched_arrays = T**;

template <typename T>
using const_batched_arrays = const T*const*;


template <typename T>
struct batched_arrays_traits< const_batched_arrays<T> >
{
  using base_t = T;
  using batch_array_t = const T *;
};

template <typename T>
struct batched_arrays_traits< batched_arrays<T> >
{
  using base_t = T;
  using batch_array_t = T *;
};


template <typename T>
struct batched_arrays_traits< const_strided_batched_arrays<T> >
{
  using base_t = T;
  using batch_array_t = const T *;
};

template <typename T>
struct batched_arrays_traits< strided_batched_arrays<T> >
{
  using base_t = T;
  using batch_array_t = T *;
};


template <typename U>
typename batched_arrays_traits<U>::batch_array_t load_batched_ptr(U x,rocblas_int i,rocblas_int stride);


template <typename T>
typename batched_arrays_traits< const_batched_arrays<T> >::batch_array_t load_batched_ptr(const_batched_arrays<T> x, rocblas_int i, rocblas_int stride)
{
  return x[i];
};

template <typename T>
typename batched_arrays_traits< const_strided_batched_arrays<T> >::batch_array_t load_batched_ptr(const_strided_batched_arrays<T> x, rocblas_int i, rocblas_int stride)
{
  return x + i * stride;
};

template <typename T>
typename batched_arrays_traits< strided_batched_arrays<T> >::batch_array_t load_batched_ptr(strided_batched_arrays<T> x, rocblas_int i, rocblas_int stride)
{
  return x + i * stride;
};

template <typename T>
typename batched_arrays_traits< batched_arrays<T> >::batch_array_t load_batched_ptr(batched_arrays<T> x, rocblas_int i, rocblas_int stride)
{
  return x[i];
};
