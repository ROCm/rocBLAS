/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#ifndef HELPERS_H
#define HELPERS_H

inline size_t idx2D(const size_t i, const size_t j, const size_t lda) { return j * lda + i; }

#endif /* HELPERS_H */
