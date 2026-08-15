#pragma once

#include "core/static_view.h"
#include "dtype.h"


// Register tile dimensions. They must be compile-time constants: the tile has
// to live in registers, which means acc[][] needs literal bounds. Defined here
// rather than in cpu.cc so every backend TU sees the same values; override with
// -D (bench/ builds one shared library per tile).
#ifndef _CPU_MATMUL_TILE_M
#define _CPU_MATMUL_TILE_M 4
#endif
#ifndef _CPU_MATMUL_TILE_N
#define _CPU_MATMUL_TILE_N 8
#endif

namespace backend
{
template <size_t R> using TensorF32    = core::StaticView<const float32_t, R>;
template <size_t R> using MutTensorF32 = core::StaticView<float32_t, R>;

template <size_t R>
void matmul(TensorF32<R> a /* M x K */, TensorF32<R> b_trans /* N x K */,
            MutTensorF32<R> out /* M * N */);
} // namespace backend
