#pragma once

#include "core/static_view.h"
#include "dtype.h"

#ifndef _CPU_MATMUL_FOLDING_FACTOR
#define _CPU_MATMUL_FOLDING_FACTOR 8ul
#endif

namespace backend
{
template <size_t R> using TensorF32    = core::StaticView<const float32_t, R>;
template <size_t R> using MutTensorF32 = core::StaticView<float32_t, R>;

template <size_t R>
void matmul(TensorF32<R> a /* M x K */, TensorF32<R> b_trans /* N x K */,
            MutTensorF32<R> out /* M * N */);
} // namespace backend
