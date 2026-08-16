#pragma once

#ifndef __METAL_VERSION__
#include <cstdint>
#endif

namespace backend
{
#define MAX_RANK 8
struct TensorDesc
{
    uint32_t rank, count;

    // precompute strides on CPU since GPU integer divs
    // are expensive
    uint32_t shape[MAX_RANK], strides[MAX_RANK];
};
} // namespace backend
