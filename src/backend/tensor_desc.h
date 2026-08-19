#pragma once

#ifndef __METAL_VERSION__
#include <cstdint>
#endif

namespace backend
{

struct TensorDesc
{
    enum : uint32_t
    {
        MAX_RANK = 8
    };

    uint32_t rank, count;
    uint32_t shape[MAX_RANK], strides[MAX_RANK];

#ifndef __METAL_VERSION__
    TensorDesc(TensorDesc const &other)            = default;
    TensorDesc &operator=(TensorDesc const &other) = default;

    TensorDesc(TensorDesc &&other)            = delete;
    TensorDesc &operator=(TensorDesc &&other) = delete;
    TensorDesc()                              = delete;

    template <uint32_t R>
    TensorDesc(uint32_t const (&tensor_shape)[R]) noexcept : rank(R)
    {
        static_assert(R > 0);
        static_assert(R <= MAX_RANK);

        strides[MAX_RANK - 1] = 1;
        shape[MAX_RANK - 1]   = tensor_shape[R - 1];
        count                 = tensor_shape[R - 1];

        for (uint32_t r = 1; r < R; ++r)
        {
            uint32_t const in_idx   = R - 1 - r;
            uint32_t const cnst_idx = MAX_RANK - 1 - r;

            shape[cnst_idx] = tensor_shape[in_idx];
            strides[cnst_idx] =
                strides[cnst_idx + 1] * tensor_shape[in_idx + 1];
            count *= tensor_shape[in_idx];
        }

        // pad the leading slots so a stride walk over them is a no-op
        for (uint32_t r = 0; r < MAX_RANK - R; ++r)
        {
            shape[r] = strides[r] = 1;
        }
    }
#endif
};
} // namespace backend
