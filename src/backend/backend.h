#pragma once

#include "backend/tensor_desc.h"
#include "core/static_view.h"
#include "dtype.h"

#include <exception>
#include <expected>
#include <span>
#include <stdexcept>
#include <type_traits>

#ifndef _CPU_MATMUL_TILE_M
#define _CPU_MATMUL_TILE_M 4
#endif
#ifndef _CPU_MATMUL_TILE_N
#define _CPU_MATMUL_TILE_N 8
#endif

#ifndef _CPU_MATMUL_CACHE_BLOCK
#define _CPU_MATMUL_CACHE_BLOCK 512
#endif

namespace backend
{
template <size_t R> using TensorF32    = core::StaticView<float32_t const, R>;
template <size_t R> using MutTensorF32 = core::StaticView<float32_t, R>;

enum class DataBufferType : uint8_t
{
    Wgt,
    Actvt,
    KV,
    COUNT
};

struct Tensor
{
    /* Store offset into corresponding metal buffer in bytes */
    uint32_t offset;

    TensorDesc     desc;
    DataBufferType buf_type;
};

template <typename B> struct Allocator
{
    uint32_t offsets[util::enum_size<DataBufferType>()] = {0};
    B const &backend;

    Tensor push_tensor(TensorDesc const desc, DataBufferType alloc_type)
    {
        uint32_t offset = offsets[util::enum_index(alloc_type)];
        if (offset + desc.count * sizeof(float32_t) >
            backend.buf_caps[util::enum_index(alloc_type)])
        {
            throw std::runtime_error("ran out of capacity");
        }

        offsets[util::enum_index(alloc_type)] =
            offset + desc.count * sizeof(float32_t);

        return Tensor{.desc = desc, .buf_type = alloc_type, .offset = offset};
    }

    std::span<float32_t>       view_tensor_mut(Tensor const tensor) const;
    std::span<float32_t const> view_tensor(Tensor const tensor) const;
};

namespace cpu
{
// Register tiles come from the _CPU_MATMUL_* macros above, so one translation
// unit exposes exactly one tile.
template <size_t R>
void matmul(TensorF32<R> a /* M x K */, TensorF32<R> b_trans /* N x K */,
            MutTensorF32<R> out /* M x N */);
} // namespace cpu

} // namespace backend
