#pragma once

#include "backend/backend.h"
#include "backend/tensor_desc.h"
#include "dtype.h"
#include <metal_cpp.hpp>

#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <utility>

namespace backend
{
namespace metal
{
template <typename T> using shared_ptr = NS::SharedPtr<T>;

#define HOJI_METAL_KERNELS                                                     \
    X(matmul, 1, 1)                                                            \
    X(matmul, 1, 4)                                                            \
    X(matmul, 2, 2)                                                            \
    X(matmul, 4, 4)                                                            \
    X(matmul, 4, 8)                                                            \
    X(matmul, 8, 4)

#define X(k, T_M, T_N) k##_##T_M##x##T_N,
enum class KernelTypes : uint8_t
{
    HOJI_METAL_KERNELS COUNT
};
#undef X

#define X(k, T_M, T_N) #k "_" #T_M "x" #T_N,
constexpr std::string_view pKernels[] = {HOJI_METAL_KERNELS};
#undef X
static_assert(std::size(pKernels) == static_cast<size_t>(KernelTypes::COUNT),
              "pKernels and KernelTypes must stay in step");

struct Kernel
{
    shared_ptr<MTL::ComputePipelineState> pso;
};

/*
 * At a high level, a backend should support:
 */
struct Backend
{
    shared_ptr<MTL::Device>       device;
    shared_ptr<MTL::CommandQueue> cmd_q;

    /* indexed by backend::metal::KernelTypes */
    Kernel kernels[util::enum_size<KernelTypes>()];

    NS::SharedPtr<MTL::Buffer> buffers[util::enum_size<DataBufferType>()];
    uint32_t                   buf_caps[util::enum_size<DataBufferType>()];

    static std::expected<Backend, std::string> load() noexcept;

    Allocator<Backend> get_allocator() const noexcept;

    template <size_t T_M, size_t T_N>
    void matmul(Tensor const a, Tensor const b, Tensor const out) const
    {
        MTL::CommandBuffer         *cmd_buf = cmd_q->commandBuffer();
        MTL::ComputeCommandEncoder *cmp_enc = cmd_buf->computeCommandEncoder();

        // KernelTypes is a scoped enum, so it needs an explicit conversion to
        // index kernels[] -- that is the point of `enum class`.
        cmp_enc->setComputePipelineState(
            this->kernels[util::enum_index(
                              match_kernel_type<T_M, T_N>("matmul"))]
                .pso.get());

        /* encode tensors */
        encode_tensor(cmp_enc, a, 0);
        encode_tensor(cmp_enc, b, 1);
        encode_tensor(cmp_enc, out, 2);

        cmp_enc->setBytes(&a.desc, sizeof a.desc, 3);
        cmp_enc->setBytes(&out.desc, sizeof out.desc, 4);

        size_t const M = out.desc.shape[TensorDesc::MAX_RANK - 2];
        size_t const N = out.desc.shape[TensorDesc::MAX_RANK - 1];

        auto const grid_size =
            MTL::Size::Make(util::div_ceil(N, T_N), util::div_ceil(M, T_M), 1);
        auto const thread_group_size = MTL::Size::Make(
            32, 32, 1); /* TODO: for now, use an 32x32 thread group */

        cmp_enc->dispatchThreads(grid_size, thread_group_size);
        cmp_enc->endEncoding();

        cmd_buf->commit();
        cmd_buf->waitUntilCompleted();
    }

private:
    void encode_tensor(MTL::ComputeCommandEncoder *cmp_enc, Tensor const tensor,
                       uint8_t const index_in_fn) const;

    constexpr NS::SharedPtr<MTL::Buffer>
    get_databuf_from_type(DataBufferType const type) const
    {
        return buffers[util::enum_index(type)];
    }

    template <size_t T_M, size_t T_N>
    constexpr static KernelTypes
    match_kernel_type(std::string_view const kernel)
    {
#define X(k_id, t_M, t_N)                                                      \
    if (kernel == #k_id && T_M == t_M && T_N == t_N)                           \
    {                                                                          \
        return KernelTypes::k_id##_##t_M##x##t_N;                              \
    }

        HOJI_METAL_KERNELS
#undef X
        std::unreachable();
    }
}; // namespace metal
} // namespace metal
} // namespace backend
