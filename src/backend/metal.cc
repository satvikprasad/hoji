// NS_PRIVATE_IMPLEMENTATION / MTL_PRIVATE_IMPLEMENTATION are set on this target
// by CMakeLists.txt, not here: they must precede every inclusion of
// metal_cpp.hpp, and clang-format sorts include blocks, so a #define at the top
// of this file cannot hold that position.
#include "metal.h"
#include "backend/backend.h"
#include "dtype.h"

#include <iostream>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <metal_cpp.hpp>
#include <string>
#include <string_view>
#include <utility>

#ifndef HOJI_METALLIB_PATH
#define HOJI_METALLIB_PATH ""
#endif

namespace backend
{
namespace metal
{
namespace
{
std::expected<void, std::string> load_buffers(
    shared_ptr<MTL::Device> const device,
    NS::SharedPtr<MTL::Buffer> (&buffers)[util::enum_size<DataBufferType>()],
    uint32_t (&buf_caps)[util::enum_size<DataBufferType>()])
{
#define _INIT_SHARED(sz, idx)                                                  \
    do                                                                         \
    {                                                                          \
        std::cout << "allocating " << (sz) << " bytes to " << (idx) << " pool" \
                  << std::endl;                                                \
        buffers[(idx)] = NS::TransferPtr(                                      \
            device->newBuffer((sz), MTL::ResourceStorageModeShared));          \
        buf_caps[(idx)] = sz;                                                  \
        if (!buffers[(idx)])                                                   \
        {                                                                      \
            return std::unexpected("failed to allocate " +                     \
                                   std::to_string((idx)) + " buffer");         \
        }                                                                      \
    } while (false)
    for (uint8_t i = 0; i < util::enum_size<DataBufferType>(); ++i)
    {
        _INIT_SHARED(1 << 20, static_cast<size_t>(i));
    }
#undef _INIT_SHARED
    return {};
}

// Load all the kernels specified in pKernels and store their pipeline state
std::expected<void, std::string>
load_kernels(shared_ptr<MTL::Device> device, shared_ptr<MTL::Library> lib,
             Kernel (&kernels)[static_cast<size_t>(KernelTypes::COUNT)])
{
    NS::Error *error;

    for (uint8_t i = 0; i < static_cast<uint8_t>(KernelTypes::COUNT); ++i)
    {
        std::string_view const &kernel_name = pKernels[i];
        std::cout << "loading kernel " << kernel_name << std::endl;

        auto const kernel_fn =
            NS::TransferPtr(lib->newFunction(NS::String::string(
                kernel_name.data(), NS::StringEncoding::ASCIIStringEncoding)));
        if (!kernel_fn)
        {
            return std::unexpected("failed to load kernel " +
                                   std::string(kernel_name));
        }

        auto pipeline_state = NS::TransferPtr(
            device->newComputePipelineState(kernel_fn.get(), &error));
        if (error)
        {
            return std::unexpected("failed to create pipeline for kernel " +
                                   std::string(kernel_name));
        }

        kernels[i] = Kernel{.pso = std::move(pipeline_state)};
    }

    return {};
}

} // namespace

std::expected<Backend, std::string> Backend::load() noexcept
{
    NS::Error *error = nullptr;

    // create GPU device and load library
    auto const device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
    auto const lib    = NS::TransferPtr(device->newLibrary(
        NS::String::string(HOJI_METALLIB_PATH,
                              NS::StringEncoding::ASCIIStringEncoding),
        &error));

    if (error)
    {
        return std::unexpected(
            std::string("could not load metal library: ") +
            error->localizedDescription()->cString(NS::ASCIIStringEncoding));
    }
    if (!lib)
    {
        return std::unexpected("failed to find the default library");
    }

#define _UNWRAP(m)                                                             \
    if (auto mm = (m); !mm)                                                    \
    {                                                                          \
        return std::unexpected(mm.error());                                    \
    }
    Backend b{.device = device};

    {
        // load kernels
        _UNWRAP(load_kernels(device, lib, b.kernels));

        // create command queue
        b.cmd_q = NS::TransferPtr(device->newCommandQueue());
        _UNWRAP(load_buffers(device, b.buffers, b.buf_caps));
    }
#undef _UNWRAP

    return b;
}

Allocator<Backend> Backend::get_allocator() const noexcept
{
    return {.backend = *this};
};

void Backend::encode_tensor(MTL::ComputeCommandEncoder *cmp_enc,
                            Tensor const                tensor,
                            uint8_t const               index_in_fn) const
{
    auto const buf = get_databuf_from_type(tensor.buf_type);
    cmp_enc->setBuffer(buf.get(), tensor.offset, index_in_fn);
}
} // namespace metal

template <>
std::span<float32_t>
Allocator<metal::Backend>::view_tensor_mut(Tensor const tensor) const
{
    uint8_t *bytes =
        reinterpret_cast<uint8_t *>(
            backend.buffers[util::enum_index(tensor.buf_type)]->contents()) +
        tensor.offset;

    return std::span(reinterpret_cast<float32_t *>(bytes), tensor.desc.count);
}

template <>
std::span<float32_t const>
Allocator<metal::Backend>::view_tensor(Tensor const tensor) const
{
    uint8_t const *bytes =
        reinterpret_cast<uint8_t const *>(
            backend.buffers[util::enum_index(tensor.buf_type)]->contents()) +
        tensor.offset;

    return std::span(reinterpret_cast<float32_t const *>(bytes),
                     tensor.desc.count);
}
}; // namespace backend
