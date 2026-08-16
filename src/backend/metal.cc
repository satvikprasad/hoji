#include "backend.h"

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <metal_cpp.hpp>

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// Absolute path of the metallib the build produced. CMake compiles it in; the
// fallback keeps this TU compilable when the Metal toolchain was missing at
// configure time, in which case metal_init() reports it at runtime instead.
#ifndef HOJI_METALLIB_PATH
#define HOJI_METALLIB_PATH ""
#endif

namespace backend
{
namespace
{
struct Kernel
{
    std::string                              name;
    NS::SharedPtr<MTL::ComputePipelineState> pipeline;

    // Parsed from matmul_<T_M>x<T_N>; 0 when the name doesn't follow it.
    uint32_t tile_m = 0, tile_n = 0;

    // Both are properties of the *pipeline*, not the device: they depend on how
    // many registers the compiled kernel needs. Cached here because picking a
    // threadsPerThreadgroup needs them at every dispatch.
    uint32_t max_threads_per_threadgroup = 0;
    uint32_t thread_execution_width      = 0;
};

struct MetalBackend
{
    NS::SharedPtr<MTL::Device>       device;
    NS::SharedPtr<MTL::CommandQueue> queue;
    NS::SharedPtr<MTL::Library>      library;
    std::vector<Kernel>              kernels;
};

std::string describe(NS::Error *err)
{
    if (err == nullptr)
    {
        return "unknown error";
    }
    NS::String *msg = err->localizedDescription();
    return msg != nullptr ? std::string(msg->utf8String()) : "unknown error";
}

// "matmul_4x8" -> (4, 8). Unrecognised names still get a pipeline, just no
// tile.
bool parse_tile(std::string_view name, uint32_t &tm, uint32_t &tn)
{
    constexpr std::string_view prefix = "matmul_";
    if (!name.starts_with(prefix))
    {
        return false;
    }
    name.remove_prefix(prefix.size());
    size_t const sep = name.find('x');
    if (sep == std::string_view::npos)
    {
        return false;
    }
    std::string_view const lhs = name.substr(0, sep),
                           rhs = name.substr(sep + 1);
    auto const a               = std::from_chars(lhs.begin(), lhs.end(), tm);
    auto const b               = std::from_chars(rhs.begin(), rhs.end(), tn);
    return a.ec == std::errc{} && a.ptr == lhs.end() && b.ec == std::errc{} &&
           b.ptr == rhs.end();
}

// Opens hoji.metallib and builds a compute pipeline for every kernel it
// exports, so one library covers the whole register-tile sweep and the tile is
// chosen by name at dispatch rather than by rebuilding.
std::expected<MetalBackend, std::string> _load_backend()
{
    // Everything metal-cpp returns from a non-new* method is autoreleased, so
    // the whole load runs inside one pool. The objects we keep are all +1 via
    // TransferPtr, so they outlive the drain.
    NS::SharedPtr<NS::AutoreleasePool> const pool =
        NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

    MetalBackend be;

    be.device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
    if (!be.device)
    {
        return std::unexpected("no Metal device available");
    }

    be.queue = NS::TransferPtr(be.device->newCommandQueue());
    if (!be.queue)
    {
        return std::unexpected("could not create a Metal command queue");
    }

    // The env var exists so a test can point at a metallib it built itself.
    char const *path = std::getenv("HOJI_METALLIB");
    if (path == nullptr || *path == '\0')
    {
        path = HOJI_METALLIB_PATH;
    }
    if (*path == '\0')
    {
        return std::unexpected(
            "no metallib path compiled in; the Metal toolchain was missing at "
            "configure time, or set HOJI_METALLIB");
    }

    NS::Error  *err     = nullptr;
    NS::String *ns_path = NS::String::string(path, NS::UTF8StringEncoding);
    be.library          = NS::TransferPtr(
        be.device->newLibrary(NS::URL::fileURLWithPath(ns_path), &err));
    if (!be.library)
    {
        return std::unexpected("could not load " + std::string(path) + ": " +
                               describe(err));
    }

    NS::Array *names = be.library->functionNames();
    if (names == nullptr || names->count() == 0)
    {
        return std::unexpected("no kernels in " + std::string(path));
    }

    be.kernels.reserve(names->count());
    for (NS::UInteger i = 0; i < names->count(); ++i)
    {
        NS::String *ns_name = names->object<NS::String>(i);
        std::string name    = ns_name->utf8String();

        NS::SharedPtr<MTL::Function> const fn =
            NS::TransferPtr(be.library->newFunction(ns_name));
        if (!fn)
        {
            return std::unexpected("no function '" + name + "' in " + path);
        }

        err = nullptr;
        NS::SharedPtr<MTL::ComputePipelineState> pipeline =
            NS::TransferPtr(be.device->newComputePipelineState(fn.get(), &err));
        if (!pipeline)
        {
            return std::unexpected("could not build a pipeline for '" + name +
                                   "': " + describe(err));
        }

        Kernel k;
        k.name = std::move(name);
        k.max_threads_per_threadgroup =
            static_cast<uint32_t>(pipeline->maxTotalThreadsPerThreadgroup());
        k.thread_execution_width =
            static_cast<uint32_t>(pipeline->threadExecutionWidth());
        k.pipeline = std::move(pipeline);
        parse_tile(k.name, k.tile_m, k.tile_n);
        be.kernels.push_back(std::move(k));
    }

    // Smallest tile first, matching how bench/ orders the CPU backends.
    std::ranges::sort(be.kernels,
                      [](Kernel const &a, Kernel const &b)
                      {
                          return std::pair(a.tile_m * a.tile_n, a.tile_m) <
                                 std::pair(b.tile_m * b.tile_n, b.tile_m);
                      });

    return be;
}

// Loaded once on first use; the error is cached too, so a missing metallib is
// reported identically on every call rather than retried per dispatch.
std::expected<MetalBackend, std::string> const &backend_once()
{
    static std::expected<MetalBackend, std::string> const be = _load_backend();
    return be;
}
} // namespace

std::expected<void, std::string> metal_init()
{
    auto const &be = backend_once();
    if (!be)
    {
        return std::unexpected(be.error());
    }
    return {};
}

std::vector<KernelInfo> metal_kernels()
{
    auto const &be = backend_once();
    if (!be)
    {
        return {};
    }
    std::vector<KernelInfo> out;
    out.reserve(be->kernels.size());
    for (Kernel const &k : be->kernels)
    {
        out.push_back(KernelInfo{k.name, k.tile_m, k.tile_n,
                                 k.max_threads_per_threadgroup,
                                 k.thread_execution_width});
    }
    return out;
}

namespace
{
uint32_t ceil_div(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

// Which register tile to run. At M=1 (decode) a 4-row tile wastes three of its
// four rows, so prefer a 1-row one; otherwise 4x8, the fastest of the compiled
// set on prefill shapes. Falls back to whatever the metallib actually has.
Kernel const &choose_kernel(MetalBackend const &be, uint32_t M)
{
    uint32_t const want_m = M == 1 ? 1u : 4u;
    uint32_t const want_n = M == 1 ? 4u : 8u;
    for (Kernel const &k : be.kernels)
    {
        if (k.tile_m == want_m && k.tile_n == want_n)
        {
            return k;
        }
    }
    return be.kernels.front();
}

// threadsPerThreadgroup, which is the largest lever in the whole kernel: it
// measured an 11.6x spread on an M4 at M=512 N=4864 K=896, because a threadgroup
// is the unit of co-residency on a core and its aspect ratio decides what fits
// in that core's cache. The footprint is T_M*ty + T_N*tx rows, so x stays narrow
// -- with T_N > T_M, growing tx costs more than growing ty.
//
// 8x8 measured best at M=512 and 4x8 at M=50, so the row-tile count picks
// between them. Both are then clamped to what this pipeline actually permits.
MTL::Size choose_threadgroup(Kernel const &k, uint32_t M)
{
    uint32_t const row_tiles = ceil_div(M, k.tile_m);
    uint32_t       tx        = row_tiles >= 64 ? 8u : 4u;
    uint32_t       ty        = 8u;

    uint32_t const width =
        k.thread_execution_width > 0 ? k.thread_execution_width : 32u;
    uint32_t const cap =
        k.max_threads_per_threadgroup > 0 ? k.max_threads_per_threadgroup : 1024u;

    // Shrink until it fits and is a whole number of SIMD-groups. Halving ty
    // first keeps x wide, which is what keeps the C stores coalesced.
    while (tx * ty > cap || (tx * ty) % width != 0)
    {
        if (ty > 1)
        {
            ty /= 2;
        }
        else if (tx > 1)
        {
            tx /= 2;
        }
        else
        {
            break;
        }
    }
    return MTL::Size::Make(tx, ty, 1);
}

TensorDesc desc_2d(uint32_t rows, uint32_t cols)
{
    TensorDesc d{};
    d.rank     = 2;
    d.count    = rows * cols;
    d.shape[0] = rows;
    d.shape[1] = cols;
    // Row-major. The kernel indexes arithmetically and ignores these, but the
    // descriptor carries them so a strided variant needn't change the ABI.
    d.strides[0] = cols;
    d.strides[1] = 1;
    return d;
}
} // namespace

// Deliberately backend::metal::matmul, not backend::matmul: cpu.cc explicitly
// instantiates backend::matmul<2> and <3> as strong symbols, so a second
// definition of that name would be a duplicate-symbol link error.
namespace metal
{
// Returns an error rather than void (unlike the CPU overload) because every step
// here can fail for reasons the caller can't check in advance -- no device, no
// metallib, allocation failure, a GPU-side fault.
template <size_t R>
std::expected<void, std::string> matmul(TensorF32<R> a /* M x K */,
                                        TensorF32<R> b_trans /* N x K */,
                                        MutTensorF32<R> out /* M x N */)
{
    static_assert(R >= 2);

    auto const &loaded = backend_once();
    if (!loaded)
    {
        return std::unexpected(loaded.error());
    }
    MetalBackend const &be = *loaded;

    uint32_t const M = static_cast<uint32_t>(a.dims[R - 2]);
    uint32_t const K = static_cast<uint32_t>(a.dims[R - 1]);
    uint32_t const N = static_cast<uint32_t>(b_trans.dims[R - 2]);
    if (b_trans.dims[R - 1] != a.dims[R - 1])
    {
        return std::unexpected("inner dims disagree: " + std::to_string(K) +
                               " vs " +
                               std::to_string(b_trans.dims[R - 1]));
    }
    if (M == 0 || N == 0 || K == 0)
    {
        return {};
    }

    size_t batches = 1;
    for (size_t i = 0; i + 2 < R; ++i)
    {
        batches *= a.dims[i];
    }

    Kernel const   &k  = choose_kernel(be, M);
    MTL::Size const tg = choose_threadgroup(k, M);

    NS::SharedPtr<NS::AutoreleasePool> const pool =
        NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

    size_t const bytes_a = batches * M * K * sizeof(float32_t);
    size_t const bytes_b = batches * N * K * sizeof(float32_t);
    size_t const bytes_c = batches * M * N * sizeof(float32_t);

    // Unified memory: the GPU can read the caller's allocation directly, so wrap
    // it rather than copy. newBuffer(ptr, len, opts, deallocator) hands back a
    // handle whose contents() IS ptr -- no host memcpy in either direction, and
    // the output is written straight into out.data. A null deallocator means
    // Metal never frees it; the caller keeps ownership, and the buffer must not
    // outlive the array or see it resized mid-dispatch.
    //
    // Apple documents a page-aligned-pointer requirement for this call which is
    // not enforced on Apple silicon (measured: a 400-byte vector at byte offset
    // 7248 wraps and dispatches correctly). wrap_or_copy keeps the copying
    // overload as a fallback so a future OS enforcing it degrades to slow rather
    // than broken.
    auto wrap = [&](void const *p, size_t bytes) {
        NS::SharedPtr<MTL::Buffer> buf = NS::TransferPtr(be.device->newBuffer(
            p, bytes, MTL::ResourceStorageModeShared, nullptr));
        if (!buf)
        {
            buf = NS::TransferPtr(
                be.device->newBuffer(p, bytes, MTL::ResourceStorageModeShared));
        }
        return buf;
    };

    NS::SharedPtr<MTL::Buffer> const buf_a = wrap(a.data, bytes_a);
    NS::SharedPtr<MTL::Buffer> const buf_b = wrap(b_trans.data, bytes_b);

    // The output is wrapped too, so the kernel writes into out.data in place. On
    // the fallback path the buffer is a separate allocation and has to be copied
    // back after the dispatch.
    NS::SharedPtr<MTL::Buffer> buf_c = NS::TransferPtr(be.device->newBuffer(
        out.data, bytes_c, MTL::ResourceStorageModeShared, nullptr));
    bool const wrote_in_place = static_cast<bool>(buf_c);
    if (!buf_c)
    {
        buf_c = NS::TransferPtr(
            be.device->newBuffer(bytes_c, MTL::ResourceStorageModeShared));
    }
    if (!buf_a || !buf_b || !buf_c)
    {
        return std::unexpected("could not create Metal buffers");
    }

    MTL::CommandBuffer *cb = be.queue->commandBuffer();
    if (cb == nullptr)
    {
        return std::unexpected("could not create a command buffer");
    }
    MTL::ComputeCommandEncoder *enc = cb->computeCommandEncoder();
    if (enc == nullptr)
    {
        return std::unexpected("could not create a compute encoder");
    }

    TensorDesc const a_desc   = desc_2d(M, K);
    TensorDesc const out_desc = desc_2d(M, N);

    // The kernel's grid is counted in threads, each owning one T_M x T_N tile,
    // so the threadgroup count is ceil(ceil(N/T_N)/tx) x ceil(ceil(M/T_M)/ty).
    MTL::Size const grid = MTL::Size::Make(
        ceil_div(ceil_div(N, k.tile_n), static_cast<uint32_t>(tg.width)),
        ceil_div(ceil_div(M, k.tile_m), static_cast<uint32_t>(tg.height)), 1);

    enc->setComputePipelineState(k.pipeline.get());
    // One dispatch per batch, sharing the buffers via offsets. The kernel is 2D,
    // so the grid's z axis stays free rather than being spent on the batch.
    for (size_t batch = 0; batch < batches; ++batch)
    {
        enc->setBuffer(buf_a.get(), batch * M * K * sizeof(float32_t), 0);
        enc->setBuffer(buf_b.get(), batch * N * K * sizeof(float32_t), 1);
        enc->setBuffer(buf_c.get(), batch * M * N * sizeof(float32_t), 2);
        enc->setBytes(&a_desc, sizeof a_desc, 3);
        enc->setBytes(&out_desc, sizeof out_desc, 4);
        enc->dispatchThreadgroups(grid, tg);
    }
    enc->endEncoding();
    cb->commit();
    cb->waitUntilCompleted();

    if (NS::Error *err = cb->error(); err != nullptr)
    {
        return std::unexpected("dispatch failed: " + describe(err));
    }

    // Only needed on the fallback path; when the output was wrapped, the kernel
    // has already written into out.data.
    if (!wrote_in_place)
    {
        std::memcpy(out.data, buf_c->contents(), bytes_c);
    }
    return {};
}

template std::expected<void, std::string>
    matmul<2>(TensorF32<2>, TensorF32<2>, MutTensorF32<2>);
template std::expected<void, std::string>
    matmul<3>(TensorF32<3>, TensorF32<3>, MutTensorF32<3>);
} // namespace metal
} // namespace backend
