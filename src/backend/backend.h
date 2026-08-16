#pragma once

#include "backend/tensor_desc.h"
#include "core/static_view.h"
#include "dtype.h"

#include <cstdint>
#include <expected>
#include <string>
#include <vector>

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

#ifndef _CPU_MATMUL_CACHE_BLOCK
#define _CPU_MATMUL_CACHE_BLOCK 512
#endif

namespace backend
{
// TensorDesc and MAX_RANK live in tensor_desc.h so the .metal kernels can share
// them without dragging the standard library into MSL.

template <size_t R> using TensorF32    = core::StaticView<float32_t const, R>;
template <size_t R> using MutTensorF32 = core::StaticView<float32_t, R>;

// ---------------------------------------------------------------------------
// Backends
//
// Both namespaces are compiled and linked together, so either can be called and
// both always type-check. _BACKEND_CPU only decides which one backend_c.cc's
// hoji_matmul_f32 routes to; nothing here is conditional on it.
//
// The signatures differ deliberately: the CPU path cannot fail, while the Metal
// path can before it ever reaches arithmetic -- no device, no metallib, a failed
// allocation, a GPU fault.
// ---------------------------------------------------------------------------
namespace cpu
{
// Register tiles come from the _CPU_MATMUL_* macros above, so one translation
// unit exposes exactly one tile.
template <size_t R>
void matmul(TensorF32<R> a /* M x K */, TensorF32<R> b_trans /* N x K */,
            MutTensorF32<R> out /* M x N */);
} // namespace cpu

// ---------------------------------------------------------------------------
// Metal backend (metal.cc)
//
// One hoji.metallib holds every register-tile specialisation of the matmul
// kernel; metal_init() loads it and builds a compute pipeline for each, so a
// tile is selected by name at dispatch rather than by rebuilding.
// ---------------------------------------------------------------------------
struct KernelInfo
{
    std::string name;

    // From matmul_<T_M>x<T_N>; 0 if the name doesn't follow that form.
    uint32_t tile_m, tile_n;

    // Per pipeline, not per device: both depend on the compiled kernel's
    // register use, so a dispatch needs the values for the tile it picked.
    uint32_t max_threads_per_threadgroup, thread_execution_width;
};

// Loads the metallib on first call and caches the outcome, error included.
std::expected<void, std::string> metal_init();

// Every kernel that loaded, smallest tile first. Empty if metal_init() failed.
std::vector<KernelInfo> metal_kernels();

namespace metal
{
// Same operand convention as cpu::matmul. The register tile is chosen per
// dispatch from the shapes, so there is no compile-time tile here.
template <size_t R>
std::expected<void, std::string> matmul(TensorF32<R> a /* M x K */,
                                        TensorF32<R> b_trans /* N x K */,
                                        MutTensorF32<R> out /* M x N */);
} // namespace metal
} // namespace backend
