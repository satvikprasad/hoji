// C ABI over the backend kernels, so Python (ctypes) and any other language can
// call them without a binding library. Pointers are raw so numpy arrays can be
// passed straight through with no copy.
//
// The kernels take no timing or benchmarking arguments: measured ctypes call
// overhead is ~150 ns, which is under 0.4% of the smallest kernel we run, so
// the caller can time in Python without distorting results.
//
// Both backends are exposed by name, not just the selected one, so a single
// loaded library can run the same inputs through both and compare them. The
// unsuffixed hoji_matmul_f32 is whichever _BACKEND_CPU selects.
#include "backend.h"

#include "core/static_view.h"
#include "dtype.h"

#include <cstddef>
#include <cstdio>
#include <limits>

namespace
{
backend::TensorF32<2> in_view(float32_t const *p, size_t rows, size_t cols)
{
    return backend::TensorF32<2>{
        p, {rows, cols}
    };
}

backend::MutTensorF32<2> out_view(float32_t *p, size_t rows, size_t cols)
{
    return backend::MutTensorF32<2>{
        p, {rows, cols}
    };
}
} // namespace

extern "C"
{
    // Register tile cpu.cc was compiled with. They are macros, so one library
    // exposes exactly one CPU tile and sweeps load several libraries. The Metal
    // backend picks its tile per dispatch, so these say nothing about it.
    int hoji_backend_tile_m(void)
    {
        return int(_CPU_MATMUL_TILE_M);
    }

    int hoji_backend_tile_n(void)
    {
        return int(_CPU_MATMUL_TILE_N);
    }

    // Whether this library was linked with metal.cc, so a caller can skip the
    // Metal entry point rather than hitting an unresolved symbol.
    int hoji_backend_has_metal(void)
    {
#ifdef _BACKEND_METAL
        return 1;
#else
        return 0;
#endif
    }

    // out(m x n) = a(m x k) * b_trans(n x k)^T, row-major float32.
    void hoji_matmul_f32_cpu(const float32_t *a, const float32_t *b_trans,
                             float32_t *out, size_t m, size_t k, size_t n)
    {
        backend::cpu::matmul<2>(in_view(a, m, k), in_view(b_trans, n, k),
                                out_view(out, m, n));
    }

#ifdef _BACKEND_METAL
    // Returns 0 on success and non-zero on failure, unlike the CPU entry point:
    // this path can fail before it reaches any arithmetic. The message goes to
    // stderr, since a C ABI has nowhere better to put a string.
    int hoji_matmul_f32_metal(const float32_t *a, const float32_t *b_trans,
                              float32_t *out, size_t m, size_t k, size_t n)
    {
        const auto r = backend::metal::matmul<2>(
            in_view(a, m, k), in_view(b_trans, n, k), out_view(out, m, n));
        if (!r)
        {
            std::fprintf(stderr, "hoji_matmul_f32_metal: %s\n",
                         r.error().c_str());
            return 1;
        }
        return 0;
    }
#endif

    // The selected backend. Kept void for ABI compatibility with callers that
    // predate the suffixed entry points.
    void hoji_matmul_f32(const float32_t *a, const float32_t *b_trans,
                         float32_t *out, size_t m, size_t k, size_t n)
    {
#if defined(_BACKEND_CPU) || !defined(_BACKEND_METAL)
        hoji_matmul_f32_cpu(a, b_trans, out, m, k, n);
#else
        // No return value to hand back, so rather than leave stale data that
        // could silently pass a numerical check, poison the output.
        if (hoji_matmul_f32_metal(a, b_trans, out, m, k, n) != 0)
        {
            for (size_t i = 0; i < m * n; ++i)
            {
                out[i] = std::numeric_limits<float32_t>::quiet_NaN();
            }
        }
#endif
    }
}
