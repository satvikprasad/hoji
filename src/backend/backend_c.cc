// C ABI over the backend kernels, so Python (ctypes) and any other language can
// call them without a binding library. Pointers are raw so numpy arrays can be
// passed straight through with no copy.
//
// The kernels take no timing or benchmarking arguments: measured ctypes call
// overhead is ~150 ns, which is under 0.4% of the smallest kernel we run, so
// the caller can time in Python without distorting results.
#include "backend.h"

#include "core/static_view.h"
#include "dtype.h"

#include <cstddef>

extern "C"
{
    // Folding factor this library was compiled with. The value is a macro, so
    // one library exposes exactly one setting; sweeps load several libraries.
    int hoji_backend_folding_factor(void)
    {
        return int(_CPU_MATMUL_FOLDING_FACTOR);
    }

    // out(m x n) = a(m x k) * b_trans(n x k)^T, row-major float32.
    void hoji_matmul_f32(const float32_t *a, const float32_t *b_trans,
                         float32_t *out, size_t m, size_t k, size_t n)
    {
        const backend::TensorF32<2> av{
            a, {m, k}
        };
        const backend::TensorF32<2> bv{
            b_trans, {n, k}
        };
        const backend::MutTensorF32<2> ov{
            out, {m, n}
        };
        backend::matmul<2>(av, bv, ov);
    }
}
