#include "backend.h"

#include <arm_neon.h>
#include <cstddef>

namespace backend
{
template <size_t R>
void matmul(TensorF32<R> a /* M x K */, TensorF32<R> b_trans /* N x K */,
            MutTensorF32<R> out /* M * N */)
{
    static_assert(R >= 2);

    /*
     * To sature the FMAC units on the M-series chips, fold
     * along the k-dimension allowing _CPU_MATMUL_FOLDING_FACTOR
     * accumulators to be issued on non RAW-dependent parallel
     * streams.
     */
    constexpr size_t folding_factor = _CPU_MATMUL_FOLDING_FACTOR;

    size_t batches = 1;
    for (uint32_t i = 0; i < R - 2; ++i)
    {
        batches *= a.dims[i];
    }

    for (size_t batch = 0; batch < batches; ++batch)
    {
        size_t M = a.dims[R - 2];
        size_t K = a.dims[R - 1];
        size_t N = b_trans.dims[R - 2];

        const size_t out_batch_offset = batch * M * N;
        const size_t a_batch_offset   = batch * M * K;
        const size_t b_batch_offset   = batch * K * N;

        for (size_t i = 0; i < M; ++i)
        {
            const size_t out_column_offset = i * N + out_batch_offset;
            const size_t a_column_offset   = i * K + a_batch_offset;

            for (size_t j = 0; j < N; ++j)
            {
                const size_t out_idx = out_column_offset + j;

                const size_t fold_size      = K / folding_factor;
                const size_t fold_remainder = K % folding_factor;

                float32_t accs[folding_factor] = {0.f};

                for (size_t k = 0; k < fold_size; ++k)
                {
                    for (size_t fold = 0; fold < folding_factor; ++fold)
                    {
                        const size_t inner = fold * fold_size + k;

                        accs[fold] += a[a_column_offset + inner] *
                                      b_trans[b_batch_offset + j * K + inner];
                    }
                }

                // handle fold remainder
                for (size_t k = 0; k < fold_remainder; ++k)
                {
                    const size_t inner = folding_factor * fold_size + k;

                    accs[k] += a[a_column_offset + inner] *
                               b_trans[b_batch_offset + j * K + inner];
                }

                float32_t acc = 0.f;
                for (size_t fold = 0; fold < folding_factor; ++fold)
                {
                    acc += accs[fold];
                }

                out[out_idx] = acc;
            }
        }
    }
}

// Explicit instantiation: the definition lives here, not in the header, so
// every rank a caller needs must be named.
template void matmul<2>(TensorF32<2>, TensorF32<2>, MutTensorF32<2>);
template void matmul<3>(TensorF32<3>, TensorF32<3>, MutTensorF32<3>);
}; // namespace backend
