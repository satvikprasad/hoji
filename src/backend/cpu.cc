#include "backend.h"
#include "dtype.h"

#include <arm_neon.h>
#include <cstddef>
#include <span>

namespace backend
{
namespace
{
inline float32x4_t load_partial(float32_t const *p, size_t const n)
{
    float32x4_t v = vdupq_n_f32(0.f);

    switch (n)
    {
    case 3:
        v = vld1q_lane_f32(p + 2, v, 2);
        [[fallthrough]];
    case 2:
        v = vld1q_lane_f32(p + 1, v, 1);
        [[fallthrough]];
    case 1:
        v = vld1q_lane_f32(p + 0, v, 0);
        [[fallthrough]];
    default:
        break;
    }

    return v;
}

template <size_t TILE_M, size_t TILE_N>
void _matmul_at_chunk(std::span<float32_t const> a,
                      std::span<float32_t const> b_trans,
                      std::span<float32_t> out, size_t const K, size_t const N,
                      size_t const tile_m, size_t const tile_n,
                      size_t const tile_m_size, size_t const tile_n_size)
{
    size_t const i0 = tile_m * TILE_M;
    size_t const j0 = tile_n * TILE_N;

    [[assume(tile_m_size <= TILE_M)]];
    [[assume(tile_n_size <= TILE_N)]];

    // accumulators remain in registers until
    // write back at the end
    float32x4_t acc[TILE_M][TILE_N] = {};
    for (size_t k = 0; k + 4 <= K; k += 4)
    {
        float32x4_t a_col[TILE_M], b_row[TILE_N];

        for (size_t i = 0; i < tile_m_size; ++i)
        {
            a_col[i] = vld1q_f32(a.data() + (i0 + i) * K + k);
        }

        for (size_t j = 0; j < tile_n_size; ++j)
        {
            b_row[j] = vld1q_f32(b_trans.data() + (j0 + j) * K + k);
        }

        for (size_t i = 0; i < tile_m_size; ++i)
        {
            for (size_t j = 0; j < tile_n_size; ++j)
            {
                acc[i][j] = vfmaq_f32(acc[i][j], a_col[i], b_row[j]);
            }
        }
    }

    if (K % 4)
    {
        size_t const k_start = K - K % 4;
        size_t const k_rem   = K % 4;
        float32x4_t  a_col[TILE_M], b_row[TILE_N];

        for (size_t i = 0; i < tile_m_size; ++i)
        {
            a_col[i] = load_partial(a.data() + (i0 + i) * K + k_start, k_rem);
        }

        for (size_t j = 0; j < tile_n_size; ++j)
        {
            b_row[j] =
                load_partial(b_trans.data() + (j0 + j) * K + k_start, k_rem);
        }

        for (size_t i = 0; i < tile_m_size; ++i)
        {
            for (size_t j = 0; j < tile_n_size; ++j)
            {
                acc[i][j] = vfmaq_f32(acc[i][j], a_col[i], b_row[j]);
            }
        }
    }

    for (size_t i = 0; i < tile_m_size; ++i)
    {
        size_t const offs = (i0 + i) * N;

        for (size_t j = 0; j < tile_n_size; ++j)
        {
            out[offs + j0 + j] = vaddvq_f32(acc[i][j]);
        }
    }
}
} // namespace

template <size_t R>
void matmul(TensorF32<R> a /* M x K */, TensorF32<R> b_trans /* N x K */,
            MutTensorF32<R> out /* M * N */)
{
    static_assert(R >= 2);

    constexpr size_t T_M = _CPU_MATMUL_TILE_M;
    constexpr size_t T_N = _CPU_MATMUL_TILE_N;

    size_t batches = 1;
    for (uint32_t i = 0; i < R - 2; ++i)
    {
        batches *= a.dims[i];
    }

    for (size_t batch = 0; batch < batches; ++batch)
    {
        size_t const M = a.dims[R - 2];
        size_t const K = a.dims[R - 1];
        size_t const N = b_trans.dims[R - 2];

        // batch offsets folded into the spans, so the tile kernel indexes from
        // zero and never needs to know about batching
        std::span<float32_t const> const a_batch{a.data + batch * M * K, M * K};
        std::span<float32_t const> const b_batch{b_trans.data + batch * N * K,
                                                 N * K};
        std::span<float32_t> const out_batch{out.data + batch * M * N, M * N};

        size_t const m_chunks = M / T_M;
        size_t const m_rem    = M % T_M;
        size_t const n_chunks = N / T_N;
        size_t const n_rem    = N % T_N;

        for (size_t tm = 0; tm < m_chunks; ++tm)
        {
            for (size_t tn = 0; tn < n_chunks; ++tn)
            {
                _matmul_at_chunk<T_M, T_N>(a_batch, b_batch, out_batch, K, N,
                                           tm, tn, T_M, T_N);
            }
            if (n_rem > 0)
            {
                _matmul_at_chunk<T_M, T_N>(a_batch, b_batch, out_batch, K, N,
                                           tm, n_chunks, T_M, n_rem);
            }
        }

        if (m_rem == 1)
        {
            size_t const i0 = m_chunks * T_M;

            for (size_t tn = 0; tn < n_chunks; ++tn)
            {
                _matmul_at_chunk<1, T_N>(a_batch, b_batch, out_batch, K, N, i0,
                                         tn, 1, T_N);
            }
            if (n_rem > 0)
            {
                _matmul_at_chunk<1, T_N>(a_batch, b_batch, out_batch, K, N, i0,
                                         n_chunks, 1, n_rem);
            }
        }
        else if (m_rem > 1)
        {
            for (size_t tn = 0; tn < n_chunks; ++tn)
            {
                _matmul_at_chunk<T_M, T_N>(a_batch, b_batch, out_batch, K, N,
                                           m_chunks, tn, m_rem, T_N);
            }
            if (n_rem > 0)
            {
                _matmul_at_chunk<T_M, T_N>(a_batch, b_batch, out_batch, K, N,
                                           m_chunks, n_chunks, m_rem, n_rem);
            }
        }
    }
}

// Explicit instantiation: the definition lives here, not in the header, so
// every rank a caller needs must be named.
template void matmul<2>(TensorF32<2>, TensorF32<2>, MutTensorF32<2>);
template void matmul<3>(TensorF32<3>, TensorF32<3>, MutTensorF32<3>);
} // namespace backend
