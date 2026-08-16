#include <metal_stdlib>

#include "backend/tensor_desc.h"

using namespace metal;
using backend::TensorDesc;

template <uint T_M, uint T_N>
kernel void matmul(device const float *A [[buffer(0)]],
                   device const float *B [[buffer(1)]],
                   device float       *C [[buffer(2)]],

                   constant TensorDesc &a_desc [[buffer(3)]],
                   constant TensorDesc &out_desc [[buffer(4)]],

                   uint2 tgid [[threadgroup_position_in_grid]],
                   uint2 tptg [[threads_per_threadgroup]],
                   uint2 lid [[thread_position_in_threadgroup]])
{
    uint const K = a_desc.shape[a_desc.rank - 1];
    uint const M = out_desc.shape[out_desc.rank - 2];
    uint const N = out_desc.shape[out_desc.rank - 1];

    // thread position in grid, then the corner of its register tile
    uint2 const gid = tgid * tptg + lid;

    uint const row = gid.y * T_M;
    uint const col = gid.x * T_N;
    uint       a_base[T_M], b_base[T_N];

    for (uint m = 0; m < T_M; ++m)
    {
        a_base[m] = min(row + m, M - 1) * K;
    }
    for (uint n = 0; n < T_N; ++n)
    {
        b_base[n] = min(col + n, N - 1) * K;
    }

    float acc[T_M][T_N] = {};
    for (uint k = 0; k < K; ++k)
    {
        float a[T_M], b[T_N];
        for (uint m = 0; m < T_M; ++m)
        {
            a[m] = A[a_base[m] + k];
        }
        for (uint n = 0; n < T_N; ++n)
        {
            b[n] = B[b_base[n] + k];
        }

        for (uint m = 0; m < T_M; ++m)
        {
            for (uint n = 0; n < T_N; ++n)
            {
                acc[m][n] = fma(a[m], b[n], acc[m][n]);
            }
        }
    }

    for (uint m = 0; m < T_M; ++m)
    {
        for (uint n = 0; n < T_N; ++n)
        {
            if (row + m < M && col + n < N)
            {
                C[(row + m) * N + col + n] = acc[m][n];
            }
        }
    }
}

#define HOJI_MATMUL_TILE(T_M, T_N)                                             \
    template [[host_name("matmul_" #T_M "x" #T_N)]]                            \
    kernel decltype(matmul<T_M, T_N>) matmul<T_M, T_N>;

HOJI_MATMUL_TILE(1, 1)
HOJI_MATMUL_TILE(1, 4)
HOJI_MATMUL_TILE(2, 2)
HOJI_MATMUL_TILE(4, 4)
HOJI_MATMUL_TILE(4, 8)
HOJI_MATMUL_TILE(8, 4)

#undef HOJI_MATMUL_TILE
