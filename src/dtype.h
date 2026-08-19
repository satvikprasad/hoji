#pragma once

#include <mdspan>
#include <type_traits>

using bfloat16_t = __bf16;
using float16_t  = __fp16;
using float32_t  = float;
using float64_t  = double;

static_assert(sizeof(bfloat16_t) == 2 && sizeof(float16_t) == 2);

namespace util
{
template <typename T> constexpr size_t enum_index(T e)
{
    return static_cast<size_t>(e);
}

template <typename T> constexpr size_t enum_size()
{
    return enum_index(T::COUNT);
}

template <typename T>
    requires std::is_integral_v<T>
T div_ceil(T a, T b)
{
    return (a + b - 1) / b;
}
} // namespace util
