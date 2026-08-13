#pragma once
#include <cstdint>

using bfloat16_t = __bf16;
using float16_t = _Float16;
using float32_t = float;
using float64_t = double;

static_assert(sizeof(bfloat16_t) == 2 && sizeof(float16_t) == 2);
