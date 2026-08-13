#pragma once

#include <cstddef>

namespace core {
template <typename T, std::size_t N> struct FixedVector {
  T elems[N];
  size_t size;

  const size_t cap{N};
};
} // namespace core
