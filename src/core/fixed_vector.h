#pragma once

#include <cstddef>

namespace core {
template <typename T, std::size_t N> struct FixedVector {
  T elems[N];
  size_t size = 0;

  // static, not a const member: a const member deletes copy-assignment, which
  // blocks `a = b` on any struct holding a FixedVector. Costs no storage.
  static constexpr size_t cap = N;

  constexpr bool push_back(const T &value) {
    if (size >= N) {
      return false;
    }
    elems[size++] = value;
    return true;
  }
};
} // namespace core
