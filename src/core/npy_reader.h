#pragma once

#include "core/static_view.h"
#include "dtype.h"
#include "fixed_vector.h"

#include <cassert>
#include <cstddef>
#include <expected>
#include <filesystem>
#include <span>
#include <string>

namespace core
{
// Rank never exceeds 4 for transformer tensors (batch, heads, seq, dim);
// 8 leaves headroom while keeping Shape small enough not to care about.
constexpr size_t MAX_RANK = 8;
using Shape               = FixedVector<size_t, MAX_RANK>;

struct Npy
{
    Shape                  shape{};
    std::span<const float> data{};

    std::byte *base  = nullptr;
    size_t     bytes = 0;

    Npy()                       = default;
    Npy(const Npy &)            = delete;
    Npy &operator=(const Npy &) = delete;

    Npy(Npy &&other) noexcept;
    Npy &operator=(Npy &&other) noexcept;
    ~Npy();

    void unmap() noexcept;
};

template <size_t R> StaticView<const float32_t, R> view(const Npy &n)
{
    StaticView<const float32_t, R> view{.data = n.data.data()};
    for (size_t i = 0; i < R; ++i)
    {
        view.dims[i] = n.shape[i];
    }

    return view;
}

// mmaps npy_path and returns its shape plus a float32 view of the data. The
// data is not copied; it is the mapped file.
std::expected<Npy, std::string>
npy_read_shape(const std::filesystem::path &npy_path);
} // namespace core
