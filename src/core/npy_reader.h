#pragma once

#include "backend/tensor_desc.h"
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
using Shape = FixedVector<size_t, backend::TensorDesc::MAX_RANK>;

struct Npy
{
    Shape                  shape{};
    std::span<float const> data{};

    std::byte *base  = nullptr;
    size_t     bytes = 0;

    Npy()                       = default;
    Npy(Npy const &)            = delete;
    Npy &operator=(Npy const &) = delete;

    Npy(Npy &&other) noexcept;
    Npy &operator=(Npy &&other) noexcept;
    ~Npy();

    void unmap() noexcept;
};

template <size_t R> StaticView<float32_t const, R> view(Npy const &n)
{
    StaticView<float32_t const, R> view{.data = n.data.data()};
    for (size_t i = 0; i < R; ++i)
    {
        view.dims[i] = n.shape[i];
    }

    return view;
}

// mmaps npy_path and returns its shape plus a float32 view of the data. The
// data is not copied; it is the mapped file.
std::expected<Npy, std::string>
npy_read_shape(std::filesystem::path const &npy_path);
} // namespace core
