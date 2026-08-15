#pragma once

#include <cstddef>
#include <cstdint>

namespace core
{
template <typename T, size_t R> struct StaticView
{
    static_assert(R >= 1);

    T     *data{nullptr};
    size_t dims[R]{};

    constexpr T &operator[](size_t idx)
    {
        return data[idx];
    }

    constexpr T &operator[](size_t idx) const
    {
        return data[idx];
    }

    template <class... I> constexpr T &operator[](I... idx) const
    {
        static_assert(sizeof...(I) == R, "index count must equal rank");

        size_t const ix[R]{static_cast<size_t>(idx)...};

        size_t   i           = 0;
        uint32_t curr_stride = 1;

        for (size_t j = R - 1; j >= 0; --j)
        {
            i += ix[j] * curr_stride;
            curr_stride *= dims[j];
        }

        return data[i];
    };

    size_t size() const
    {
        size_t sz = 1;
        for (uint32_t i = 0; i < R; ++i)
        {
            sz *= dims[i];
        }
        return sz;
    };
};
} // namespace core
