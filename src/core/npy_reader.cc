#include "npy_reader.h"

#include <cstddef>
#include <expected>
#include <filesystem>
#include <string>
#include <string_view>

#include <cstring>
#include <sys/fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace core
{

inline std::expected<Shape, std::string> decode_shape(std::string_view header)
{
    const size_t shape_pos = header.find("'shape':");
    if (shape_pos == std::string_view::npos)
    {
        return std::unexpected("no 'shape' key in npy header");
    }

    const size_t lparen = header.find('(', shape_pos);
    const size_t rparen = header.find(')', lparen);
    if (lparen == std::string_view::npos || rparen == std::string_view::npos)
    {
        return std::unexpected("malformed 'shape' tuple in npy header");
    }

    Shape shape{};
    for (size_t i = lparen + 1; i < rparen;)
    {
        if (header[i] == ' ' || header[i] == ',')
        {
            ++i;
            continue;
        }
        if (header[i] < '0' || header[i] > '9')
        {
            return std::unexpected("unexpected character in 'shape' tuple");
        }

        size_t dim = 0;
        while (i < rparen && header[i] >= '0' && header[i] <= '9')
        {
            dim = dim * 10 + static_cast<size_t>(header[i] - '0');
            ++i;
        }

        if (!shape.push_back(dim))
        {
            return std::unexpected("npy rank exceeds " +
                                   std::to_string(MAX_RANK));
        }
    }

    return shape;
}

inline std::expected<void, std::string> check_layout(std::string_view header)
{
    if (header.find("'descr': '<f4'") == std::string_view::npos)
    {
        return std::unexpected(
            "npy dtype is not little-endian float32 ('<f4')");
    }
    if (header.find("'fortran_order': False") == std::string_view::npos)
    {
        return std::unexpected(
            "npy is not C-ordered ('fortran_order' must be False)");
    }
    return {};
}

inline size_t elem_count(const Shape &shape)
{
    size_t n = 1;
    for (size_t d = 0; d < shape.size; ++d)
    {
        n *= shape.elems[d];
    }
    return n;
}

Npy::Npy(Npy &&other) noexcept
    : shape(other.shape), data(other.data), base(other.base), bytes(other.bytes)
{
    other.base  = nullptr;
    other.bytes = 0;
    other.data  = {};
}

Npy &Npy::operator=(Npy &&other) noexcept
{
    if (this != &other)
    {
        unmap();
        shape       = other.shape;
        data        = other.data;
        base        = other.base;
        bytes       = other.bytes;
        other.base  = nullptr;
        other.bytes = 0;
        other.data  = {};
    }
    return *this;
}

Npy::~Npy()
{
    unmap();
}

void Npy::unmap() noexcept
{
    if (base != nullptr)
    {
        ::munmap(base, bytes);
        base  = nullptr;
        bytes = 0;
        data  = {};
    }
}

std::expected<Npy, std::string>
npy_read_shape(const std::filesystem::path &npy_path)
{
    constexpr size_t MAGIC_LEN         = 6;
    constexpr size_t HEADER_LEN_OFFSET = 8;
    constexpr size_t PREAMBLE_LEN      = HEADER_LEN_OFFSET + 2;

    const int fd = ::open(npy_path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd == -1)
    {
        return std::unexpected("failed to open " + std::string(npy_path));
    }

    struct stat sb{};
    if (::fstat(fd, &sb) == -1)
    {
        ::close(fd);
        return std::unexpected("failed to stat " + std::string(npy_path));
    }
    if (!S_ISREG(sb.st_mode) || static_cast<size_t>(sb.st_size) < PREAMBLE_LEN)
    {
        ::close(fd);
        return std::unexpected("not a regular npy file: " +
                               std::string(npy_path));
    }

    void *mapped = ::mmap(nullptr, static_cast<size_t>(sb.st_size), PROT_READ,
                          MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (mapped == MAP_FAILED)
    {
        return std::unexpected("mmap failed for " + std::string(npy_path));
    }

    // out owns the mapping from here, so every return below unmaps via ~Npy.
    Npy out;
    out.base  = static_cast<std::byte *>(mapped);
    out.bytes = static_cast<size_t>(sb.st_size);

    const std::string_view magic(reinterpret_cast<const char *>(out.base),
                                 MAGIC_LEN);
    if (magic != "\x93NUMPY")
    {
        return std::unexpected("not an npy file: " + std::string(npy_path));
    }

    uint16_t header_len = 0;
    std::memcpy(&header_len, out.base + HEADER_LEN_OFFSET, sizeof(header_len));
    const size_t data_offset = PREAMBLE_LEN + header_len;
    if (data_offset > out.bytes)
    {
        return std::unexpected("npy header runs past the end of " +
                               std::string(npy_path));
    }

    const std::string_view header(
        reinterpret_cast<const char *>(out.base + PREAMBLE_LEN), header_len);

    if (const auto layout = check_layout(header); !layout)
    {
        return std::unexpected(layout.error() + " of " + std::string(npy_path));
    }

    const auto shape = decode_shape(header);
    if (!shape)
    {
        return std::unexpected(shape.error() + " of " + std::string(npy_path));
    }

    const size_t count = elem_count(*shape);
    if (data_offset % alignof(float) != 0)
    {
        return std::unexpected("npy data is misaligned in " +
                               std::string(npy_path));
    }
    if (data_offset + count * sizeof(float) > out.bytes)
    {
        return std::unexpected("npy data is truncated in " +
                               std::string(npy_path));
    }

    out.shape = *shape;
    out.data = {reinterpret_cast<const float *>(out.base + data_offset), count};
    return out;
}
}; // namespace core
