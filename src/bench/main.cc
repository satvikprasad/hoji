#include "backend/backend.h"
#include "backend/metal.h"
#include "backend/tensor_desc.h"
#include <expected>
#include <iostream>

void bench_metal()
{
    using namespace backend;

    try
    {
        metal::Backend const      mtl   = metal::Backend::load().value();
        Allocator<metal::Backend> alloc = mtl.get_allocator();

        /* a is an identity mat */
        auto const a =
            alloc.push_tensor(TensorDesc({64, 64}), DataBufferType::Wgt);
        float32_t *a_data = alloc.view_tensor_mut(a).data();
        for (size_t i = 0; i < 64; ++i)
        {
            for (size_t j = 0; j < 64; ++j)
            {
                a_data[i * 64 + j] = static_cast<float32_t>(i == j);
            }
        }

        /* b[i, j] = 5 */
        auto const b_trans =
            alloc.push_tensor(TensorDesc({128, 64}), DataBufferType::Actvt);
        auto b_data = alloc.view_tensor_mut(b_trans);
        for (size_t i = 0; i < 128; ++i)
        {
            for (size_t j = 0; j < 64; ++j)
            {
                b_data[i * 64 + j] = 5.f;
            }
        }

        auto const out =
            alloc.push_tensor(TensorDesc({64, 128}), DataBufferType::Actvt);
        mtl.matmul<4, 4>(a, b_trans, out);

        auto out_data = alloc.view_tensor(out);
        for (float32_t const f : out_data)
        {
            std::cout << f << std::endl;
        }
    }
    catch (std::bad_expected_access<std::string> const &ex)
    {
        std::cerr << "ERROR: " << ex.error() << std::endl;
    }
}

int main()
{
    std::cout << "INFO: benching metal\n=========================="
              << std::endl;
    bench_metal();

    return 0;
}
