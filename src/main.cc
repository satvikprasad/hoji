#include "args.h"
#include "backend/metal.h"
#include "model/config.h"

#include <expected>
#include <iostream>

int main(int argc, char **argv)
{
    try
    {
        // parse arguments
        args::Args const    args = args::parse(argc, argv).value();
        model::Config const conf = model::load_config(args.model).value();
        (void)conf;

        auto const b = backend::metal::Backend::load().value();
    }
    catch (std::bad_expected_access<std::string> const &ex)
    {
        std::cerr << "ERROR: " << ex.error() << std::endl;
        return 1;
    }

    return 0;
}
