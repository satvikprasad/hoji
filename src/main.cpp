#include "args.h"
#include "model/config.h"

#include <expected>
#include <iostream>

int main(int argc, char **argv) {
  try {
    // parse arguments
    const args::Args args = args::parse(argc, argv).value();

    // load model config.json
    const model::Config conf = model::load_config(args.model).value();

  } catch (const std::bad_expected_access<model::ConfigError> &ex) {
    std::cout << "ERROR (config): " << ex.error().why << std::endl;
  } catch (const std::bad_expected_access<args::ArgsError> &ex) {
    std::cout << "ERROR (arguments): " << ex.error().why << std::endl;
  }

  return 0;
}
