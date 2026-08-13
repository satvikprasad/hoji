#include "args.h"
#include "core/npy_reader.h"
#include "model/config.h"

#include <expected>
#include <iostream>

int main(int argc, char **argv) {
  try {
    // parse arguments
    const args::Args args = args::parse(argc, argv).value();

    // load model config.json
    const model::Config conf = model::load_config(args.model).value();

    const auto attn_in = core::npy_read_shape("ref/L0_attn_in.npy").value();
    std::cout << "L0_attn_in rank=" << attn_in.shape.size << " elems="
              << attn_in.data.size() << " first=" << attn_in.data[0] << std::endl;
  } catch (const std::bad_expected_access<std::string> &ex) {
    std::cerr << "ERROR: " << ex.error() << std::endl;
    return 1;
  }

  return 0;
}
