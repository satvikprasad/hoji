#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <iterator>
#include <string>
#include <string_view>

namespace args {
namespace fs = std::filesystem;
struct Args {
  // No defaults: --model and --prompt are required. Presence is checked via
  // the seen[] table in assert_defaults, not by inspecting these fields.
  fs::path model;
  std::string prompt;

  uint32_t max_tokens = 20;
  uint32_t page_size = 16;

  bool greedy = false;
};

enum class Kind { Flag, U32, Str, Path };

union Target {
  bool Args::*flag;
  uint32_t Args::*u32;
  std::string Args::*str;
  std::filesystem::path Args::*path;
};

// --help is handled directly in parse(), so it has no Arg / kSpecs entry.
enum Arg { Model = 0, MaxTokens = 1, PageSz, Prompt, Greedy, COUNT };

struct Spec {
  std::string_view name;
  char alias;
  Kind kind;
  Target target;
  std::string_view help;
  Arg arg_type;
};

// specifications for argument list
constexpr Spec kSpecs[] = {
    {"model",
     'm',
     Kind::Path,
     {.path = &Args::model},
     "model directory",
     Arg::Model},
    {"max-tokens",
     'n',
     Kind::U32,
     {.u32 = &Args::max_tokens},
     "tokens to generate",
     Arg::MaxTokens},
    {"page-size",
     0,
     Kind::U32,
     {.u32 = &Args::page_size},
     "KV page size in tokens",
     Arg::PageSz},
    {"prompt",
     0,
     Kind::Str,
     {.str = &Args::prompt},
     "input prompt",
     Arg::Prompt},
    {"greedy", 0, Kind::Flag, {.flag = &Args::greedy}, "greedy", Arg::Greedy},
};

// kSpecs is indexed by Arg, so row i must carry arg_type == i. Both asserts
// fire at compile time if a row is added, removed, or moved out of order.
static_assert(std::size(kSpecs) == static_cast<size_t>(Arg::COUNT),
              "kSpecs and Arg disagree on the number of arguments");

constexpr bool specs_match_enum() {
  for (size_t i = 0; i < std::size(kSpecs); ++i)
    if (static_cast<size_t>(kSpecs[i].arg_type) != i)
      return false;
  return true;
}
static_assert(specs_match_enum(), "kSpecs order must match the Arg enum");

constexpr const Spec &spec_of(Arg a) { return kSpecs[static_cast<size_t>(a)]; }

std::expected<Args, std::string> parse(size_t argc, char **argv);
} // namespace args
