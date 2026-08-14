#include "args.h"

#include <charconv>
#include <cstdint>
#include <cstdio>
#include <expected>
#include <string>
#include <string_view>
#include <utility>

namespace args {

// flags with no default in Args; every one must appear on the command line.
constexpr Arg kRequired[] = {Arg::Model, Arg::Prompt};

// flags that do have a default in Args.
constexpr Arg kOptional[] = {Arg::MaxTokens, Arg::PageSz, Arg::Greedy};

// --help has no Args field and no kSpecs row, so its spelling lives here.
static bool is_help_flag(const std::string_view arg) {
  return arg == "--help" || arg == "-h";
}

static void emit_help() {
  std::fputs("usage: hoji [options]\n", stdout);
  for (const Spec &spec : kSpecs) {
    std::string line = "  --" + std::string(spec.name);
    if (spec.alias)
      line += ", -" + std::string(1, spec.alias);
    if (spec.kind != Kind::Flag)
      line += " <value>";
    line += "\n        " + std::string(spec.help) + "\n";
    std::fputs(line.c_str(), stdout);
  }
  std::fputs("  --help, -h\n        emit help string\n", stdout);
}

constexpr bool args_partitioned() {
  bool covered[Arg::COUNT] = {};
  for (const Arg a : kRequired) {
    if (a >= Arg::COUNT || covered[a])
      return false;
    covered[a] = true;
  }
  for (const Arg a : kOptional) {
    if (a >= Arg::COUNT || covered[a])
      return false;
    covered[a] = true;
  }
  for (const bool c : covered)
    if (!c)
      return false;
  return true;
}

static_assert(std::size(kRequired) + std::size(kOptional) ==
                  static_cast<size_t>(Arg::COUNT),
              "every Arg must appear in exactly one of kRequired / kOptional");

static_assert(args_partitioned(),
              "kRequired and kOptional must cover Arg with no gaps or repeats");

inline std::expected<void, std::string>
assert_defaults(const bool (&seen)[Arg::COUNT]) {
  for (const Arg required : kRequired) {
    if (!seen[required])
      return std::unexpected("missing required flag: --" +
                                       std::string(spec_of(required).name));
  }
  return {};
}

static std::expected<void, std::string> decode_arg(Args &a, const Spec &spec,
                                                 const std::string_view arg) {
  switch (spec.kind) {
  case Kind::Str:
    a.*spec.target.str = std::string(arg);
    return {};

  case Kind::Path:
    a.*spec.target.path = std::string(arg);
    return {};

  case Kind::U32: {
    uint32_t value = 0;
    const char *const end = arg.data() + arg.size();
    const auto r = std::from_chars(arg.data(), end, value);
    if (r.ec == std::errc::result_out_of_range)
      return std::unexpected(
          "--" + std::string(spec.name) +
                    " does not fit in 32 bits: " + std::string(arg));
    if (r.ec != std::errc{} || r.ptr != end)
      return std::unexpected("--" + std::string(spec.name) +
                                       " expects a number, got " +
                                       std::string(arg));
    a.*spec.target.u32 = value;
    return {};
  }

  case Kind::Flag:
    return std::unexpected(
        "--" + std::string(spec.name) + " takes no value");
  }

  std::unreachable();
}

std::expected<Args, std::string> parse(size_t argc, char **argv) {
  Args a{};

  constexpr Arg WAITING_ON_FLAG = Arg::COUNT;

  bool seen[Arg::COUNT] = {false};
  Arg curr{WAITING_ON_FLAG};

  for (size_t i = 1; i < argc; ++i) {
    std::string_view arg(argv[i]);

    if (is_help_flag(arg)) {
      emit_help();
      return std::unexpected("help requested");
    }

    if (!arg.starts_with("-")) {
      if (curr == WAITING_ON_FLAG) {
        return std::unexpected("encountered arg " + std::string(arg) +
                                         " when waiting on a flag.");
      }

      const Spec &spec = kSpecs[curr];
      if (const auto r = decode_arg(a, spec, arg); !r)
        return std::unexpected(r.error());

      curr = WAITING_ON_FLAG;
      continue;
    }

    bool matched = false;
    for (const Spec &spec : kSpecs) {
      if (arg.starts_with("--")) {
        matched = arg.substr(2) == spec.name;
      } else {
        matched = spec.alias && arg[0] == spec.alias;
      }

      if (!matched) {
        continue;
      }

      if (seen[spec.arg_type] == true) {
        return std::unexpected(
            "duplicate argument " + std::string(spec.name));
      }

      seen[spec.arg_type] = true;

      if (spec.kind == Kind::Flag) {
        a.*(spec.target.flag) = true;
      } else {
        curr = spec.arg_type;
      }

      break;
    }

    if (!matched)
      return std::unexpected(
          "unrecognized flag " + std::string(arg));
  }

  if (curr != WAITING_ON_FLAG) {
    return std::unexpected(
        "trailing flag " + std::string(kSpecs[curr].name));
  }

  // --help already returned above, so required flags always apply here.
  if (const auto r = assert_defaults(seen); !r)
    return std::unexpected(r.error());

  return a;
}
}; // namespace args
