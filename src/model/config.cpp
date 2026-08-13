#include "config.h"

#include "json.hpp"

#include <expected>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iterator>
#include <string>
#include <string_view>
#include <type_traits>

namespace model {
namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

template <class T> bool holds(const json &v) {
  if constexpr (std::is_same_v<T, bool>)
    return v.is_boolean();
  else if constexpr (std::is_same_v<T, std::string>)
    return v.is_string();
  else if constexpr (std::is_unsigned_v<T>)
    return v.is_number_unsigned();
  else if constexpr (std::is_integral_v<T>)
    return v.is_number_integer();
  else
    return v.is_number();
}

template <class T>
std::expected<T, ConfigError> req(const json &j, std::string_view key) {
  const auto it = j.find(key);
  if (it == j.end() || it->is_null())
    return std::unexpected(
        ConfigError{"missing required key: " + std::string(key)});
  if (!holds<T>(*it))
    return std::unexpected(
        ConfigError{"wrong type for key: " + std::string(key)});
  return it->get<T>();
}

template <class T> T opt(const json &j, std::string_view key, T fallback) {
  const auto it = j.find(key);
  if (it == j.end() || it->is_null() || !holds<T>(*it))
    return fallback;
  return it->get<T>();
}

} // namespace

std::expected<Config, ConfigError> parse_config(std::string_view text) {
  const json j = json::parse(text, nullptr, /*allow_exceptions=*/false);

  if (j.is_discarded())
    return std::unexpected(ConfigError{"malformed JSON"});
  if (!j.is_object())
    return std::unexpected(ConfigError{"config root is not an object"});

  Config c{};

  const auto model_type = req<std::string>(j, "model_type");
  if (!model_type)
    return std::unexpected(model_type.error());
  if (*model_type == "qwen2")
    c.model_type = Type::Qwen2;
  else
    return std::unexpected(
        ConfigError{"unsupported model_type: " + *model_type});

  const auto architectures = j.find("architectures");
  if (architectures == j.end() || !architectures->is_array() ||
      architectures->empty() || !architectures->front().is_string())
    return std::unexpected(ConfigError{"missing or malformed architectures"});

  const auto architecture = architectures->front().get<std::string>();
  if (architecture == "Qwen2ForCausalLM")
    c.architecture = Architecture::Qwen2ForCausalLM;
  else
    return std::unexpected(
        ConfigError{"unsupported architecture: " + architecture});

  const auto hidden_act = req<std::string>(j, "hidden_act");
  if (!hidden_act)
    return std::unexpected(hidden_act.error());
  if (*hidden_act == "silu")
    c.hidden_act = Activation::SiLU;
  else
    return std::unexpected(
        ConfigError{"unsupported hidden_act: " + *hidden_act});

  struct U32Field {
    uint32_t *dst;
    const char *key;
  };
  const U32Field required[] = {
      {&c.hidden_sz, "hidden_size"},
      {&c.intermediate_size, "intermediate_size"},
      {&c.num_attn_heads, "num_attention_heads"},
      {&c.num_hidden_layers, "num_hidden_layers"},
      {&c.num_kv_heads, "num_key_value_heads"},
      {&c.vocab_size, "vocab_size"},
      {&c.max_position_embeddings, "max_position_embeddings"},
  };
  for (const auto &f : required) {
    const auto v = req<uint32_t>(j, f.key);
    if (!v)
      return std::unexpected(v.error());
    *f.dst = *v;
  }

  c.bos_token_id = opt<uint32_t>(j, "bos_token_id", 0);
  c.eos_token_id = opt<uint32_t>(j, "eos_token_id", 0);
  c.sliding_window = opt<uint32_t>(j, "sliding_window", 0);
  c.max_window_layers =
      opt<uint32_t>(j, "max_window_layers", c.num_hidden_layers);

  c.rms_norm_eps = opt<float32_t>(j, "rms_norm_eps", 1e-6f);
  c.attention_dropout = opt<float32_t>(j, "attention_dropout", 0.0f);
  c.initializer_range = opt<float32_t>(j, "initializer_range", 0.02f);

  const auto rope = j.find("rope_parameters");
  if (rope != j.end() && rope->is_object())
    c.rope_theta = opt<float32_t>(*rope, "rope_theta", 10000.0f);
  else
    c.rope_theta = opt<float32_t>(j, "rope_theta", 10000.0f);

  c.tie_word_embeddings = opt<bool>(j, "tie_word_embeddings", false);
  c.use_cache = opt<bool>(j, "use_cache", true);
  c.use_sliding_window = opt<bool>(j, "use_sliding_window", false);

  if (c.num_kv_heads == 0 || c.num_attn_heads % c.num_kv_heads != 0)
    return std::unexpected(ConfigError{
        "num_attention_heads is not divisible by num_key_value_heads"});
  if (c.num_attn_heads == 0 || c.hidden_sz % c.num_attn_heads != 0)
    return std::unexpected(
        ConfigError{"hidden_size is not divisible by num_attention_heads"});

  return c;
}

std::expected<Config, ConfigError>
load_config(const fs::path &dir) {
  const auto p = dir / "config.json";

  std::ifstream f(p, std::ios::binary);
  if (!f) {
    return std::unexpected(ConfigError{"cannot open " + p.string()});
  }

  std::string bytes{std::istreambuf_iterator<char>(f), {}};
  return parse_config(bytes);
}
} // namespace model
