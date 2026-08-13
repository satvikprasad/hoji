#pragma once

#include "dtype.h"

#include <cstdint>
#include <expected>
#include <filesystem>

namespace model {
enum class Type { Qwen2 };
enum class Architecture { Qwen2ForCausalLM };
enum class Activation { SiLU };

struct Config {
  Architecture architecture;
  Type model_type;
  Activation hidden_act;

  bool tie_word_embeddings, use_cache, use_sliding_window;

  float32_t attention_dropout, initializer_range, rms_norm_eps, rope_theta;

  uint32_t bos_token_id, eos_token_id, hidden_sz, intermediate_size,
      max_position_embeddings, max_window_layers, num_attn_heads,
      num_hidden_layers, num_kv_heads, sliding_window, vocab_size;
};

struct ConfigError {
  std::string why;
};

std::expected<Config, ConfigError>
load_config(const std::filesystem::path &dir);
} // namespace model
