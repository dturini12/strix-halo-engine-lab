#pragma once

#include "gemma4/contracts/g4_types.hpp"
#include "strix/model_description.hpp"

namespace strix::gemma4 {

// Export-only typed template. Existing loaders, tensor-name mappings, activation
// formulas, tokenizer handling, KV transactions and compiled schedules remain the
// explicit adapter implementation. This object is never used by a run path or
// as a source of printed identity text.
struct DescriptionTemplate {
    ModelConfig config;
    std::vector<std::uint32_t> taps;
    std::uint32_t attention_chunk_rows = 0;
    float embedding_multiplier = 0, ple_embedding_multiplier = 0;
    float ple_model_multiplier = 0, ple_combine_multiplier = 0;
};

[[nodiscard]] std::string export_model_description();
// Aggregates every missing, unknown, wrongly typed or unequal field with its
// expected and actual value; throws std::runtime_error on any mismatch.
[[nodiscard]] DescriptionTemplate validate_model_description(const model_description::ModelDescription& description);

} // namespace strix::gemma4
