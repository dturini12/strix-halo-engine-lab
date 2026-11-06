#pragma once

#include "strix/qwen_model_description.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

namespace strix::qwen_description {

inline void refuse_unless_described(const strix::ModelContract& baked, const strix::gguf::FileInfo& info) {
    if (const char* check = std::getenv("STRIX_QWEN_MODEL_DESCRIPTION_CHECK"); check && std::string_view(check) == "0")
        return;
    try {
        // The returned keys describe coverage; mismatches are reported by exception.
        (void)assert_gguf_matches(baked, info);
        if (const char* path = std::getenv("STRIX_QWEN_MODEL_DESCRIPTION"))
            assert_matches(baked, model_description::load(path));
    } catch (const std::exception& error) {
        std::string mismatches = error.what();
        for (const std::string_view prefix : {"Qwen GGUF metadata refused:", "Qwen model description refused:"}) {
            if (mismatches.starts_with(prefix)) {
                mismatches.erase(0, prefix.size());
                break;
            }
        }
        if (mismatches.starts_with("\n  ")) mismatches.erase(0, 3);
        for (auto pos = mismatches.find("\n  "); pos != std::string::npos; pos = mismatches.find("\n  ", pos + 2))
            mismatches.replace(pos, 3, "; ");
        throw std::runtime_error("model description mismatch: " + mismatches);
    }
}

} // namespace strix::qwen_description
