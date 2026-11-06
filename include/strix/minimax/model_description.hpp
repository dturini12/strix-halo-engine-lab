#pragma once
#include "strix/model_description.hpp"
#include <cstdint>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace strix::minimax::description {
// Explicit adapter for the compiled MiniMax schedule. Validation only: the
// description never selects arithmetic, defaults, schedules or identity text.
void validate(const model_description::ModelDescription& model);
model_description::ModelDescription load(const std::string& path);
struct HeaderTensor {
    std::string name, dtype;
    std::vector<std::uint64_t> shape;
    std::uint64_t begin = 0, end = 0;
};
// Strict safetensors JSON and dimensions of recognised compiled tensor roles.
// Partial/draft-only packs remain supported; missing tensors still fail at info().
// semantic=false keeps the structural checks (JSON, offsets, overflow) and skips the role/dimension assertions.
std::vector<HeaderTensor> header(std::string_view json, bool semantic = true);
// STRIX_MM_MODEL_DESCRIPTION_CHECK=0 restores the previous admission: the sidecar description is neither opened nor
// interpreted and the role/dimension assertions are skipped. Not part of any numerical identity.
bool check_enabled();
// Check before allocating or reading a header. The returned size excludes both
// the length word and JSON; all offsets in the header are relative to it.
std::uint64_t header_payload_bytes(std::uint64_t header_bytes, std::uint64_t file_bytes);
// One catalog per PackReader, shared by all of its shards (including drafter
// packs). These guards only reject malformed input; they never select a route.
class PackHeaders {
public:
    explicit PackHeaders(bool semantic = true) : semantic_(semantic) {}
    std::vector<HeaderTensor> add(std::string_view json, std::uint64_t payload_bytes);
private:
    bool semantic_;
    std::set<std::string> names_;
};
} // namespace strix::minimax::description
