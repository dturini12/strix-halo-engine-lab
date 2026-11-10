#pragma once
// DeepSeek V4.1 vision tower on the device (HIP): the ViT + aligner of ds41_vision.hpp with the same declared
// arithmetic as the host reference, bit for bit (docs/models/deepseek-v4.1/vision.md, "Device tower"). The weights
// stay resident; work buffers are sized for the largest picture the configuration admits.
#include "strix/ds41_vision.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace strix::deepseek41::vision {

// Largest patch grid any picture can produce under the configuration's token budget (9 per aligner cell).
[[nodiscard]] std::uint32_t max_patches(const VisionConfig& config) noexcept;
[[nodiscard]] std::uint32_t max_cells(const VisionConfig& config) noexcept;

class DeviceTower {
public:
    // Uploads the weights to the current device and allocates the work buffers; `stream` is a hipStream_t.
    DeviceTower(const VisionWeights& weights, void* stream);
    ~DeviceTower();
    DeviceTower(const DeviceTower&) = delete;
    DeviceTower& operator=(const DeviceTower&) = delete;

    [[nodiscard]] std::size_t weight_bytes() const noexcept;
    [[nodiscard]] std::size_t work_bytes() const noexcept;
    [[nodiscard]] const VisionConfig& config() const noexcept;

    // Encodes one picture into its span rows [span_tokens][out_dim] (BF16) at `device_span_rows` (a device pointer
    // with room for span_tokens * out_dim values). Asynchronous on the stream. observer (tests): each step's BF16
    // output, downloaded synchronously ("embed", "block" per layer, "norm", "rows").
    void encode(const PreparedImage& image, std::uint16_t* device_span_rows, const StepObserver& observer = {});
    // encode() then copy to the host (synchronous).
    [[nodiscard]] std::vector<std::uint16_t> encode_to_host(const PreparedImage& image, const StepObserver& observer = {});

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strix::deepseek41::vision
