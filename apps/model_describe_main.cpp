#include "strix/deepseek41_model_description.hpp"

#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) try {
    if (argc != 5 || (std::string(argv[1]) != "--config-json" && std::string(argv[1]) != "--gguf") ||
        std::string(argv[3]) != "--output")
        throw std::invalid_argument("usage: strix-deepseek41-model-describe --config-json CONFIG.json|--gguf MODEL.gguf --output MODEL.toml");
    strix::deepseek41::Config config;
    if (std::string(argv[1]) == "--gguf") {
        config = strix::deepseek41::validate_target(strix::deepseek41::read_file(argv[2])).config;
    } else {
        std::ifstream in(argv[2], std::ios::binary);
        if (!in) throw std::invalid_argument("cannot read config JSON");
        const std::string json{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
        if (in.bad()) throw std::invalid_argument("cannot finish reading config JSON");
        config = strix::deepseek41::parse_config_json(json);
    }
    const auto toml = strix::deepseek41::describe_model_toml(config);
    // Validate the exact emitted subset before creating the output file.
    (void)strix::deepseek41::config_from_model_description(strix::model_description::parse(toml), config);
    std::ofstream out(argv[4], std::ios::binary);
    if (!out || !out.write(toml.data(), static_cast<std::streamsize>(toml.size())))
        throw std::invalid_argument("cannot write model TOML");
    out.close();
    if (!out) throw std::invalid_argument("cannot finish writing model TOML");
    return 0;
} catch (const std::exception& error) {
    std::cerr << "strix-deepseek41-model-describe: " << error.what() << '\n';
    return 1;
}
