#include "strix/model_registry.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {
int checks = 0, failures = 0;
void expect(bool ok, const std::string& message) {
    ++checks;
    if (!ok) { ++failures; std::cerr << "FAIL: " << message << '\n'; }
}
namespace reg = strix::model_registry;
namespace md = strix::model_description;
bool refused(const char* text) {
    try { (void)reg::parse(md::parse(text)); }
    catch (const std::invalid_argument&) { return true; }
    return false;
}
} // namespace

int main(int argc, char** argv) try {
    if (argc != 2) throw std::invalid_argument("usage: model_registry_tests REPO_ROOT");
    const std::string root = argv[1];
    const auto entries = reg::load(root + "/models/registry.toml");
    expect(entries.size() >= 5, "five engines registered");
    for (const auto& problem : reg::check(entries, root)) expect(false, problem);

    // Negative cases: a wrong schema, a missing key and a name that differs from its description.
    expect(refused("schema = \"strix-registry/v2\"\n[models.a]\nname = \"a\"\n"), "wrong schema refused");
    expect(refused("schema = \"strix-registry/v1\"\n[models.a]\nname = \"a\"\n"), "missing description refused");
    const auto bad = reg::parse(md::parse("schema = \"strix-registry/v1\"\n[models.a]\nname = \"zzz\"\n"
        "description = \"models/qwen3.8-flash-next.toml\"\nfamily = \"qwen\"\n"));
    expect(!reg::check(bad, root).empty(), "mismatching name and empty tables reported");
    std::cout << checks << " checks, " << failures << " failures\n";
    return failures == 0 ? 0 : 1;
} catch (const std::exception& error) {
    std::cerr << "model_registry_tests: " << error.what() << '\n';
    return 1;
}
