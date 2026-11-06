#include "strix/model_registry.hpp"

#include <filesystem>
#include <set>
#include <stdexcept>

namespace strix::model_registry {
namespace {
namespace md = model_description;
const md::Value::Table& table_of(const md::Value& value, const std::string& what) {
    const auto* table = std::get_if<md::Value::Table>(&value.data);
    if (!table) throw std::invalid_argument(what + " must be a table");
    return *table;
}
std::string text(const md::Value::Table& t, const char* key, const std::string& id, bool required) {
    const auto it = t.find(key);
    if (it == t.end()) {
        if (required) throw std::invalid_argument("model " + id + ": missing " + key);
        return {};
    }
    const auto* s = std::get_if<std::string>(&it->second.data);
    if (!s) throw std::invalid_argument("model " + id + ": " + key + " must be a string");
    return *s;
}
std::vector<std::string> list(const md::Value::Table& t, const char* key, const std::string& id) {
    std::vector<std::string> out;
    const auto it = t.find(key);
    if (it == t.end()) return out;
    const auto* array = std::get_if<md::Value::Array>(&it->second.data);
    if (!array) throw std::invalid_argument("model " + id + ": " + key + " must be an array");
    for (const auto& item : *array) {
        const auto* s = std::get_if<std::string>(&item.data);
        if (!s) throw std::invalid_argument("model " + id + ": " + key + " holds a non-string");
        out.push_back(*s);
    }
    return out;
}
} // namespace

std::vector<Entry> parse(const md::ModelDescription& file) {
    const md::Value* schema = file.find("schema");
    const auto* sv = schema ? std::get_if<std::string>(&schema->data) : nullptr;
    if (!sv || *sv != "strix-registry/v1") throw std::invalid_argument("registry: schema must be strix-registry/v1");
    const md::Value* models = file.find("models");
    if (!models) throw std::invalid_argument("registry: no [models.*] tables");
    std::vector<Entry> entries;
    const std::string models_where = "models";
    for (const auto& [id, value] : table_of(*models, models_where)) {
        const std::string where = "model " + id;
        const auto& t = table_of(value, where);
        Entry e;
        e.id = id;
        e.name = text(t, "name", id, true);
        e.description = text(t, "description", id, true);
        e.family = text(t, "family", id, true);
        e.cmake = text(t, "cmake", id, false);
        e.profile_tag = text(t, "profile_tag", id, false);
        e.allow_pending_env = text(t, "allow_pending_env", id, false);
        e.defaults = list(t, "defaults", id);
        e.apps = list(t, "apps", id);
        e.adapter = list(t, "adapter", id);
        entries.push_back(std::move(e));
    }
    return entries;
}

std::vector<Entry> load(const std::string& path) { return parse(md::load(path)); }

std::vector<std::string> check(const std::vector<Entry>& entries, const std::string& root) {
    namespace fs = std::filesystem;
    std::vector<std::string> problems;
    std::set<std::string> described, names;
    for (const auto& e : entries) {
        if (!names.insert(e.name).second) problems.push_back("duplicate model name " + e.name);
        described.insert(fs::path(e.description).generic_string());
        const fs::path file = fs::path(root) / e.description;
        if (!fs::exists(file)) {
            problems.push_back(e.id + ": description missing: " + e.description);
        } else {
            try {
                const auto described_file = md::load(file.string());
                const auto* v = described_file.find("name");
                const auto* s = v ? std::get_if<std::string>(&v->data) : nullptr;
                if (!s || *s != e.name) problems.push_back(e.id + ": description name differs from registry name");
            } catch (const std::exception& error) {
                problems.push_back(e.id + ": " + error.what());
            }
        }
        for (const auto* group : {&e.defaults, &e.apps, &e.adapter})
            for (const auto& rel : *group)
                if (!fs::exists(fs::path(root) / rel)) problems.push_back(e.id + ": listed file missing: " + rel);
        if (e.defaults.empty() || e.apps.empty()) problems.push_back(e.id + ": defaults and apps must not be empty");
    }
    for (const auto& item : fs::directory_iterator(fs::path(root) / "models")) {
        if (item.path().extension() != ".toml" || item.path().filename() == "registry.toml") continue;
        const auto rel = "models/" + item.path().filename().generic_string();
        if (!described.count(rel)) problems.push_back("models file without a registry row: " + rel);
    }
    return problems;
}

} // namespace strix::model_registry
