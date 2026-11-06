#include "strix/engine_profile.hpp"
#include "strix/deepseek41_pair_profile_table.hpp"

#include <iostream>
#include <map>

// A new host hash namespace must not collide with existing local digest helpers.
namespace strix {
namespace {
constexpr std::uint64_t fnv(std::string_view) { return 7; }
}
static_assert(fnv("legacy digest") == 7);
}

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
}

int main() {
    using namespace strix::profile;
    static_assert(strix::host_fnv::text("") == 0xcbf29ce484222325ULL);
    static_assert(strix::host_fnv::text("hello") == 0xa430d84680aabd0bULL);
    const DefaultItem row{"item", "SWITCH", "1", "0", "receipt"};
    std::map<std::string, std::string> vars;
    const Lookup env = [&vars](const char* key) -> const char* {
        const auto it = vars.find(key);
        return it == vars.end() ? nullptr : it->second.c_str();
    };
    require(resolve_default(row, false, env).value == "1", "default");
    require(resolve_default(row, true, env).value == "0", "baseline");
    vars["SWITCH"] = "";
    require(resolve_default(row, true, env).value.empty(), "explicit empty");
    require(resolve_default(row, true, env, "cli").value == "cli", "CLI precedence");
    auto exempt = row;
    exempt.in_baseline = false;
    vars.clear();
    require(resolve_default(exempt, true, env).value == "1", "baseline exemption");
    for (unsigned bits = 0; bits < 16; ++bits) {
        const bool ready = (bits & 1U) != 0, runnable = (bits & 2U) != 0;
        const bool allow = (bits & 4U) != 0, provisional = (bits & 8U) != 0;
        const auto expected = !ready && !((provisional || allow) && runnable) ? Status::pending :
            provisional ? Status::provisional : ready ? Status::set : Status::unqualified;
        require(admit_column(ready, runnable, allow, provisional) == expected, "column admission matrix");
    }

    const Lever pending{"item", "SWITCH", "--item", {"0", "1", ""}, "0",
        Kind::bit_changing, {}, {false, true, false}, "candidate", "0", "pending"};
    const std::array table{pending};
    const Engine engine{"test", "ALLOW", table};
    Inputs inputs;
    inputs.env = env;
    inputs.profile = Name::fast;
    const auto refused = resolve(engine, inputs);
    require(refused.entries[0].status == Status::pending, "FAST pending refusal");
    require(profile_args(refused).empty(), "pending args");
    inputs.allow_pending = true;
    require(resolve(engine, inputs).entries[0].status == Status::unqualified, "FAST admission stamp");
    vars["SWITCH"] = "0";
    require(resolve(engine, inputs).entries[0].status == Status::off, "OFF wins");
    inputs.cli.emplace_back("item", "2");
    require(resolve(engine, inputs).entries[0].value == "2", "CLI beats env");
    vars.clear();
    inputs.cli.clear();
    inputs.profile = Name::standard;
    require(resolve(engine, inputs).entries[0].value == "0", "DEFAULT unchanged");

    // Historical fingerprint must keep its nonstandard seed and FF separators.
    const auto legacy = resolve(engine, inputs);
    std::uint64_t hash = 1469598103934665603ULL;
    for (const std::string_view field : {"default", "qualified", "item", "0"}) {
        for (const char c : field) { hash ^= static_cast<unsigned char>(c); hash *= 1099511628211ULL; }
        hash ^= 255U; hash *= 1099511628211ULL;
    }
    require(legacy.fingerprint() == hash, "legacy fingerprint bytes");
    const Engine shipped{"deepseek41-pair", "ALLOW", strix::deepseek41::pair_defaults::profile_table};
    inputs.profile = Name::fast;
    const auto shipped_fast = resolve(shipped, inputs);
    require(shipped_fast.find("moe_relaxed")->status == Status::provisional, "shipped table positional compatibility");
    check_peer(legacy, [](auto out, auto in) { std::memcpy(in.data(), out.data(), out.size()); });
    bool mismatch = false;
    try {
        check_peer(legacy, [](auto out, auto in) {
            std::memcpy(in.data(), out.data(), out.size());
            in[8] ^= std::byte{1};
        });
    } catch (const std::runtime_error&) { mismatch = true; }
    require(mismatch, "rank mismatch rejected");

    ExecutionReceipt receipt;
    receipt.engine = "test\n\"";
    receipt.identity.arithmetic_contract = "contract";
    const auto numerical = receipt.identity.id();
    receipt.build = "different build";
    receipt.kernel = "different kernel";
    require(receipt.identity.id() == numerical, "execution independent of numerical identity");
    const auto original = receipt.identity;
    for (auto member : {&NumericalIdentity::arithmetic_contract, &NumericalIdentity::model_sha256,
                        &NumericalIdentity::pack_hash, &NumericalIdentity::table_hash,
                        &NumericalIdentity::state_format, &NumericalIdentity::prefill_lineage,
                        &NumericalIdentity::reduction_plan}) {
        receipt.identity = original;
        receipt.identity.*member += "changed";
        require(receipt.identity.id() != numerical, "every numerical field participates");
    }
    NumericalIdentity a, b;
    a.arithmetic_contract = "a"; a.model_sha256 = "bc";
    b.arithmetic_contract = "ab"; b.model_sha256 = "c";
    require(a.id() != b.id(), "field boundaries");
    receipt.identity = original;
    receipt.digests.emplace_back("state-sha256", "unknown");
    const auto witness = receipt.witness();
    require(witness.starts_with("strix-engine-witness {\"schema\":\"strix-engine-receipt/v1\""), "witness schema");
    require(witness.find('\n') == witness.size() - 1, "one line, escaped controls");
    receipt.digests.emplace_back("state-sha256", "duplicate");
    bool duplicate = false;
    try { (void)receipt.json(); } catch (const std::invalid_argument&) { duplicate = true; }
    require(duplicate, "duplicate digest rejected");
    std::cout << "common profile/receipt tests passed\n";
}
