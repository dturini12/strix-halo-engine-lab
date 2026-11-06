#pragma once
// Host-only receipt adapter. This is a partial description of the resolved defaults,
// not a complete execution receipt or evidence of numerical compatibility. Callers
// must add verified asset, state, build, device and workload digests when available.
#include "glm_rank_defaults.hpp"

namespace strix::glm::rank_defaults {

inline strix::profile::NumericalIdentity numerical_identity(const Effective& e) {
    strix::profile::NumericalIdentity identity;
    // Keep the declared arithmetic separate from the selected R1/R1b mode.
    // R1b is not an assertion of R1 bit equality. See glm_math.hpp and the
    // rank app's R1 prompt / R1b generation contract.
    identity.arithmetic_contract = "src/glm/common/glm_math.hpp: IEEE binary32; R1 ascending fmaf chain; "
                                   "R1b 256-term ascending fmaf blocks then pairwise halving tree; "
                                   "R2 lane-strided fmaf partials with pairwise halving tree";
    identity.reduction_plan = e.identity.value;
    identity.prefill_lineage = e.prefill;
    return identity;
}

inline strix::profile::ExecutionReceipt host_receipt(const Effective& e) {
    strix::profile::ExecutionReceipt receipt;
    receipt.engine = "glm";
    receipt.implementation = "strix-glm5-rank";
    receipt.identity = numerical_identity(e);
    // Inspection has no model-content hashes, canonical state, build provenance,
    // native kernel/device identity, observed topology or completed-workload digests.
    return receipt;
}

}  // namespace strix::glm::rank_defaults
