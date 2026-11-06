# Engine identity and receipt contract

`include/strix/engine_profile.hpp` defines the host-only `NumericalIdentity`,
`ExecutionReceipt`, product-profile resolver and defaults row. Engine adapters
retain their own startup admission, baseline rules and environment exports.
This interface does not change a kernel or admit a new numerical route.

The JSON schema tag is `strix-engine-receipt/v1`. A witness is exactly
`strix-engine-witness ` followed by that compact JSON object and one newline.
The machine-readable contract is [receipt-v1.schema.json](receipt-v1.schema.json).
Existing inspection text and receipts remain available; consumers opt into the
common receipt through the engine's adapter. Inspection can produce an incomplete
receipt before assets or devices are loaded.

Every receipt has string fields `engine`, `implementation`, `build`, `kernel`,
`device`, `topology`, `command`, an `identity` object, and a `digests` object mapping unique
names to strings. All fields are emitted, including unknown empty strings.
Digest names must include their algorithm and scope, for example
`committed-kv-sha256`; the producer defines byte order and payload scope.
Do not reinterpret historical FNV digests as SHA-256.

The identity object contains strings `id`, `arithmetic_contract`, `model_sha256`,
`pack_hash`, `table_hash`, `state_format`, `prefill_lineage`, `reduction_plan`.
`id` is a lowercase 16-digit FNV-1a fingerprint over the domain
`strix-numerical-identity/v1` and those seven fields in that order. Each UTF-8 byte
string is prefixed by its unsigned 64-bit byte length in little-endian order.
Model SHA-256 values, when available, use lowercase hex; other content hashes
must identify their algorithm. Empty means unknown. An incomplete identity or
a matching noncryptographic fingerprint alone never establishes compatibility.
Compare the full declared fields and qualification evidence.

Each engine keeps its arithmetic contract. In particular, a separate multiply/add
contract with gradual underflow is distinct from an ascending FMA chain. Changes
to model or pack bytes, tables, state format, prefill history or distributed
reduction plans require recording the corresponding field. Exact implementation
overrides may share numerical identity while changing build/kernel provenance.
FAST combinations require qualification of the resulting configuration; component
receipts do not qualify their composition. Neither a shared profile name nor rank
agreement proves numerical equivalence.

The existing 32-byte product-profile handshake and its historical FNV seed and
separator remain unchanged. It agrees profile selection, not the complete execution
plan. `strix/fnv.hpp` accepts explicit seeds so historical receipt and identity
formats retain their bytes. New compositional identities use its standard seed
and length framing. Device digest implementations are outside this extraction.

Bench, qualification and roofline consumers should preserve this envelope, attach
their own workload and validation evidence separately, and reject unknown schema
versions. A receipt records observations; missing build/device/model/digest fields
are pending evidence, never a passing execution gate. Startup adapters must not
invent those values. Device equality checks, compiler/offline checks and complete
workload qualification remain required before release readiness.
