# Kernel-time roofline table

`roofline.py` uses only the Python 3 standard library. It reads kernel CSV, explicit
kernel-to-op-family rules and ceiling inputs. It emits measured time, configured
ceiling time (a lower bound), efficiency and signed gap, ordered by largest gap.
It does not profile a device or infer work, bandwidth, compute capability, rank
aggregation, numerical identity or concurrency from kernel names.

From the repository root, with Python 3.9 or newer:

```sh
python tools/roofline/roofline.py --trace tools/roofline/example-kernels.csv --map tools/roofline/example-map.json --ceilings tools/roofline/example-ceilings.json --json-output roofline-result.json
python -m unittest discover -s tools/roofline -p 'test_*.py' -v
```

All example inputs are synthetic. Keep generated tables and receipts untracked.
The example produces:

| family | measured ms/token | ceiling ms/token | % of ceiling | gap ms/token | owner |
|---|---:|---:|---:|---:|---|
| weights | 20.000 | 10.000 | 50.000 | 10.000 | weights |
| linears | 2.000 | 1.000 | 50.000 | 1.000 | - |

## Mapping schema: `strix-roofline-map/v1`

See [example-map.json](example-map.json). All fields are required:

- `schema`: exactly `strix-roofline-map/v1`.
- `kernel_column`: the CSV header containing the full kernel name.
- `time`: `unit` (`ns`, `us`, `ms`, `s`) and `kind` (`total` or `mean`), plus either
  `duration_column` or both `start_column` and `end_column`. Timestamp intervals
  require `kind: total`. Optional `calls_column` contains positive integer counts;
  it is mandatory for `mean`. Total durations are already summed for that row and
  are never multiplied by calls. Mean durations are multiplied by calls first.
- `rules`: nonempty array of `{ "pattern": "weight_.*", "family": "weights" }`.
  Patterns are full-match Python regular expressions. Exactly one rule must match
  each classified kernel; overlapping rules are errors, even for the same family.
- `unknown`: `error` refuses unmatched kernels; `unclassified` aggregates them into
  an explicit row with null family, ceiling, efficiency and gap. An unconfigured
  family named by a rule is always an error. No ceiling is fabricated.

CSV uses standard quoted fields and UTF-8 (optional BOM). Header names must be
unique and include all selected columns; unrelated columns are allowed. Per-launch
rocprof CSV may select `KernelName` and `BeginNs` / `EndNs`. Aggregated kernel CSV
may select its total or mean duration and call-count columns explicitly. Select the
correct columns/units for the actual export. Empty CSV, ragged rows, missing columns,
nonfinite/zero/negative durations, negative start timestamps, reversed intervals,
ambiguous names and malformed regexes are refused. The input should include only
the intended measurement window, with warm-up excluded by the caller.

## Ceiling schema: `strix-roofline-ceilings/v1`

See [example-ceilings.json](example-ceilings.json). Required fields:

- `schema`: exactly `strix-roofline-ceilings/v1`.
- `normalization`: positive integer `samples`, positive integer
  `iterations_per_sample`, nonempty string `unit` (for example `token`, `step`,
  or `workload`). The trace contains the complete samples combined; they must have
  equal iteration counts. Samples and iterations multiply, never divide twice.
- `families`: nonempty object keyed by family. Each value has `kind`, `work`,
  `work_unit`, `rate`, `rate_unit`, and optional `owner` / `source` strings.
  Work and rate are finite positive numbers. Work means bytes or FLOP **per one
  normalized iteration**, not total work for the CSV. The source is caller-provided
  provenance, retained in JSON; the tool does not verify it.

Bandwidth kind supports work `B`, `kB`, `MB`, `GB`, `KiB`, `MiB`, `GiB`, and rate
`B/s`, `GB/s`, `GiB/s`. Compute kind supports work `FLOP`, `GFLOP`, `TFLOP`, and
rate `FLOP/s`, `GFLOP/s`, `TFLOP/s`. Decimal prefixes use powers of 1000; binary
prefixes use powers of 1024. Bits per second and ambiguous `GBps` are refused.
FLOP counting (including whether an FMA counts as two) and the work/rate hardware
scope must match. Choose the bandwidth or compute floor appropriate to each family;
there is no implicit hardware default. Unknown fields, duplicate JSON keys,
unsupported schemas, invalid types and nonfinite JSON numbers are errors.

## Calculations and interpretation

For each family:

```text
N = samples * iterations_per_sample
measured_ms = sum(kernel durations in seconds) * 1000 / N
ceiling_ms = work per iteration / configured work rate * 1000
percent_of_ceiling = 100 * ceiling_ms / measured_ms
gap_ms = measured_ms - ceiling_ms
```

A 10 ms floor and 20 ms measured duration gives 50%, with a 10 ms gap. Above-100%
efficiency and negative gaps are retained: investigate inventory, units, rate scope
and overlapping work rather than silently clamping. These floors are bounds, not
predicted runtimes or qualified optimization gains. Adding overlapping kernel times
is busy-time accounting and can exceed wall time. A pair CSV must not silently sum
both concurrent ranks against a single-rank rate: analyze each rank separately or
supply deliberately compatible aggregate work/rate and normalization. Untraced
wall time, communication and idle gaps need independent evidence.

The optional JSON output records the schema, normalization, time unit, aggregation
scope, per-family kernels, row counts, calls per iteration, configured work/rate
provenance and computed values. Missing classifications use JSON null, rendered as
`-` in the table. CLI exits 0 on success or 2 on invalid input/I/O. No device or
network access occurs.

## CPU tests

The unittest command above runs **20 tests** covering aggregation, sample/iteration
normalization, time units, decimal/binary bandwidth and compute units, timestamp
intervals, total/mean call semantics, unknown/ambiguous classifications, percentage,
signed gap, CSV structure/numbers/calls, malformed mappings/ceilings/JSON, CLI output
and JSON output. All data are synthetic; no native timing is claimed.
