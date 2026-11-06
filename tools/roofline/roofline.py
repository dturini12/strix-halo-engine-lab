#!/usr/bin/env python3
"""Aggregate kernel CSV time against explicitly configured bandwidth/compute floors."""

import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys


class InputError(ValueError):
    pass


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InputError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _constant(value):
    raise InputError(f"invalid JSON number {value}")


def _float(text):
    value = float(text)
    if not math.isfinite(value):
        raise InputError(f"invalid finite JSON number {text}")
    return value


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_pairs,
                      parse_constant=_constant, parse_float=_float)


def _object(value, path, required, optional=()):
    if not isinstance(value, dict):
        raise InputError(f"{path}: expected object, actual {value!r}")
    missing = set(required) - value.keys()
    unknown = value.keys() - set(required) - set(optional)
    if missing or unknown:
        raise InputError(f"{path}: missing fields {sorted(missing)}, unsupported fields {sorted(unknown)}")


def _string(value, path):
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{path}: expected nonempty string, actual {value!r}")


def _positive(value, path):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise InputError(f"{path}: expected finite positive number, actual {value!r}")
    return value


TIME = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}
WORK = {"bandwidth": {"B": 1.0, "kB": 1e3, "MB": 1e6, "GB": 1e9,
                       "KiB": 1024.0, "MiB": 1024.0 ** 2, "GiB": 1024.0 ** 3},
        "compute": {"FLOP": 1.0, "GFLOP": 1e9, "TFLOP": 1e12}}
RATE = {"bandwidth": {"B/s": 1.0, "GB/s": 1e9, "GiB/s": 1024.0 ** 3},
        "compute": {"FLOP/s": 1.0, "GFLOP/s": 1e9, "TFLOP/s": 1e12}}


def validate(mapping, ceilings):
    _object(mapping, "mapping", ("schema", "kernel_column", "time", "rules", "unknown"))
    if mapping["schema"] != "strix-roofline-map/v1":
        raise InputError(f"mapping.schema: expected strix-roofline-map/v1, actual {mapping['schema']!r}")
    _string(mapping["kernel_column"], "kernel_column")
    if mapping["unknown"] not in ("error", "unclassified"):
        raise InputError(f"unknown: expected error/unclassified, actual {mapping['unknown']!r}")
    timing = mapping["time"]
    _object(timing, "time", ("unit", "kind"), ("duration_column", "start_column", "end_column", "calls_column"))
    duration = "duration_column" in timing
    if duration == ("start_column" in timing or "end_column" in timing):
        raise InputError("time: expected duration_column OR start_column and end_column")
    if not duration and not {"start_column", "end_column"} <= timing.keys():
        raise InputError("time: missing start_column or end_column")
    if not isinstance(timing["unit"], str) or timing["unit"] not in TIME or timing["kind"] not in ("total", "mean"):
        raise InputError(f"time: expected ns/us/ms/s and total/mean, actual {timing!r}")
    if not duration and timing["kind"] != "total":
        raise InputError("time.kind: timestamp interval requires total")
    if timing["kind"] == "mean" and "calls_column" not in timing:
        raise InputError("time.calls_column: required for mean durations")
    for key, value in timing.items():
        if key.endswith("_column"):
            _string(value, f"time.{key}")
    if not isinstance(mapping["rules"], list) or not mapping["rules"]:
        raise InputError("rules: expected nonempty array")
    rules = []
    for index, rule in enumerate(mapping["rules"]):
        _object(rule, f"rules[{index}]", ("pattern", "family"))
        _string(rule["pattern"], f"rules[{index}].pattern")
        _string(rule["family"], f"rules[{index}].family")
        try:
            rules.append((re.compile(rule["pattern"]), rule["family"]))
        except re.error as error:
            raise InputError(f"rules[{index}].pattern: {error}") from error
    _object(ceilings, "ceilings", ("schema", "normalization", "families"))
    if ceilings["schema"] != "strix-roofline-ceilings/v1":
        raise InputError(f"ceilings.schema: expected strix-roofline-ceilings/v1, actual {ceilings['schema']!r}")
    normalization = ceilings["normalization"]
    _object(normalization, "normalization", ("samples", "iterations_per_sample", "unit"))
    for key in ("samples", "iterations_per_sample"):
        if type(normalization[key]) is not int or normalization[key] <= 0:
            raise InputError(f"normalization.{key}: expected positive integer, actual {normalization[key]!r}")
    _string(normalization["unit"], "normalization.unit")
    if not isinstance(ceilings["families"], dict) or not ceilings["families"]:
        raise InputError("families: expected nonempty object")
    for family, ceiling in ceilings["families"].items():
        _string(family, "family name")
        _object(ceiling, f"families.{family}", ("kind", "work", "work_unit", "rate", "rate_unit"),
                ("owner", "source"))
        kind = ceiling["kind"]
        if not isinstance(kind, str) or kind not in WORK:
            raise InputError(f"families.{family}.kind: expected bandwidth/compute, actual {kind!r}")
        _positive(ceiling["work"], f"families.{family}.work")
        _positive(ceiling["rate"], f"families.{family}.rate")
        if (not isinstance(ceiling["work_unit"], str) or not isinstance(ceiling["rate_unit"], str)
                or ceiling["work_unit"] not in WORK[kind] or ceiling["rate_unit"] not in RATE[kind]):
            raise InputError(f"families.{family}: incompatible work/rate units {ceiling!r}")
        for key in ("owner", "source"):
            if key in ceiling:
                _string(ceiling[key], f"families.{family}.{key}")
    for _, family in rules:
        if family not in ceilings["families"]:
            raise InputError(f"rules.family: expected configured ceiling, actual {family!r}")
    return rules


def _number(text, field, row):
    try:
        value = float(text)
    except (TypeError, ValueError) as error:
        raise InputError(f"CSV row {row} {field}: expected finite number, actual {text!r}") from error
    if not math.isfinite(value):
        raise InputError(f"CSV row {row} {field}: expected finite number, actual {text!r}")
    return value


def analyze(trace, mapping, ceilings):
    rules = validate(mapping, ceilings)
    timing = mapping["time"]
    required = {mapping["kernel_column"]} | {value for key, value in timing.items() if key.endswith("_column")}
    totals = {}
    with Path(trace).open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source, strict=True)
        headers = reader.fieldnames
        if not headers or len(headers) != len(set(headers)) or not required <= set(headers):
            raise InputError(f"CSV headers: expected unique fields including {sorted(required)}, actual {headers!r}")
        for row_number, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise InputError(f"CSV row {row_number}: wrong number of columns")
            kernel = row[mapping["kernel_column"]]
            _string(kernel, f"CSV row {row_number} kernel")
            matches = [family for pattern, family in rules if pattern.fullmatch(kernel)]
            if len(matches) > 1:
                raise InputError(f"CSV row {row_number} kernel {kernel!r}: ambiguous family rules {matches}")
            if not matches and mapping["unknown"] == "error":
                raise InputError(f"CSV row {row_number}: unclassified kernel {kernel!r}")
            # None cannot collide with any user-provided family name.
            family = matches[0] if matches else None
            calls = 1
            if "calls_column" in timing:
                calls_text = row[timing["calls_column"]]
                if re.fullmatch(r"[1-9][0-9]*", calls_text) is None:
                    raise InputError(f"CSV row {row_number} calls: expected positive integer, actual {calls_text!r}")
                calls = int(calls_text)
            if "duration_column" in timing:
                column = timing["duration_column"]
                duration = _number(row[column], column, row_number)
            else:
                start, end = timing["start_column"], timing["end_column"]
                begin = _number(row[start], start, row_number)
                finish = _number(row[end], end, row_number)
                if begin < 0:
                    raise InputError(f"CSV row {row_number} {start}: negative timestamp")
                duration = finish - begin
            if duration <= 0:
                raise InputError(f"CSV row {row_number}: duration must be positive, actual {duration}")
            seconds = duration * TIME[timing["unit"]] * (calls if timing["kind"] == "mean" else 1)
            entry = totals.setdefault(family, {"seconds": 0.0, "calls": 0, "rows": 0, "kernels": set()})
            entry["seconds"] += seconds
            if not math.isfinite(entry["seconds"]):
                raise InputError(f"CSV row {row_number}: duration overflow")
            entry["calls"] += calls
            entry["rows"] += 1
            entry["kernels"].add(kernel)
    if not totals:
        raise InputError("CSV: no kernel rows")
    norm = ceilings["normalization"]
    iterations = norm["samples"] * norm["iterations_per_sample"]
    rows = []
    for family, total in totals.items():
        measured = total["seconds"] * 1000 / iterations
        config = ceilings["families"].get(family)
        row = {"family": family, "status": "classified" if config else "unclassified", "measured_ms": measured,
               "ceiling_ms": None, "percent_of_ceiling": None, "gap_ms": None,
               "calls_per_iteration": total["calls"] / iterations, "csv_rows": total["rows"],
               "kernels": sorted(total["kernels"]), "ceiling": config}
        if config:
            kind = config["kind"]
            floor = config["work"] * WORK[kind][config["work_unit"]] / (config["rate"] * RATE[kind][config["rate_unit"]]) * 1000
            if not math.isfinite(floor) or floor <= 0 or not math.isfinite(measured) or measured <= 0:
                raise InputError(f"family {family!r}: nonfinite or zero derived timing")
            row.update(ceiling_ms=floor, percent_of_ceiling=100 * floor / measured, gap_ms=measured - floor)
        rows.append(row)
    rows.sort(key=lambda row: (row["gap_ms"] is None, -(row["gap_ms"] or 0), row["family"] or ""))
    return {"schema": "strix-roofline-result/v1", "normalization": norm, "duration_unit": "ms",
            "aggregation": "sum of kernel durations; not elapsed wall time or a concurrency model",
            "rows": rows}


def table(result):
    unit = result["normalization"]["unit"]
    lines = [f"| family | measured ms/{unit} | ceiling ms/{unit} | % of ceiling | gap ms/{unit} | owner |",
             "|---|---:|---:|---:|---:|---|"]
    for row in result["rows"]:
        def cell(value):
            return "-" if value is None else f"{value:.3f}"
        family = row["family"] if row["family"] is not None else "UNCLASSIFIED"
        owner = (row["ceiling"] or {}).get("owner", "-")
        family = family.replace("|", "\\|").replace("\n", " ").replace("\r", " ")
        owner = owner.replace("|", "\\|").replace("\n", " ").replace("\r", " ")
        lines.append(f"| {family} | {cell(row['measured_ms'])} | {cell(row['ceiling_ms'])} | "
                     f"{cell(row['percent_of_ceiling'])} | {cell(row['gap_ms'])} | {owner} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--map", required=True, type=Path, dest="mapping")
    parser.add_argument("--ceilings", required=True, type=Path)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = analyze(args.trace, load_json(args.mapping), load_json(args.ceilings))
        if args.json_output:
            args.json_output.write_bytes((json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
        print(table(result), end="")
    except (OSError, ValueError, csv.Error) as error:
        print(f"roofline: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
