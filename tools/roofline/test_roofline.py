"""CPU-only unit, normalization, aggregation and input-contract tests."""

import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import roofline


HERE = Path(__file__).resolve().parent


def inputs():
    mapping = {"schema": "strix-roofline-map/v1", "kernel_column": "KernelName",
               "time": {"unit": "ns", "kind": "total", "duration_column": "DurationNs"},
               "rules": [{"pattern": "weight_.*", "family": "weights"},
                         {"pattern": "gemm", "family": "linears"}], "unknown": "error"}
    ceilings = {"schema": "strix-roofline-ceilings/v1",
                "normalization": {"samples": 2, "iterations_per_sample": 2, "unit": "token"},
                "families": {"weights": {"kind": "bandwidth", "work": 1, "work_unit": "GB",
                                         "rate": 100, "rate_unit": "GB/s", "owner": "weights"},
                             "linears": {"kind": "compute", "work": 1, "work_unit": "GFLOP",
                                         "rate": 1, "rate_unit": "TFLOP/s"}}}
    return mapping, ceilings


class RooflineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.trace = self.root / "trace.csv"
        self.mapping, self.ceilings = inputs()
        self.write("KernelName,DurationNs\nweight_up,40000000\nweight_down,40000000\ngemm,8000000\n")

    def write(self, text):
        self.trace.write_bytes(text.encode())

    def analyze(self):
        return roofline.analyze(self.trace, self.mapping, self.ceilings)

    def test_aggregation_normalization_percent_and_gap(self):
        result = self.analyze()
        weights, linears = result["rows"]
        self.assertEqual(weights["family"], "weights")
        self.assertAlmostEqual(weights["measured_ms"], 20)
        self.assertAlmostEqual(weights["ceiling_ms"], 10)
        self.assertAlmostEqual(weights["percent_of_ceiling"], 50)
        self.assertAlmostEqual(weights["gap_ms"], 10)
        self.assertEqual(weights["csv_rows"], 2)
        self.assertEqual(weights["calls_per_iteration"], 0.5)
        self.assertEqual(weights["kernels"], ["weight_down", "weight_up"])
        self.assertAlmostEqual(linears["measured_ms"], 2)
        self.assertAlmostEqual(linears["ceiling_ms"], 1)
        self.assertAlmostEqual(linears["percent_of_ceiling"], 50)
        self.assertAlmostEqual(linears["gap_ms"], 1)
        self.assertIn("measured ms/token", roofline.table(result))

    def test_time_units(self):
        for unit, value in (("ns", 80000000), ("us", 80000), ("ms", 80), ("s", 0.08)):
            with self.subTest(unit=unit):
                self.mapping["time"]["unit"] = unit
                self.write(f"KernelName,DurationNs\nweight_up,{value}\n")
                self.assertAlmostEqual(self.analyze()["rows"][0]["measured_ms"], 20)

    def test_binary_and_decimal_bandwidth_units(self):
        config = self.ceilings["families"]["weights"]
        config.update(work=1, work_unit="GiB", rate=1, rate_unit="GB/s")
        self.assertAlmostEqual(next(row for row in self.analyze()["rows"] if row["family"] == "weights")["ceiling_ms"], 1073.741824)
        config["rate_unit"] = "GiB/s"
        self.assertAlmostEqual(next(row for row in self.analyze()["rows"] if row["family"] == "weights")["ceiling_ms"], 1000)

    def test_compute_units(self):
        config = self.ceilings["families"]["linears"]
        for work, work_unit, rate, rate_unit in ((1e9, "FLOP", 1e12, "FLOP/s"),
                                               (1, "GFLOP", 1000, "GFLOP/s"),
                                               (0.001, "TFLOP", 1, "TFLOP/s")):
            with self.subTest(work_unit=work_unit):
                config.update(work=work, work_unit=work_unit, rate=rate, rate_unit=rate_unit)
                row = next(row for row in self.analyze()["rows"] if row["family"] == "linears")
                self.assertAlmostEqual(row["ceiling_ms"], 1)

    def test_timestamp_csv(self):
        self.mapping["time"] = {"start_column": "BeginNs", "end_column": "EndNs", "unit": "ns", "kind": "total"}
        self.write('KernelName,BeginNs,EndNs\n"weight_up",1000,80001000\n')
        self.assertAlmostEqual(self.analyze()["rows"][0]["measured_ms"], 20)

    def test_mean_duration_multiplies_calls(self):
        self.mapping["time"].update(kind="mean", calls_column="Calls")
        self.write("KernelName,DurationNs,Calls\nweight_up,20000000,4\n")
        row = self.analyze()["rows"][0]
        self.assertAlmostEqual(row["measured_ms"], 20)
        self.assertEqual(row["calls_per_iteration"], 1)

    def test_total_duration_does_not_multiply_calls(self):
        self.mapping["time"]["calls_column"] = "Calls"
        self.write("KernelName,DurationNs,Calls\nweight_up,80000000,4\n")
        self.assertAlmostEqual(self.analyze()["rows"][0]["measured_ms"], 20)

    def test_unknown_kernel_error(self):
        self.write("KernelName,DurationNs\nunknown,1000\n")
        with self.assertRaisesRegex(roofline.InputError, "unclassified kernel"):
            self.analyze()

    def test_unknown_kernel_explicit_unclassified(self):
        self.mapping["unknown"] = "unclassified"
        self.write("KernelName,DurationNs\nunknown,1000\n")
        row = self.analyze()["rows"][0]
        self.assertEqual(row["status"], "unclassified")
        self.assertIsNone(row["family"])
        for key in ("ceiling_ms", "percent_of_ceiling", "gap_ms"):
            self.assertIsNone(row[key])
        self.assertIn("UNCLASSIFIED", roofline.table(self.analyze()))

    def test_unknown_family_has_no_fabricated_ceiling(self):
        del self.ceilings["families"]["weights"]
        with self.assertRaisesRegex(roofline.InputError, "configured ceiling"):
            self.analyze()

    def test_ambiguous_rules_error(self):
        self.mapping["rules"].append({"pattern": ".*", "family": "weights"})
        with self.assertRaisesRegex(roofline.InputError, "ambiguous"):
            self.analyze()

    def test_above_ceiling_is_visible(self):
        self.write("KernelName,DurationNs\nweight_up,20000000\n")
        row = self.analyze()["rows"][0]
        self.assertAlmostEqual(row["percent_of_ceiling"], 200)
        self.assertAlmostEqual(row["gap_ms"], -5)

    def test_invalid_csv_numbers(self):
        for value in ("NaN", "inf", "1e999", "-1", "0", "bad", ""):
            with self.subTest(value=value), self.assertRaises(roofline.InputError):
                self.write(f"KernelName,DurationNs\nweight_up,{value}\n")
                self.analyze()

    def test_invalid_csv_structure(self):
        for text in ("", "KernelName,DurationNs\n", "KernelName,Wrong\nweight_up,1\n",
                     "KernelName,DurationNs,DurationNs\nweight_up,1,2\n",
                     "KernelName,DurationNs\nweight_up\n", "KernelName,DurationNs\nweight_up,1,extra\n"):
            with self.subTest(text=text), self.assertRaises(roofline.InputError):
                self.write(text)
                self.analyze()

    def test_invalid_calls(self):
        self.mapping["time"].update(kind="mean", calls_column="Calls")
        for value in ("0", "-2", "1.2", "NaN", ""):
            with self.subTest(value=value), self.assertRaises(roofline.InputError):
                self.write(f"KernelName,DurationNs,Calls\nweight_up,1000,{value}\n")
                self.analyze()

    def test_invalid_timestamp_interval(self):
        self.mapping["time"] = {"start_column": "Begin", "end_column": "End", "unit": "ns", "kind": "total"}
        for begin, end in ((10, 5), (-1, 5), (5, 5)):
            with self.subTest(begin=begin, end=end), self.assertRaises(roofline.InputError):
                self.write(f"KernelName,Begin,End\nweight_up,{begin},{end}\n")
                self.analyze()

    def test_invalid_mapping_and_ceilings(self):
        for key, value in (("schema", "future"), ("unknown", "guess"), ("rules", [])):
            mapping = copy.deepcopy(self.mapping)
            mapping[key] = value
            with self.subTest(key=key), self.assertRaises(roofline.InputError):
                roofline.validate(mapping, self.ceilings)
        for key, value in (("samples", 0), ("iterations_per_sample", True), ("unit", "")):
            ceilings = copy.deepcopy(self.ceilings)
            ceilings["normalization"][key] = value
            with self.subTest(key=key), self.assertRaises(roofline.InputError):
                roofline.validate(self.mapping, ceilings)
        for key, value in (("work", 0), ("rate", -1), ("rate", float("nan")), ("kind", "guess"),
                           ("work_unit", "GFLOP"), ("rate_unit", "Gbps")):
            ceilings = copy.deepcopy(self.ceilings)
            ceilings["families"]["weights"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(roofline.InputError):
                roofline.validate(self.mapping, ceilings)

    def test_invalid_time_config_and_regex(self):
        for timing in ({"unit": "ns", "kind": "mean", "duration_column": "DurationNs"},
                       {"unit": "ns", "kind": "total", "start_column": "Start"},
                       {"unit": "ns", "kind": "total", "duration_column": "DurationNs", "end_column": "End"},
                       {"unit": "minutes", "kind": "total", "duration_column": "DurationNs"}):
            mapping = copy.deepcopy(self.mapping)
            mapping["time"] = timing
            with self.subTest(timing=timing), self.assertRaises(roofline.InputError):
                roofline.validate(mapping, self.ceilings)
        self.mapping["rules"][0]["pattern"] = "["
        with self.assertRaises(roofline.InputError):
            self.analyze()

    def test_duplicate_nonfinite_json(self):
        path = self.root / "bad.json"
        for text in ('{"schema":1,"schema":2}', '{"rate":NaN}', '{"rate":1e999}'):
            path.write_text(text, encoding="utf-8")
            with self.subTest(text=text), self.assertRaises(ValueError):
                roofline.load_json(path)

    def test_cli_and_json(self):
        mapping_path, ceilings_path = self.root / "map.json", self.root / "ceilings.json"
        mapping_path.write_text(json.dumps(self.mapping), encoding="utf-8")
        ceilings_path.write_text(json.dumps(self.ceilings), encoding="utf-8")
        output = self.root / "result.json"
        command = [sys.executable, str(HERE / "roofline.py"), "--trace", str(self.trace),
                   "--map", str(mapping_path), "--ceilings", str(ceilings_path), "--json-output", str(output)]
        completed = subprocess.run(command, capture_output=True, timeout=10)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn(b"50.000", completed.stdout)
        self.assertEqual(json.loads(output.read_bytes()), self.analyze())
        self.write("KernelName,DurationNs\nunknown,1\n")
        failed = subprocess.run(command, capture_output=True, timeout=10)
        self.assertEqual(failed.returncode, 2)
        self.assertIn(b"unclassified kernel", failed.stderr)


if __name__ == "__main__":
    unittest.main()
