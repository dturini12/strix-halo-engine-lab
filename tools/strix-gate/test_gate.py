"""CPU-only subprocess and evidence contract tests."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import gate


HERE = Path(__file__).resolve().parent
WITNESSES = [r"^strix-build-info [^\n]+$", r"^strix-startup-event [^\n]+$",
             r"^decode_identity [^\n]+$", r"^auto-prefill identity [^\n]+$",
             r"^decode attention identity [^\n]+$", r"^rank [01] prefill \[mla\] identity [^\n]+$"]


def make_job(ranks=(0,)):
    return {"schema": "strix-gate/v1", "engine": "synthetic", "ticket": "host-fixture",
            "sources": {"A": "original-host-fixture", "B": "changed-host-fixture"},
            "ranks": list(ranks), "timeout_s": 5,
            "commands": {arm: [{"rank": rank, "argv": [sys.executable, str(HERE / "synthetic_engine.py"),
                                                       "--rank", str(rank)]} for rank in ranks] for arm in ("A", "B")},
            "compare": {"stdout_keys": ["tokens", "decode_identity", "typed", "qwen_receipt.identity",
                                         "deepseek_receipt.identity", "minimax_receipt.identity"],
                        "stderr_witnesses": WITNESSES[:],
                        "digest_fields": [{"stream": "stdout", "key": "state_fnv", "format": "fnv64"},
                                          {"stream": "stdout", "key": "logits_sha256", "format": "sha256"}]}}


class GateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def execute(self, job=None):
        output = self.root / "evidence"
        return gate.run_job(job or make_job(), output), output

    def variant(self, variant, ranks=(0,), rank=0):
        job = make_job(ranks)
        job["commands"]["B"][rank]["argv"] += ["--variant", variant]
        return job

    def test_equal_preserves_bytes_and_hashes_every_artifact(self):
        result, output = self.execute()
        self.assertEqual(result["verdict"], "PASS")
        self.assertEqual(result["differing_fields"], [])
        self.assertEqual(len(result["files"]), 7)
        for name, digest in result["files"].items():
            self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), digest)
        self.assertEqual((output / "A-r0.stdout.bin").read_bytes(), (output / "B-r0.stdout.bin").read_bytes())
        stderr = (output / "A-r0.stderr.bin").read_bytes()
        self.assertEqual(stderr, (output / "B-r0.stderr.bin").read_bytes())
        self.assertTrue(stderr.endswith(b"raw-tail:\xff\r\n"))
        for form in (b"decode_identity ", b"auto-prefill identity ", b"decode attention identity ",
                     b"rank 0 prefill [mla] identity ", b"strix-build-info ", b"strix-startup-event "):
            self.assertIn(form, stderr)
        extracts = json.loads((output / "A-canonical.json").read_bytes())
        self.assertEqual(len(extracts["0"]), 14)
        self.assertEqual(json.loads((output / "result.json").read_bytes()), result)

    def test_value_mismatch(self):
        result, _ = self.execute(self.variant("value"))
        self.assertEqual(result["verdict"], "MISMATCH")
        self.assertEqual(result["differing_fields"][0]["field"], "stdout.tokens")
        self.assertEqual(result["differing_fields"][0]["arm"], "B")

    def test_witness_mismatch(self):
        result, _ = self.execute(self.variant("witness"))
        self.assertEqual(result["verdict"], "MISMATCH")
        self.assertIn("auto-prefill", result["differing_fields"][0]["field"])

    def test_digest_mismatch(self):
        result, _ = self.execute(self.variant("digest"))
        self.assertEqual(result["verdict"], "MISMATCH")
        self.assertEqual(result["differing_fields"][0]["field"], "digest.stdout.state_fnv")

    def test_missing_witness(self):
        result, _ = self.execute(self.variant("missing-witness"))
        self.assertEqual(result["verdict"], "MISSING-EVIDENCE")
        self.assertEqual(result["differing_fields"][0]["rank"], 0)

    def test_missing_digest(self):
        result, _ = self.execute(self.variant("missing-digest"))
        self.assertEqual(result["verdict"], "MISSING-EVIDENCE")
        self.assertEqual(result["differing_fields"][0]["field"], "digest.stdout.state_fnv")

    def test_missing_key(self):
        result, _ = self.execute(self.variant("missing-key"))
        self.assertEqual(result["verdict"], "MISSING-EVIDENCE")

    def test_invalid_digest(self):
        result, _ = self.execute(self.variant("invalid-digest"))
        self.assertEqual(result["verdict"], "MISSING-EVIDENCE")
        self.assertIn("16 hex", result["differing_fields"][0]["reason"])

    def test_missing_in_both_arms_cannot_pass(self):
        job = self.variant("missing-witness")
        job["commands"]["A"][0]["argv"] += ["--variant", "missing-witness"]
        result, _ = self.execute(job)
        self.assertEqual(result["verdict"], "MISSING-EVIDENCE")
        self.assertEqual({field["arm"] for field in result["differing_fields"]}, {"A", "B"})

    def test_nonzero_even_with_equal_evidence(self):
        result, _ = self.execute(self.variant("nonzero"))
        self.assertEqual(result["verdict"], "ERROR")
        self.assertEqual(result["runs"][1]["rc"], 7)

    def test_timeout_preserves_partial_evidence(self):
        job = self.variant("timeout")
        job["timeout_s"] = 1
        result, output = self.execute(job)
        self.assertEqual(result["verdict"], "ERROR")
        self.assertTrue(result["runs"][1]["timed_out"])
        self.assertIn(b"decode attention identity", (output / "B-r0.stderr.bin").read_bytes())

    def test_launch_failure_is_receipted(self):
        job = make_job()
        job["commands"]["B"][0]["argv"] = [str(self.root / "missing-executable")]
        result, _ = self.execute(job)
        self.assertEqual(result["verdict"], "ERROR")
        self.assertTrue(result["runs"][1]["launch_error"])

    def test_both_ranks_equal(self):
        result, _ = self.execute(make_job((0, 1)))
        self.assertEqual(result["verdict"], "PASS")
        self.assertEqual(len(result["runs"]), 4)

    def test_rank1_only_mismatch(self):
        result, _ = self.execute(self.variant("digest", (0, 1), rank=1))
        self.assertEqual(result["verdict"], "MISMATCH")
        self.assertEqual([(field["rank"], field["arm"]) for field in result["differing_fields"]], [(1, "B")])

    def test_rank1_missing_witness(self):
        result, _ = self.execute(self.variant("missing-witness", (0, 1), rank=1))
        self.assertEqual(result["verdict"], "MISSING-EVIDENCE")
        self.assertEqual(result["differing_fields"][0]["rank"], 1)

    def test_concurrent_pair_startup(self):
        job = make_job((0, 1))
        for arm in ("A", "B"):
            for command in job["commands"][arm]:
                command["argv"] += ["--rendezvous", str(self.root / arm)]
        result, _ = self.execute(job)
        self.assertEqual(result["verdict"], "PASS")
        self.assertTrue(all(run["rc"] == 0 for run in result["runs"]))

    def test_order_is_not_sorted_away(self):
        result, _ = self.execute(self.variant("order"))
        self.assertEqual(result["verdict"], "MISMATCH")

    def test_json_value_types_are_compared(self):
        result, _ = self.execute(self.variant("type"))
        self.assertEqual(result["verdict"], "MISMATCH")

    def test_environment_precedence_is_retained(self):
        job = make_job()
        job["env"] = {"SYNTHETIC_ENV": "job"}
        job["compare"]["stdout_keys"].append("environment")
        for arm in ("A", "B"):
            job["commands"][arm][0]["env"] = {"SYNTHETIC_ENV": "command"}
        result, _ = self.execute(job)
        self.assertEqual(result["verdict"], "PASS")
        self.assertTrue(all(command["environment"]["SYNTHETIC_ENV"] == "command" for command in result["commands"]))

    def test_existing_output_is_never_overwritten(self):
        self.execute()
        with self.assertRaises(FileExistsError):
            self.execute()

    def test_cli(self):
        job = self.root / "job.json"
        job.write_text(json.dumps(make_job()), encoding="utf-8")
        command = [sys.executable, str(HERE / "gate.py"), str(job), "--output", str(self.root / "cli")]
        completed = subprocess.run(command, capture_output=True, timeout=15)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout)["verdict"], "PASS")

    def test_extraction_pretty_json_key_value_and_ordered_records(self):
        self.assertEqual(gate.values(b'{\n "nested": {"x": [1,2]}\n}', "nested.x"), [[1, 2]])
        self.assertEqual(gate.values(b"key = first\nkey: second\n", "key"), ["first", "second"])
        self.assertEqual(gate.values(b'{"key": 1}\n{"key": 2}\n', "key"), [1, 2])

    def test_stderr_digest(self):
        compare = {"stdout_keys": [], "stderr_witnesses": [],
                   "digest_fields": [{"key": "state", "stream": "stderr", "format": "fnv64"}]}
        fields, problems = gate.extract(b"", b"state=0123456789abcdef\n", compare)
        self.assertEqual(problems, [])
        self.assertEqual(fields["digest.stderr.state"], ["0123456789abcdef"])

    def test_duplicate_and_nonfinite_json(self):
        for text in ('{"schema":1,"schema":2}', '{"x":NaN}', '{"x":1e999}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                gate.loads(text)

    def test_invalid_jobs(self):
        changes = [("schema", "future"), ("ranks", [0, 0]), ("ranks", [True]), ("timeout_s", 0),
                   ("timeout_s", float("nan")), ("env", {"BAD=KEY": "x"})]
        for key, value in changes:
            job = make_job()
            job[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(gate.ContractError):
                gate.validate(job)
        jobs = []
        job = make_job((0, 1)); job["commands"]["B"].pop(); jobs.append(job)
        job = make_job(); job["commands"]["B"] *= 2; jobs.append(job)
        job = make_job(); job["compare"] = {key: [] for key in job["compare"]}; jobs.append(job)
        job = make_job(); job["compare"]["stderr_witnesses"] = ["["]; jobs.append(job)
        job = make_job(); job["compare"]["stderr_witnesses"] = [".*"]; jobs.append(job)
        job = make_job(); job["compare"]["stdout_keys"] *= 2; jobs.append(job)
        job = make_job(); job["commands"]["A"][0]["argv"] = "shell string"; jobs.append(job)
        job = make_job(); job["typo"] = True; jobs.append(job)
        for index, job in enumerate(jobs):
            with self.subTest(index=index), self.assertRaises(gate.ContractError):
                gate.validate(copy.deepcopy(job))


if __name__ == "__main__":
    unittest.main()
