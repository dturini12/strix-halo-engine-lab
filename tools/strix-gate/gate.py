#!/usr/bin/env python3
"""Run declared A/B commands and compare only requested, rank-local evidence."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


class ContractError(ValueError):
    pass


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _constant(value):
    raise ContractError(f"invalid JSON number {value}")


def _float(text):
    value = float(text)
    if not math.isfinite(value):
        raise ContractError(f"invalid finite JSON number {text}")
    return value


def loads(text):
    return json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant, parse_float=_float)


def _object(value, path, required, optional=()):
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected object, actual {value!r}")
    missing = set(required) - value.keys()
    unknown = value.keys() - set(required) - set(optional)
    if missing or unknown:
        raise ContractError(f"{path}: missing fields {sorted(missing)}, unsupported fields {sorted(unknown)}")


def _string(value, path):
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ContractError(f"{path}: expected nonempty NUL-free string, actual {value!r}")


def _env(value, path):
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected string-to-string object, actual {value!r}")
    for key, item in value.items():
        _string(key, path)
        if "=" in key or not isinstance(item, str) or "\x00" in item:
            raise ContractError(f"{path}.{key}: expected valid environment entry, actual {item!r}")


def validate(job):
    _object(job, "job", ("schema", "engine", "ticket", "sources", "ranks", "commands", "compare"),
            ("env", "timeout_s"))
    if job["schema"] != "strix-gate/v1":
        raise ContractError(f"schema: expected strix-gate/v1, actual {job['schema']!r}")
    for key in ("engine", "ticket"):
        _string(job[key], key)
    _object(job["sources"], "sources", ("A", "B"))
    for arm in ("A", "B"):
        _string(job["sources"][arm], f"sources.{arm}")
    ranks = job["ranks"]
    if (not isinstance(ranks, list) or not ranks
            or any(type(rank) is not int or rank < 0 for rank in ranks)
            or ranks != sorted(set(ranks))):
        raise ContractError(f"ranks: expected sorted unique nonnegative integers, actual {ranks!r}")
    timeout = job.get("timeout_s", 60)
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ContractError(f"timeout_s: expected finite positive seconds, actual {timeout!r}")
    _env(job.get("env", {}), "env")
    _object(job["commands"], "commands", ("A", "B"))
    for arm in ("A", "B"):
        commands = job["commands"][arm]
        if not isinstance(commands, list):
            raise ContractError(f"commands.{arm}: expected array")
        seen = []
        for command in commands:
            _object(command, f"commands.{arm}", ("rank", "argv"), ("env", "cwd"))
            rank = command["rank"]
            if type(rank) is not int or rank not in ranks or rank in seen:
                raise ContractError(f"commands.{arm}.rank: expected each of {ranks} once, actual {rank!r}")
            seen.append(rank)
            argv = command["argv"]
            if not isinstance(argv, list) or not argv:
                raise ContractError(f"commands.{arm}.r{rank}.argv: expected nonempty string array")
            for index, arg in enumerate(argv):
                if not isinstance(arg, str) or "\x00" in arg or (index == 0 and not arg):
                    raise ContractError(f"commands.{arm}.r{rank}.argv[{index}]: invalid argument {arg!r}")
            _env(command.get("env", {}), f"commands.{arm}.r{rank}.env")
            if "cwd" in command:
                _string(command["cwd"], f"commands.{arm}.r{rank}.cwd")
        if sorted(seen) != ranks:
            raise ContractError(f"commands.{arm}.rank: expected {ranks}, actual {sorted(seen)}")
    compare = job["compare"]
    _object(compare, "compare", ("stdout_keys", "stderr_witnesses", "digest_fields"))
    for kind in ("stdout_keys", "stderr_witnesses", "digest_fields"):
        if not isinstance(compare[kind], list):
            raise ContractError(f"compare.{kind}: expected array")
    for kind in ("stdout_keys", "stderr_witnesses"):
        seen = set()
        for item in compare[kind]:
            _string(item, f"compare.{kind}")
            if item in seen:
                raise ContractError(f"compare.{kind}: duplicate {item!r}")
            seen.add(item)
            if kind == "stderr_witnesses":
                try:
                    pattern = re.compile(item.encode("utf-8"), re.MULTILINE)
                except re.error as error:
                    raise ContractError(f"stderr_witnesses {item!r}: {error}") from error
                if pattern.search(b"") is not None:
                    raise ContractError(f"stderr_witnesses {item!r}: empty match is not evidence")
    seen = set()
    for field in compare["digest_fields"]:
        _object(field, "digest_fields", ("stream", "key", "format"))
        _string(field["key"], "digest_fields.key")
        if field["stream"] not in ("stdout", "stderr") or field["format"] not in ("sha256", "fnv64"):
            raise ContractError(f"digest_fields: expected stdout/stderr and sha256/fnv64, actual {field!r}")
        identity = (field["stream"], field["key"])
        if identity in seen:
            raise ContractError(f"digest_fields: duplicate {identity!r}")
        seen.add(identity)
    if not any(compare.values()):
        raise ContractError("compare: at least one evidence selector is required")
    return job


def _lookup(value, key):
    # Dots select nested JSON objects; key=value records use the literal key.
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return []
        value = value[part]
    return [value]


def values(raw, key):
    text = raw.decode("utf-8", errors="surrogateescape")
    try:
        whole = loads(text)
    except json.JSONDecodeError:
        whole = None
    if isinstance(whole, dict):
        return _lookup(whole, key)
    found = []
    assignment = re.compile(r"^\s*" + re.escape(key) + r"\s*[=:]\s*(.*?)\s*$")
    for line in text.splitlines():
        if line.lstrip().startswith("{"):
            try:
                value = loads(line)
            except json.JSONDecodeError:
                continue
            found.extend(_lookup(value, key))
        else:
            match = assignment.fullmatch(line)
            if match:
                found.append(match.group(1))
    return found


def extract(stdout, stderr, compare):
    canonical = {}
    problems = []
    for key in compare["stdout_keys"]:
        name = f"stdout.{key}"
        canonical[name] = values(stdout, key)
        if not canonical[name]:
            problems.append((name, "missing evidence"))
    for field in compare["digest_fields"]:
        name = f"digest.{field['stream']}.{field['key']}"
        items = values(stdout if field["stream"] == "stdout" else stderr, field["key"])
        canonical[name] = items
        width = 64 if field["format"] == "sha256" else 16
        if not items:
            problems.append((name, "missing digest"))
        elif any(not isinstance(item, str) or re.fullmatch(f"[0-9a-fA-F]{{{width}}}", item) is None
                 for item in items):
            problems.append((name, f"invalid digest: expected {width} hex characters"))
    for pattern in compare["stderr_witnesses"]:
        name = f"stderr.regex:{pattern}"
        matches = list(re.finditer(pattern.encode("utf-8"), stderr, re.MULTILINE))
        canonical[name] = [match.group(0).hex() for match in matches]
        if not matches or any(match.start() == match.end() for match in matches):
            problems.append((name, "missing or empty witness"))
    return canonical, problems


def _kill(process):
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5, check=False)
        except (OSError, subprocess.TimeoutExpired):
            pass
        if process.poll() is None:
            process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()


def _write_json(path, data):
    path.write_bytes((json.dumps(data, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode())


def run_job(job, output):
    """Validate, create a fresh evidence directory, execute A then B, write receipt."""
    validate(job)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {"engine": job["engine"], "ticket": job["ticket"], "sources": job["sources"],
              "commands": [], "runs": [], "verdict": None, "elapsed_s": 0, "files": {},
              "differing_fields": [], "scope": "Only requested evidence, A versus B per required rank; no native qualification implied."}
    _write_json(output / "job.json", job)
    canonical = {"A": {}, "B": {}}
    execution_failed = False
    evidence_failed = False
    for arm in ("A", "B"):
        pending = []
        deadline = time.monotonic() + job.get("timeout_s", 60)
        # Open files instead of pipes: neither verbose rank can block its peer.
        try:
            for command in sorted(job["commands"][arm], key=lambda item: item["rank"]):
                rank = command["rank"]
                environment = dict(os.environ)
                environment.update(job.get("env", {}))
                environment.update(command.get("env", {}))
                result["commands"].append({"arm": arm, "label": "pair" if len(job["ranks"]) > 1 else "single",
                                           "rank": rank, "argv": command["argv"], "environment": environment,
                                           "cwd": str(Path(command.get("cwd", os.getcwd())).resolve())})
                run = {"arm": arm, "rank": rank, "stdout": f"{arm}-r{rank}.stdout.bin",
                       "stderr": f"{arm}-r{rank}.stderr.bin", "rc": None, "timed_out": False,
                       "launch_error": None}
                result["runs"].append(run)
                with (output / run["stdout"]).open("wb") as out, (output / run["stderr"]).open("wb") as err:
                    try:
                        process = subprocess.Popen(command["argv"], env=environment, cwd=command.get("cwd"),
                                                   stdout=out, stderr=err, stdin=subprocess.DEVNULL,
                                                   start_new_session=(os.name != "nt"))
                    except OSError as error:
                        run["launch_error"] = str(error)
                    else:
                        pending.append((process, run))
            # Every rank has been launched before waiting on any rank.
            while pending:
                for process, run in pending[:]:
                    if process.poll() is not None:
                        run["rc"] = process.returncode
                        pending.remove((process, run))
                    elif time.monotonic() >= deadline:
                        run["timed_out"] = True
                        _kill(process)
                        run["rc"] = process.returncode
                        pending.remove((process, run))
                if pending:
                    time.sleep(0.01)
        finally:
            for process, _ in pending:
                _kill(process)
        for run in [run for run in result["runs"] if run["arm"] == arm]:
            rank = run["rank"]
            if run["rc"] != 0 or run["timed_out"] or run["launch_error"]:
                execution_failed = True
                result["differing_fields"].append({"rank": rank, "arm": arm, "field": "execution",
                                                   "expected": "exit 0 before timeout", "actual": dict(run)})
            try:
                fields, problems = extract((output / run["stdout"]).read_bytes(),
                                           (output / run["stderr"]).read_bytes(), job["compare"])
            except ContractError as error:
                fields, problems = {}, [("extract", str(error))]
            canonical[arm][str(rank)] = fields
            for field, reason in problems:
                evidence_failed = True
                result["differing_fields"].append({"rank": rank, "arm": arm, "field": field,
                                                   "expected": "present valid evidence", "actual": fields.get(field),
                                                   "reason": reason})
        _write_json(output / f"{arm}-canonical.json", canonical[arm])
    mismatch = False
    for rank in job["ranks"]:
        a, b = canonical["A"][str(rank)], canonical["B"][str(rank)]
        for field in sorted(a.keys() | b.keys()):
            if a.get(field) and b.get(field) and json.dumps(a[field], sort_keys=True) != json.dumps(b[field], sort_keys=True):
                mismatch = True
                result["differing_fields"].append({"rank": rank, "arm": "B", "reference_arm": "A",
                                                   "field": field, "expected": a[field], "actual": b[field]})
    result["verdict"] = ("ERROR" if execution_failed else "MISSING-EVIDENCE" if evidence_failed
                         else "MISMATCH" if mismatch else "PASS")
    result["elapsed_s"] = time.monotonic() - started
    for path in sorted(output.iterdir()):
        result["files"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    _write_json(output / "result.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="fresh evidence directory")
    args = parser.parse_args(argv)
    try:
        result = run_job(loads(args.job.read_text(encoding="utf-8")), args.output)
    except (OSError, ValueError) as error:
        print(f"strix-gate: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"verdict": result["verdict"], "result": str(args.output / "result.json")}))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
