#!/usr/bin/env python3
"""Synthetic host emitter. These literals are fixtures, never engine observations."""

import argparse
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--variant", default="equal", choices=("equal", "value", "witness", "digest", "missing-key",
                        "missing-witness", "missing-digest", "invalid-digest", "nonzero", "timeout", "order", "type"))
    parser.add_argument("--rendezvous", type=Path)
    args = parser.parse_args()
    if args.rendezvous:
        args.rendezvous.mkdir(parents=True, exist_ok=True)
        (args.rendezvous / str(args.rank)).write_bytes(b"ready\n")
        deadline = time.monotonic() + 3
        while not (args.rendezvous / str(1 - args.rank)).exists():
            if time.monotonic() > deadline:
                return 9
            time.sleep(0.01)
    digest = "0123456789abcdef"
    record = {"tokens": [2, 3, 5], "decode_identity": "g4-v1", "state_fnv": digest,
              "logits_sha256": "a" * 64, "rank": args.rank, "typed": 1,
              "qwen_receipt": {"identity": "synthetic-qwen", "state_fnv": digest},
              "deepseek_receipt": {"identity": "synthetic-ds", "state_fnv": digest},
              "minimax_receipt": {"identity": "synthetic-mm", "state_fnv": digest}}
    witnesses = ["strix-build-info {\"build\":\"synthetic\"}",
                 "strix-startup-event {\"phase\":\"synthetic\"}",
                 "decode_identity g4-v1",
                 "auto-prefill identity synthetic-prefill",
                 "decode attention identity synthetic-attention",
                 f"rank {args.rank} prefill [mla] identity synthetic-glm"]
    if args.variant == "value":
        record["tokens"] = [2, 3, 8]
    if args.variant == "type":
        record["typed"] = True
    if args.variant == "witness":
        witnesses[3] += "-changed"
    if args.variant == "digest":
        record["state_fnv"] = "fedcba9876543210"
    if args.variant == "missing-key":
        del record["tokens"]
    if args.variant == "missing-digest":
        del record["state_fnv"]
    if args.variant == "invalid-digest":
        record["state_fnv"] = "not-a-digest"
    if args.variant == "missing-witness":
        witnesses.pop(3)
    if args.variant == "order":
        record["tokens"].reverse()
    if "SYNTHETIC_ENV" in os.environ:
        record["environment"] = os.environ["SYNTHETIC_ENV"]
    sys.stdout.buffer.write((json.dumps(record, sort_keys=True) + "\n").encode())
    # Includes CRLF and a non-UTF8 byte to verify lossless capture.
    sys.stderr.buffer.write(("\n".join(witnesses) + "\n").encode() + b"raw-tail:\xff\r\n")
    sys.stdout.buffer.flush()
    sys.stderr.buffer.flush()
    if args.variant == "timeout":
        time.sleep(30)
    return 7 if args.variant == "nonzero" else 0


if __name__ == "__main__":
    sys.exit(main())
