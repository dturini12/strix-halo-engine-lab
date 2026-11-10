#!/usr/bin/env python3
"""Reference image preprocessing fixtures for strix-ds41-vision-prep-tests.

Runs the model release's inference/image_processor.py (load_image) on synthetic pictures of many shapes and
records, per picture, the grid plan and the SHA-256 of the BF16 patch tensor. The pictures are a formula of
(x, y) that the C++ test regenerates, so no image files are stored:
    r = (7x + 3y) & 255,  g = ((x * y + 17x) >> 3) & 255,  b = ((x ^ y) * 5) & 255

    python3 tools/ds41/vision_prep_fixtures.py --reference DeepSeek-V4.1-Flash-1017/inference \
        tests/data/ds41_vision/prep_cases.json
"""
import argparse
import hashlib
import importlib.util
import io
import json
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image

SIZES = [
    (1, 1), (16, 16), (2, 3), (3, 2), (100, 7), (543, 544), (544, 544), (560, 400), (400, 560), (640, 480),
    (800, 600), (1000, 1000), (1024, 768), (1280, 720), (1366, 768), (1440, 900), (1920, 1080), (1080, 1920),
    (2560, 1440), (3000, 90), (90, 3000), (3840, 2160), (4000, 3), (3, 4000), (12000, 41), (41, 12000),
    (700, 701), (1234, 567), (333, 2222), (999, 1), (1, 999),
]


def picture(w, h):
    y, x = np.mgrid[0:h, 0:w].astype(np.uint32)
    r = (7 * x + 3 * y) & 255
    g = ((x * y + 17 * x) >> 3) & 255
    b = ((x ^ y) * 5) & 255
    return Image.fromarray(np.stack([r, g, b], -1).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True, help="the release's inference/ directory (image_processor.py)")
    ap.add_argument("--files", nargs="*", default=[], help="picture files to record instead of the synthetic set")
    ap.add_argument("out")
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("image_processor", f"{a.reference}/image_processor.py")
    ip = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ip)
    cfg = json.load(open(f"{a.reference}/../config.json"))["vision_config"]
    args = SimpleNamespace(vision_patch_size=cfg["patch_size"], vision_downsample_ratio=cfg["downsample_ratio"],
                           vision_max_n_token=cfg["max_image_tokens"], vision_min_pixels=cfg["min_pixels"],
                           vision_max_wh_ratio=cfg["max_wh_ratio"])
    cases = []
    items = [(None, w, h) for w, h in SIZES] if not a.files else [(f, 0, 0) for f in a.files]
    for path, w, h in items:
        if path is None:
            buf = io.BytesIO()
            picture(w, h).save(buf, "PNG")
            data = buf.getvalue()
        else:
            data = open(path, "rb").read()
            with Image.open(io.BytesIO(data)) as im:
                w, h = im.size
        patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = ip.load_image({"data": data}, args)
        raw = patches.contiguous().view(dtype=__import__("torch").int16).numpy().astype("<i2").tobytes()
        cases.append({**({"file": path.rsplit("/", 1)[-1]} if path else {}), "width": w, "height": h, "best_w": n_vit_w * args.vision_patch_size, "best_h": n_vit_h * args.vision_patch_size,
                      "vit_w": n_vit_w, "vit_h": n_vit_h, "llm_w": n_llm_w, "llm_h": n_llm_h,
                      "span_tokens": ip.num_image_tokens(n_llm_h, n_llm_w), "patches_sha256": hashlib.sha256(raw).hexdigest()})
        print(w, h, "->", n_vit_w, n_vit_h, n_llm_w, n_llm_h, file=sys.stderr)
    json.dump({"config": cfg, "cases": cases}, open(a.out, "w"), indent=1)
    open(a.out, "a").write("\n")


if __name__ == "__main__":
    main()
