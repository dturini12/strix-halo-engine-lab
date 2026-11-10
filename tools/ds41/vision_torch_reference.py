#!/usr/bin/env python3
"""Run the release's ViT + aligner (inference/vision.py) on pictures, for comparison with the engine's tower.

Loads the deepseek4-vision GGUF into vision.ViT / vision.Aligner, prepares each picture with image_processor.load_image
and writes, per picture, <out>/<stem>.<mode>.rows (raw little-endian BF16 [cells][5120]) and <stem>.patches (BF16
patches), plus <out>/index.json (grid plan per picture). Modes: bf16-cuda (the release's dtype), fp32-cpu (a higher
precision baseline).

    python3 tools/ds41/vision_torch_reference.py --reference DeepSeek-V4.1-Flash-1017/inference \
        --vision DeepSeek-V4.1-Flash-1017-vision-BF16.gguf --out DIR [--modes bf16-cuda fp32-cpu] PICTURE...
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from gguf import GGUFReader


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--vision", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--modes", nargs="+", default=["bf16-cuda", "fp32-cpu"])
    ap.add_argument("pictures", nargs="+")
    a = ap.parse_args()
    vision = load_module(f"{a.reference}/vision.py", "vision")
    ip = load_module(f"{a.reference}/image_processor.py", "image_processor")
    r = GGUFReader(a.vision)
    kv = {f.name: f for f in r.fields.values()}
    def m(key):
        f = kv[f"deepseek4-vision.{key}"]
        return f.parts[f.data[0]].tolist()[0]
    args = SimpleNamespace(vision_dim=m("embedding_length"), vision_n_heads=m("attention.head_count"), vision_inter_dim=m("feed_forward_length"),
                           vision_patch_size=m("patch_size"), vision_rope_theta=float(m("rope.freq_base")), vision_n_layers=m("block_count"),
                           vision_downsample_ratio=m("downsample_ratio"), dim=m("projection_length"), vision_max_n_token=m("image.max_tokens"),
                           vision_min_pixels=m("image.min_pixels"), vision_max_wh_ratio=None)
    state = {}
    for t in r.tensors:
        raw = np.asarray(t.data).view(np.uint16).reshape([int(d) for d in reversed(t.shape)])
        state[t.name] = torch.from_numpy(raw.astype(np.int16)).view(torch.bfloat16).clone()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    index = {}
    models = {}
    for mode in a.modes:
        dtype, device = {"bf16-cuda": (torch.bfloat16, "cuda"), "fp32-cpu": (torch.float32, "cpu"), "bf16-cpu": (torch.bfloat16, "cpu")}[mode]
        vit, al = vision.ViT(args), vision.Aligner(args)
        vit.load_state_dict({k[len("vision."):]: v for k, v in state.items() if k.startswith("vision.")})
        al.load_state_dict({k[len("aligner."):]: v for k, v in state.items() if k.startswith("aligner.")})
        models[mode] = (vit.to(device=device, dtype=dtype).eval(), al.to(device=device, dtype=dtype).eval(), dtype, device)
    for pic in a.pictures:
        stem = Path(pic).stem
        patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = ip.load_image({"data": Path(pic).read_bytes()}, args)
        patches.contiguous().view(torch.int16).numpy().astype("<i2").tofile(out / f"{stem}.patches")
        index[stem] = {"file": str(pic), "vit_w": n_vit_w, "vit_h": n_vit_h, "llm_w": n_llm_w, "llm_h": n_llm_h}
        for mode, (vit, al, dtype, device) in models.items():
            vision.get_vision_cos_sin.cache_clear()
            with torch.inference_mode(), torch.device(device):
                x = vit(patches.to(device=device, dtype=dtype), n_vit_h, n_vit_w)
                rows = al(x, n_vit_h, n_vit_w).to(torch.bfloat16).cpu()
            rows.contiguous().view(torch.int16).numpy().astype("<i2").tofile(out / f"{stem}.{mode}.rows")
            print(stem, mode, tuple(rows.shape), file=sys.stderr)
    json.dump(index, open(out / "index.json", "w"), indent=1)


if __name__ == "__main__":
    main()
