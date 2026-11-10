#!/usr/bin/env python3
"""Pictures and Pillow's decodes for strix-image-decode-tests.

Writes tests/data/image/<name>.{png,jpg} and decode_cases.json: for each file, the size and the SHA-256 of
Pillow's Image.open(...).convert("RGB") bytes (what DeepSeek V4.1's reference image_processor.load_image reads).
Needs Pillow 11.3 (libjpeg-turbo 3.1, zlib); the files are deterministic for a given Pillow build.

    python3 tools/image/make_decode_fixtures.py tests/data/image
"""
import hashlib
import io
import json
import struct
import sys
import zlib
from pathlib import Path

from PIL import Image, features


def pattern(w, h, seed=0):
    """A busy RGB picture: gradients, edges and noise, so every decoder path sees real data."""
    im = Image.new("RGB", (w, h))
    px = im.load()
    s = seed * 2654435761 & 0xFFFFFFFF
    for y in range(h):
        for x in range(w):
            s = (s * 1103515245 + 12345) & 0x7FFFFFFF
            n = (s >> 16) & 31
            r = (x * 255 // max(1, w - 1)) ^ (n if (x // 3 + y // 5) % 2 else 0)
            g = (y * 255 // max(1, h - 1) + n) & 255
            b = 255 if ((x - w // 2) ** 2 + (y - h // 2) ** 2) < (min(w, h) // 3) ** 2 else (x * y) & 255
            px[x, y] = (r & 255, g, b)
    return im


def png_chunk(kind, data):
    c = struct.pack(">I", len(data)) + kind + data
    return c + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def raw_png(w, h, colour, depth, rows, plte=None, trns=None, interlace=0, level=6):
    """A PNG from already-filtered scanline bytes (Pillow cannot write every depth / colour type)."""
    out = b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, depth, colour, 0, 0, interlace))
    if plte is not None:
        out += png_chunk(b"PLTE", plte)
    if trns is not None:
        out += png_chunk(b"tRNS", trns)
    out += png_chunk(b"IDAT", zlib.compress(rows, level))
    return out + png_chunk(b"IEND", b"")


def packed_rows(w, h, depth, channels, value, filters=(0,)):
    """Scanlines of `depth`-bit samples value(x, y, c), each row with a filter byte (filters applied for 8/16)."""
    bpp = max(1, depth * channels // 8)
    out = bytearray()
    prev = bytes(((w * channels * depth + 7) // 8))
    for y in range(h):
        bits, nb, row = 0, 0, bytearray()
        for x in range(w):
            for c in range(channels):
                v = value(x, y, c)
                if depth == 16:
                    row += struct.pack(">H", v)
                else:
                    bits = (bits << depth) | v
                    nb += depth
                    while nb >= 8:
                        nb -= 8
                        row.append((bits >> nb) & 255)
        if nb:
            row.append((bits << (8 - nb)) & 255)
        f = filters[y % len(filters)] if depth >= 8 else 0
        enc = bytearray(len(row))
        for i in range(len(row)):
            a = row[i - bpp] if i >= bpp else 0
            b = prev[i]
            c = prev[i - bpp] if i >= bpp else 0
            if f == 0: p = 0
            elif f == 1: p = a
            elif f == 2: p = b
            elif f == 3: p = (a + b) >> 1
            else:
                pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                p = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
            enc[i] = (row[i] - p) & 255
        out.append(f)
        out += enc
        prev = bytes(row)
    return bytes(out)


def adam7_rows(w, h, depth, channels, value):
    passes = [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)]
    out = b""
    for x0, y0, dx, dy in passes:
        pw = (w - x0 + dx - 1) // dx if w > x0 else 0
        ph = (h - y0 + dy - 1) // dy if h > y0 else 0
        if pw and ph:
            out += packed_rows(pw, ph, depth, channels, lambda x, y, c: value(x0 + x * dx, y0 + y * dy, c), filters=(1, 4, 2, 3))
    return out


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    v8 = lambda x, y, c: (x * 37 + y * 11 + c * 80) & 255
    v16 = lambda x, y, c: (x * 9001 + y * 577 + c * 20000) & 65535

    # PNG: every colour type and depth, filters, Adam7, palette shorter than the indices, tRNS, odd sizes
    for depth in (1, 2, 4, 8, 16):
        files[f"grey{depth}.png"] = raw_png(13, 7, 0, depth, packed_rows(13, 7, depth, 1, lambda x, y, c: (x + 3 * y) % (1 << depth) if depth < 16 else v16(x, y, c), filters=(0, 1, 2, 3, 4)))
    files["grey16_low.png"] = raw_png(9, 5, 0, 16, packed_rows(9, 5, 16, 1, lambda x, y, c: (x * 40 + y * 3) & 511))
    files["grey8_trns.png"] = raw_png(9, 5, 0, 8, packed_rows(9, 5, 8, 1, v8), trns=struct.pack(">H", 40))
    for depth in (1, 2, 4, 8):
        n = 1 << depth
        plte = bytes((i * 47 + k * 90) & 255 for i in range(max(1, n - 1)) for k in range(3))  # one entry short
        files[f"pal{depth}.png"] = raw_png(11, 6, 3, depth, packed_rows(11, 6, depth, 1, lambda x, y, c: (x * 5 + y) % n), plte=plte, trns=b"\x00\x80")
    for colour, ch in ((2, 3), (4, 2), (6, 4)):
        for depth in (8, 16):
            files[f"c{colour}_d{depth}.png"] = raw_png(17, 9, colour, depth, packed_rows(17, 9, depth, ch, v8 if depth == 8 else v16, filters=(0, 1, 2, 3, 4)))
    files["rgb8_adam7.png"] = raw_png(23, 19, 2, 8, adam7_rows(23, 19, 8, 3, v8), interlace=1)
    files["grey2_adam7.png"] = raw_png(10, 3, 0, 2, adam7_rows(10, 3, 2, 1, lambda x, y, c: (x + y) % 4), interlace=1)
    files["rgba16_adam7.png"] = raw_png(5, 9, 6, 16, adam7_rows(5, 9, 16, 4, v16), interlace=1)
    files["tiny_1x1.png"] = raw_png(1, 1, 2, 8, packed_rows(1, 1, 8, 3, v8))
    files["stored.png"] = raw_png(31, 4, 2, 8, packed_rows(31, 4, 8, 3, v8), level=0)
    b = io.BytesIO(); pattern(97, 61, 1).save(b, "PNG", optimize=True); files["pillow_rgb.png"] = b.getvalue()

    # JPEG: subsampling 4:4:4, 4:2:2, 4:2:0, progressive, grey, odd and tiny sizes, qualities, restarts
    def jpg(im, **kw):
        b = io.BytesIO(); im.save(b, "JPEG", **kw); return b.getvalue()
    big = pattern(83, 57, 2)
    for ss in (0, 1, 2):
        files[f"jpg_ss{ss}.jpg"] = jpg(big, quality=85, subsampling=ss)
        files[f"jpg_ss{ss}_prog.jpg"] = jpg(big, quality=70, subsampling=ss, progressive=True)
    files["jpg_q100.jpg"] = jpg(big, quality=100, subsampling=0)
    files["jpg_q5.jpg"] = jpg(big, quality=5, subsampling=2)
    files["jpg_grey.jpg"] = jpg(big.convert("L"), quality=80)
    files["jpg_grey_prog.jpg"] = jpg(big.convert("L"), quality=60, progressive=True)
    for (w, h) in ((1, 1), (2, 2), (3, 5), (4, 3), (5, 2), (17, 9), (16, 16), (33, 1)):
        files[f"jpg_{w}x{h}_420.jpg"] = jpg(pattern(w, h, w * 7 + h), quality=90, subsampling=2)
        files[f"jpg_{w}x{h}_422.jpg"] = jpg(pattern(w, h, w * 7 + h), quality=90, subsampling=1)
    files["jpg_restart.jpg"] = jpg(big, quality=75, subsampling=2, restart_marker_blocks=3)
    files["jpg_restart_prog.jpg"] = jpg(big, quality=75, subsampling=2, progressive=True, restart_marker_rows=1)
    files["jpg_optimize.jpg"] = jpg(big, quality=75, subsampling=2, optimize=True)

    cases = []
    for name, data in sorted(files.items()):
        (out / name).write_bytes(data)
        im = Image.open(io.BytesIO(data)).convert("RGB")
        cases.append({"file": name, "width": im.width, "height": im.height, "rgb_sha256": hashlib.sha256(im.tobytes()).hexdigest()})
    meta = {"pillow": Image.__version__, "libjpeg_turbo": features.version_feature("libjpeg_turbo"), "zlib": features.version("zlib")}
    (out / "decode_cases.json").write_text(json.dumps({"generator": meta, "cases": cases}, indent=1) + "\n")
    print(f"{len(cases)} cases", meta)


if __name__ == "__main__":
    main()
