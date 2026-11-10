# DeepSeek V4.1: image input

The 1017 release ships a vision tower (a 32-block ViT and a 3 x 3 aligner, 970 MB in BF16) that turns a picture into
rows of the model's hidden size; the language model reads them in place of token embeddings. This page is the contract
of the engine's implementation: what each step computes, in which order, and how the pair runs it. The model release's
reference is `inference/image_processor.py`, `inference/vision.py`, the image paths of `inference/model.py` and
`inference/engram.py`, and the image blocks of `encoding/encoding.py` (revision `1bae0e10`).

Code: `src/image/` (decoders), `src/ds41_vision_prep.cpp` (preprocessing), `src/ds41_vision.cpp` (vision file),
`src/ds41_vision_reference.cpp` (host reference), `include/strix/ds41_vision_math.hpp` (element-wise arithmetic shared
by host and device), `src/hip_ds41_vision.hip` (device tower), `include/strix/ds41_image_keys.hpp` (image keys),
`src/serve/deepseek_chat.cpp` and `src/serve/api.cpp` (requests), `apps/deepseek41_serve_main.cpp` (the pair).

## Requests

- **Routes:** OpenAI `/v1/chat/completions` (`image_url` parts, a string or `{url}`), `/v1/responses` (`input_image`),
  Anthropic `/v1/messages` (`image` blocks with a base64 source). Pictures may sit in user messages and in tool results
  (a screenshot returned by a tool), and in system / assistant content parts as the reference allows.
- **Data:** base64 `data:` URLs only (no fetching). PNG and JPEG; other formats are refused naming the format.
- **Limits:** `--vision-max-images` (default 4) pictures and `--vision-image-tokens` (default 4096, at least one full
  picture) span tokens per request; 400 beyond them. Sessions (`session` field) refuse pictures.
- **Prompt text:** `render_deepseek_v41_chat` writes the reference placeholder `<｜deepseek_image｜>` for each image part
  and joins the content parts of a message with blank lines, as `encoding.py` does (also for text-only multi-part
  content of system, assistant and tool messages). A literal placeholder in request text is escaped to
  `<|deepseek_image|>` (the reference refuses it), so every placeholder in the prompt is a picture's. The renderer equals
  the reference encoder on `tests/data/serve/deepseek_v41_chat_image_cases.json` (`tools/ds41/make_chat_image_cases.py`).
- **Tokens:** the placeholder is a normal token in the GGUF vocabulary, so the API splits the rendered text at each
  placeholder, encodes the pieces and inserts the picture's span: `[IMAGE_START] + ([IMAGE] * w + [IMAGE_NEW_LINE]) * h +
  [IMAGE_END]`, every position carrying `image_token_id` (129264), `h * (w + 1) + 2` positions.

## Pictures

`src/image/` decodes PNG (every colour type and depth, Adam7) and JPEG (baseline, extended, progressive) to 8-bit RGB
exactly as Pillow 11.3 / libjpeg-turbo 3.1 give `Image.open(...).convert("RGB")`: PNG alpha dropped, 16-bit colour
samples by their high byte and 16-bit grey clipped at 255, low-depth grey scaled; JPEG with libjpeg's integer islow
IDCT, its fancy upsampling (2x1, 1x2, 2x2; replication when the chroma is at most 2 wide), its fixed-point YCbCr tables
and its colour-space guess. No EXIF orientation (the reference applies none). `strix-image-decode-tests` compares the
bytes with Pillow on 52 generated pictures (`tools/image/make_decode_fixtures.py`); on 85 more (wallpapers, icons,
progressive photos) they were equal too.

Preprocessing (`prepare`) is `load_image`: the grid plan (`plan_image_grid`, `safe_resize`, `solve_resize_ratio` in
double precision with Python's operation order: pictures under `min_pixels` = 295,936 are planned at that area, the
token budget is 1024 positions per picture), `ImageOps.pad` on grey 127 with Pillow's 8-bit BICUBIC resample, and the
patches `bf16(((v / 255) - 0.5) / 0.5)` in FP32, `[vit_h * vit_w][3][14][14]`. `strix-ds41-vision-prep-tests` compares
plan and patch bytes with the reference on 31 shapes (`tools/ds41/vision_prep_fixtures.py`). A picture's identity is
the SHA-256 of its plan and patches.

## Declared arithmetic

The tower is the reference's in the engine's declared-arithmetic style: every operation is fixed, so the host
reference (`encode_reference`) and the device tower give the same bits. BF16 rounding is round-to-nearest-even (torch's
`.to(bfloat16)`); `bf(x)` below. Values are BF16 where the release's BF16 model keeps them.

- **Linear** `y = bf(acc + b)`: one FP32 accumulator per output, `acc = fma(x[k], w[o][k], acc)` for `k = 0, 1, ...`
  from `+0` (BF16 times BF16 is exact in FP32, so this is also `acc + x * w`); bias (BF16) added once.
- **RMSNorm:** `ss` is the wave-order sum of `fma(x, x, part)` (lane `l` of 32 sums elements `l, l + 32, ...` in
  ascending order; lanes combined by the butterfly `v = v + v[l ^ d]`, `d = 16, 8, 4, 2, 1`); `r = 1 / sqrt(ss / D +
  1e-6)` with a correctly rounded root and division; `out = bf(w * (x * r))`.
- **2D rotary:** tables computed on the host, `f_j = 1 / 10000^(2j / 32)` rounded from double, angle = `pos * f_j` in
  FP32 (`pos` = the patch's grid row for `j < 16`, its column otherwise), cos / sin from double. Rotate-half within each
  64-wide head: `o1 = (x1 c) - (x2 s)`, `o2 = (x2 c) + (x1 s)`, each product and sum one FP32 operation, then BF16.
- **Attention** (per head and query row, keys in tiles of 64; an online softmax in a fixed order):
  `s_j = (sum_d fma(q[d], k_j[d])) * 0.125`; `m' = max(m, max over the tile)`; `a = exp(m - m')`; `p_j = exp(s_j - m')`;
  `l = l * a`, then `l = l + tile sum` (16 groups of 4 consecutive keys `((p0 + p1) + p2) + p3`, combined by the
  butterfly `d = 8, 4, 2, 1`); `o = o * a`, then `o[d] = fma(p_j, v_j[d], o[d])` over the tile's keys in order;
  finally `bf(o / l)`. `exp` is `declared_exp` (ds41_moe.hpp).
- **Residuals** `x = bf(x + y)`; **MLP** `bf(bf(silu(g)) * u)` with `silu(g) = g / (1 + exp(-g))`; **aligner**
  `bf(gelu(a))` with `gelu(a) = (a * 0.5) * (1 + erf(a * sqrt(1/2)))`, `erf` = `declared_erf` (fdlibm/musl rational
  forms, basic operations only; worst relative error 7.1e-8 against double `erf`).
- **Unfold:** the final norm's rows on the patch grid, zero-padded to multiples of 3; cell `(i, j)` feature
  `c * 9 + ki * 3 + kj` reads patch `(3i + ki, 3j + kj)` (`F.unfold` order).

Against PyTorch (the release's `vision.py`, GGUF weights; `tools/ds41/vision_torch_reference.py`) on
`examples/images/corn.jpeg`: the host rows are closer to the FP32 tower (mean row cosine 0.99914, worst 0.983) than
the release's own BF16 CUDA run is (0.99882, worst 0.945).

## Device tower

`DeviceTower` (HIP) keeps the weights resident (926 MiB) with work buffers for the largest picture (9,189 patches,
324 MiB) and a pool of span rows for the pictures of one prefill (`--vision-image-tokens` rows of 10 KiB). Kernels:
a tiled GEMM (128 x 128 or 128 x 64 outputs per block, 16 keys per tile, each output's chain unchanged; tails pad both
operands with zeros, which leave an accumulator that starts at `+0` unchanged), wave-per-row RMSNorm, one fused
attention kernel per (64 query rows, head) with the tiles above (4 x 4 register tiles, FP32 tiles in LDS), and
element-wise kernels sharing `ds41_vision_math.hpp` with the host. `strix-hip-ds41-vision-tests` checks every step
(embedding, each block, final norm, rows, span rows) bitwise against the host reference and times a full picture: on
gfx1151 a 1920 x 1080 picture (8,418 patches, 968 span tokens) takes 3.0 s (attention 2.0 s, linear layers 0.95 s).
`STRIX_VISION_PROFILE=1` prints per-step times.

## The language model

- **Image keys.** Inside the engine every position of a picture's span carries a key instead of 129264:
  `0x80000000 | FNV-1a(picture id, index)` (31 bits), where the picture id is the SHA-256 of the tower identity (the
  vision file's SHA-256 and the arithmetic version text) and the picture's identity. The reference reads the image
  token only for the embedding (replaced) and as an Engram dead position, so keys change nothing there, but the prefix
  cache (chains and block comparison), session histories and the pair's prompt transfer now tell pictures apart, and a
  cached state never matches a picture encoded by another tower.
- **Embedding.** `DevicePrefill::set_image_spans` gives the span rows by position; the embedding step writes them
  (BF16 to FP32) into all four hyper-connection streams.
- **Routing.** Span positions are routed with the gate's image bias (`exp_probs_b_vl`, already loaded from the target
  GGUF); the router's flags are per local row of the router's view.
- **Engram.** A key is a dead position (`EngramHasher`): it hashes as pads, and no n-gram reaches into or across it
  (`NgramHashState`, `DEAD`); the Engram layers leave span rows unchanged (the reference masks their gate to 0).
- **Decoding** never meets a picture: image prompts always take the device prefill (`--prefill-min` does not apply), a
  parked state is not continued when a picture follows it, prompt-lookup drafts stop before a key, and split decode
  (`--hetero-listen`) turns image input off.

## The pair

Both ranks load the tower (`--vision auto|on|off`; auto = on when `DeepSeek-V4.1-Flash-1017-vision-BF16.gguf`, the pinned
file, sits next to the model; `--vision-file` elsewhere; admitted by size and SHA-256) before the device prefill is
sized, so `--prefill-rows auto` sees its memory; the hello compares the image-input settings. Rank 0 decodes and
prepares each picture when the request arrives (HTTP thread). A prefill command with pictures sends, after the prompt,
a header (span start and size, byte count, picture id per picture) and the pictures' bytes; rank 1 prepares each picture
itself and checks its identity and its keys in the prompt (a difference poisons the pair). Both ranks encode the
pictures whose spans end after the prefix-cache resume point (none for a whole-prompt hit) into the pool and run the
prefill with those spans. The tower runs in the prefill's round, so a picture holds the other slots' decode for its
encode time.
