# dsv4-flash architecture — verified review & GPU conversion map

> The goal is to implement **DeepSeek V4 Flash's own architecture** on GPU.
> GLM's Vulkan backend is a *primitive/boilerplate reference only* — every
> compute decision below is taken from `c/deepseek_v4.c`, the existing GPU
> reference `c/backend_cuda_dsv4.cu/.h`, and `docs/deepseek-v4.md`.

## 1. Model facts (from docs/deepseek-v4.md + engine code)

- **43 transformer layers**, hidden size **4096**, **MLA + DSA sparse attention**.
- **256 routed experts + 1 shared** per layer, **top-6**; hash router
  (`ffn.gate.tid2eid`, first `num_hash_layers`) or learned gate
  (`ffn.gate.weight` bf16 + bias, softmax) + `route_scale`.
- 284B total / 13B active. Checkpoint DeepSeek-V4-Flash-0731, ~167 GB.
- **Dense = fp8-e4m3** with **128×128-block E8M0 scales** (6.27 GiB) —
  raw bytes, byte-identical layout, no conversion.
- **Routed experts = native fp4** (~137 GiB on disk, ~12.6 MB each),
  streamed per token via the expert store.
- **Head = bf16** (1.06 GiB resident), RMSNorm + final argmax.
- Per-layer `compress_ratios[]` (43 entries): `ratio==0` = plain windowed MLA,
  `ratio!=0` = compressor, `ratio==4` = DSA (indexer).
- **mHC**: hidden state flows as `hc_mult` parallel copies; `hc_pre`
  (sinkhorn-mixed reduction) before attention/FFN, `hc_post`
  (reconstruction) after — dsv4-specific, has no GLM analogue.

## 2. Per-block compute (token path, from `block_token_impl`)

Input is `hc × 4096` (hc_mult copies). Per layer:

1. **mHC pre (attn)**: per-copy RMSNorm (`attn_norm.weight`) →
   `mixes = hc_attn_fn · input / rms` (`hc_attn_fn`: `(2+hc)·hc × hc·hidden`,
   f32; `hc_attn_base`, `hc_attn_scale[3]`) → **sinkhorn** normalize
   (`hc_split_sinkhorn`, iterations) → `reduced` (1×hidden) + `post`/`comb`.
2. **attention(reduced)** → `branch` (1×hidden).
3. **mHC post**: `state = comb · residual + post · branch` → bf16 round
   (reconstruct hc copies).
4. **mHC pre (ffn)**: same with `ffn_norm.weight` / `hc_ffn_fn`.
5. **moe_token(reduced)** → `branch` (shared expert + top-6 routed).
6. **mHC post** → output_hc → bf16 round.

GPU note: the CUDA tier exposes this as `dsv4_cuda_mhc_pre` /
`dsv4_cuda_mhc_pre_norm` (batched, `M` tokens). mHC is matvecs + sinkhorn
iterations — small per token, but a real per-token cost across 43 layers.

## 3. Attention — the dsv4-specific core (from `attention_token_impl`)

**Weights (per layer)** — all inside the resident dense set (G1 to confirm):

| Tensor | dtype | Shape | Role |
|---|---|---|---|
| `attn.wq_a` | fp8 | q_rank × hidden | input → query latent |
| `attn.q_norm.weight` | bf16 | q_rank | RMSNorm on query latent |
| `attn.wq_b` | fp8 | heads·head_dim × q_rank | latent → per-head queries |
| `attn.wkv` | fp8 | head_dim × hidden | input → **shared latent KV** |
| `attn.kv_norm.weight` | bf16 | head_dim | RMSNorm on latent KV |
| `attn.attn_sink` | f32 | heads | **sigmoid attention-sink biases** |
| `attn.wo_a` | fp8 | o_width × o_group_width | grouped output latent (o_groups) |
| `attn.wo_b` | fp8 | hidden × o_width | latent → output |
| compressor (ratio≠0): `compressor.wgate/wkv` bf16 (coff·head_dim × hidden), `compressor.ape` f32 (ratio × coff·head_dim), `compressor.norm` | | | emits a compressed latent every `ratio` positions |
| indexer (ratio==4): `indexer.compressor.wgate/wkv/ape/norm`, `indexer.weights_proj` bf16 (in × hidden), `indexer.wq_b` fp8 (in·ih × q_rank) | | | DSA top-k selection over compressed latents |

**Flow (per token):**
1. QDQ the input **once**, reused by `wq_a` and `wkv` (`_pre` variant).
2. `q_a = wq_a·x` → q_norm → bf16. `q = wq_b·q_a` (heads×head_dim) →
   per-head RMSNorm → bf16.
3. `kv = wkv·x` (**head_dim latent — one shared K/V per position**, no
   per-head K/V) → kv_norm → bf16.
4. **Compressor** (ratio layers): every `ratio` positions, pool the latent
   (wgate/wkv + ape positional + norm) → append to `compressed[]`.
5. **Indexer** (ratio 4, DSA): advance + `select_batch` over compressed
   keys (indexer compressor → weights_proj → wq_b scores) → top-`index_topk`
   compressed ordinals.
6. **RoPE (yarn)** on the last `rope_dim` of each query head and of the KV
   latent; compressed layers use `compress_rope_theta`, uncompressed use
   `rope_theta` + full yarn (`original_max_position_embeddings`).
7. **KV pool**: window ring (`sliding_window × head_dim`, ring buffer) +
   `compressed[]` + (DSA layers) the selected subset.
8. **Attention kernel — NOT softmax-MHA, NOT GLM's absorb MLA**:
   per head `score = q_head · kv_pool / sqrt(head_dim)`;
   `weight = 1/(1+exp(sink[head] − score))` (**sigmoid attention sink**);
   `output_head = Σ weight·kv` — the **shared latent value**, bf16-rounded.
9. Inverse RoPE on the output tail; `wo_a` (grouped: o_groups groups ×
   o_rank views) → bf16 → `wo_b` → output → bf16.

**Decode GPU path already exists in-engine** (`COLI_V4_GPU_TIER`): a
persistent **device KV ring** (`kv_ring_append` / `kv_comp_append`) fed to
`coli_v4_gpu_sparse_attention_batch_cached[_idx]`, gated at
`compressed_count <= 8192`, **poisoned on failure** (falls back to the CPU
reference loop; the CPU state stays canonical — the ring write always
happens). This is the exact integration pattern to replicate for Vulkan.

## 4. MoE (from `moe_token` / `coli_v4_route`)

- `route`: bf16 `gate.weight · reduced` (+bias / hash router) → top-6 →
  route_scale (a bf16 gate matvec: 4096×256 — a real per-token cost).
- **Shared expert** `w1/w2/w3` (fp8) always computed.
- **Routed experts**: top-6, fp4, streamed from disk via the expert store —
  dual loader pool, **9 lanes default** (`V4_LOADER_LANES` 1–16), prefetch,
  `swiglu`; ~230 × 12.6 MB expert reads/token at decode (measured at 6 %
  VRAM hit on the CUDA tier) — **the decode bottleneck**.

## 5. Prefill vs decode (pipeline shape)

- **Prefill**: layer-major over **128-token chunks** (`V4_PREFILL_CHUNK`)
  inside **4096-token segments** (`V4_PREFILL_SEGMENT`); batched attention
  (`coli_v4_attention_window_batch_ref` — per-token window + compressed
  prefix visibility, `meta[3t]`); expert union (`V4_EXPERT_UNION`); prefix
  checkpoints (`V4_PREFIX_CKPT*`), KVSAVE. CUDA tier: batched attention
  block + transient VRAM expert bank (`COLI_CUDA_MOE_BATCH`, 256-token min,
  2.2 GB) + batched mHC.
- **Decode**: per-token, greedy, single KV slot. CUDA tier: decode
  attention/indexer on GPU + expert mirrors (~8 MB each, cap 4096,
  sized by free VRAM).

## 6. The CUDA tier = the GPU reference for the port (from `backend_cuda_dsv4.h`)

```
uploads: fp8, fp8_bf16, fp4, bf16, f32 (+ fp4 in-place refill)
matvec (decode GEMV) · matmul_batch (prefill GEMM) · matmul_bf16_batch (gate)
sparse_attn_batch / _cached / _cached_idx  (window+compressed+selected, per-token meta)
indexer_score_batch                        (relu(q·k)·w, fp32 order = CPU)
fp8_ref_matmul                             (BITWISE replica of the CPU fp8 matmul)
qkv (fused q_a+q_norm+q_b+wkv+kv_norm) · wo (wo_a groups + wo_b)
matvec_grouped (o-grouped / 256-expert geometry)
expert_group (fp4) · expert_fp8 (shared) · moe (shared+routed fused)
mhc_pre / mhc_pre_norm (batched) · kv_ring_append · kv_comp_append
head_argmax · final_argmax (norm + bf16 head + argmax)
CUDA graphs · profiler · stream_drain (hybrid expert-fill DMA)
```

**Critical numerics fact for RDNA2/Vulkan**: the **generic** CUDA kernels
(any sm_80+, and even Pascal/Turing) are exactly the shape a Vulkan port
wants — **fp32 arithmetic throughout**, fp8 weights decoded through the
**256-entry LUT** (not tensor cores), bf16 as *storage only*
(`__bfloat162float`/`__float2bfloat16`). DeepGEMM (sm120, TMA, tensor
cores) is NVIDIA-only and irrelevant. So the Vulkan tier's numerics target
= the *generic* CUDA tier's numerics = fp32 + LUT decode, matching the CPU
reference chain.

## 7. GLM Vulkan: what is reusable vs what is not

**Reusable (primitives/boilerplate from `backend_vulkan.c`):**
- Device/queue/pipeline init, physical-device pick, ReBAR detection/warning,
  256 MB arenas, VRAM pressure-proofing, spin-then-block fences, VK_PROF,
  shader resolution, `.spv` build/commit flow.
- The matmul shader *structure* (int8 `fmt=1` decode) — we add `fmt=8`
  fp8-e4m3 with 128×128-block scales. The existing GLM int4/int8 kernels do
  NOT apply to dsv4's formats.
- The expert-group dispatch pattern (for the fp4 expert tier later).

**NOT reusable — dsv4-specific, must be implemented from dsv4's own
semantics (CUDA tier is the reference):**
- **Attention core**: sigmoid sink + shared latent KV + window/compressed/
  indexer pools + inverse-RoPE. GLM's `attention_absorb.comp` (absorbed
  query, softmax, value rows) is a *different mechanism* — only the
  "one dispatch per layer over a device KV ring" skeleton transfers.
- **Compressor** and **indexer/DSA** (GLM Vulkan does no DSA at all).
- **mHC** (sinkhorn pre/post) — no GLM analogue.
- **Grouped `wo_a`** (o_groups group views).
- **Route** (bf16 gate matvec + top-k).
- **fp4 expert format** and the batched prefill pipeline (chunks/segments/
  expert union/bank).

**Structure (mirrors the CUDA organization):**
- `backend_vulkan.c` gains the general `fmt=8` fp8 matmul primitive.
- New `backend_vulkan_dsv4.c/.h` (`ds4vk_*` ABI) implements the dsv4 op
  surface above on the shared Vulkan plumbing — the direct analogue of
  `backend_cuda_dsv4.cu`, which is the template for how the engine calls it.

## 8. Parity contract (from upstream `## Validation`)

- Each GPU stage was accepted with **greedy text byte-identical to the
  engine's own CPU reference** (826- and 3324-token prompts, 8 tokens).
- **Bitwise** where the CUDA tier does: `DSV4_IDX_VERIFY=1` (indexer
  projection/scoring), the CPU-replica fp8 matmul
  (`dsv4_cuda_fp8_ref_matmul`); numerics test
  `c/tests/test_dsv4_sparse_attn_batch_cuda.c`.
- Different kernels are **not bit-identical across configs in general** —
  GPU vs CPU diverges by a rounding flip after some tokens (same as two CPU
  runs with different hot-expert sets). Text identity is a **within-config
  regression check**, not a proof across configurations.
- **Our Vulkan tier must reproduce the same contract**: greedy byte-identical
  acceptance against the CPU chain per stage, bitwise ops where CUDA is
  bitwise, and the same honesty about cross-config divergence.

## 9. Open questions for the port

- **config.json dims** (pins every size in this doc): head_dim,
  q_rank (q_lora_rank), o_rank (o_lora_rank), o_groups, rope_dim
  (qk_rope_head_dim), sliding_window, index_head_dim, index_n_heads,
  index_topk, hc_mult, num_hash_layers, num_experts_per_tok,
  `compress_ratios[43]` (which layers are windowed / compressed / DSA).
- **Per-op GPU-vs-CPU v1 decisions** (see REVIEW B-series): attention core,
  compressor, indexer/DSA, mHC, route, shared expert, head.
- **Numerics target**: fp32 + LUT decode (generic-CUDA shape) — confirm the
  Vulkan shaders should reproduce the CPU reference chain the same way.
