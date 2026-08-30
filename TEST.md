# TEST — dsv4 Vulkan tier: parity & verification strategy

This file owns the **how we know the GPU does the same thing as the CPU**
question for the opt-in dsv4 Vulkan tier: the layered parity contract, the
harnesses, and where each layer runs. It covers only the delta the Vulkan
tier introduces — the engine's own `make check`/`test-c`/oracle targets stay
its suite.

## 0. Goals & principles

1. **Test in pieces and holistically.** Every op is verified against the CPU
   implementation it replaces (layer L2), *and* the whole engine is verified
   token-for-token against the CPU chain (layer L3). Per-op tests catch math
   errors; engine tests catch integration errors (state, bf16 rounding
   points, ordering, fallback).
2. **"Performs the same" is defined per layer.** Bitwise where the CUDA tier
   is bitwise, thresholded where fp32 reduction order legitimately differs,
   and byte-identical text at the engine level (within the documented parity
   envelope, §2 L3).
3. **Most layers run without the checkpoint.** Synthetic tensors at real
   shapes (L1/L2) and the tiny-fixture oracle (L3) run on any box, GPU or not
   (skip-if-no-ICD, exactly like `tests/test_vk_mxfp4.c`). Only the
   real-checkpoint oracle and the benches need a local checkpoint.
4. **Reuse the existing harnesses.** `-DVK_TEST` in `backend_vulkan.c`, the
   CUDA per-op test files, the upstream oracle targets
   (`deepseek-v4-tiny-check`). No new test framework.
5. **The D6 fallback contract is a testing lever.** Every op falls back to
   CPU on failure, so engine-level A/B can enable *one op at a time* on the
   GPU and compare against the all-CPU run. The per-op enable env
   (`COLI_DSV4_VK_OPS=qkv,wo,route,head,shared,attn,mhc,comp,expert`,
   unset/empty = all hooked ops) turns the whole of L3 into per-op
   isolation.
6. **GPU never owns state.** The CPU state stays canonical (the device KV
   ring write always happens, poison-on-failure — the in-engine pattern at
   the decode attention GPU block). A divergence at L3 therefore means the
   GPU *math* is wrong, not the state.

## 1. Environments

| Env | GPU | Checkpoint | Runs |
|---|---|---|---|
| GitHub CI (`make -C c check` + the `vk-build` compile gate) | none | none | L0, L1/L2 synthetic (skip w/o ICD), L4 forced-CPU fault tests |
| maintainer GPU host (e.g. Intel Arc for dev/test ICD + the target AMD/Intel card) | yes | optional; the real-checkpoint oracle needs it | L0, L1/L2 synthetic, L2 real shapes, L3 oracle (tiny + real), L4 fault injection, L5 benches |

## 2. Layers

### L0 — Build & boot parity
- **VK=0 build parity**: `make deepseek-v4` (no `VK=1`) == upstream build —
  no added flags, no link, no behavior change when the flag is off. Check by
  diffing the compile flags (`nm` the binary: 0 VK symbols) and running the
  validation prompt.
- **VK=1 boot parity**: build with `VK=1`, run with **no**
  `COLI_DSV4_VK_DENSE`: the binary must report the tier
  (`v4_gpu vk tier=unavailable` or equivalent) and produce output identical
  to the VK=0 run. This catches both the upstream #894 silent-ignore class
  and the D9 compile-out hazard (hooks gated on `COLI_V4_GPU_TIER` never
  being defined in the VK build).
- **Backend exclusivity**: `make deepseek-v4 VK=1 CUDA=1` must fail with a
  clear Makefile error — never a silent VK-wins build that orphans the CUDA
  tier. `CUDA=1` alone must remain byte-identical to upstream.
- **Seam completeness**: the VK=1 binary links with zero undefined
  `dsv4_cuda_*` / `ds4vk_*` symbols, and a deliberately injected new
  `dsv4_cuda_*` call in the GPU unit fails the build loudly (rebase-fail-fast
  — a new upstream GPU-unit call can never silently miss the seam).
- Pass: exit 0, identical output streams, warning printed when the tier is
  requested-but-unavailable.

### L1 — fp8 matmul primitive parity (`backend_vulkan.c` fmt=8)
- `tests/test_vk_dsv4_fp8.c` (template: `tests/test_vk_mxfp4.c`), run via
  `make -f Makefile.deepseek-v4 test-vk-dsv4 VK=1`; skip if no Vulkan device
  (exit 0); deterministic seed; real dsv4 shapes (wq_a/wq_b/wkv/wo_a
  grouped/wo_b/head; S=1 decode and S=128 prefill).
- **E8M0 scale expansion must be bitwise** equal to the CPU e8 decode
  (e8-table semantics from `backend_cuda_dsv4.cu` / the CPU reference).
- **Activation QDQ must be bitwise** equal to `coli_fp8_activation_qdq_ref`
  (dynamic e4m3 activations + per-128-block E8M0 scales), including the
  `_pre` hoist contract (wq_a + wkv share one QDQ). The L1 gate links the
  REAL `COLI_V4_UNIT_NATIVE_QUANT` engine unit as the oracle (10/10 cases
  bit-identical incl. the -127 scale-exponent clamp and RNE ties).
- **Bitwise oracle**: `ds4vk_fp8_ref_matmul` — a serial-order CPU-replica
  kernel mirroring `dsv4_cuda_fp8_ref_matmul` — is the bitwise reference for
  the production kernel (which is fp32/fp64-approximate like the CUDA generic
  tier). Verified bitwise vs `quant.h matmul_fp8` and the engine's AVX2
  rows8 kernel.
- **NaN-code discipline**: synthetic weight bytes MUST map the e4m3 NaN codes
  0x7f/0xff away (`nn_byte()`); unfiltered `rand()` bytes poison every dot
  product with NaN. The production-shader contract is thresholded (global
  maxabs/maxref ~1e-7..1e-6, floor-maxrel ~1e-5..1e-2 on cancellation rows).
- Edge cases: partial 128-blocks (I%128 ≠ 0, O%128 ≠ 0 tail E8M0 blocks),
  tiny shapes, o_groups group views, the unstaged path (I beyond the shader's
  staging capacity).

### L2 — Per-op parity, decode path
One test per op, **GPU op vs the exact CPU function the engine calls** (never
a reimplementation):

| Op | CPU reference in the engine | VK test (all `make -f Makefile.deepseek-v4 … VK=1`) |
|---|---|---|
| fused qkv (`q_a`+q_norm+`q_b`, `wkv`+kv_norm) | `coli_fp8_matvec_pre`/`_ref` + `coli_v4_rmsnorm` | `tests/test_dsv4_vk_ops.c` (`test-vk-dsv4-ops`) |
| grouped `wo_a`+`wo_b` | `coli_fp8_matvec_ref` + group views | `test-vk-dsv4-ops` (bitwise at o_width=8192 groups=8) |
| route (bf16 gate matvec + top-k) | `coli_v4_route`/`moe_token` | `test-vk-dsv4-ops` (ids MATCH, weights OK) |
| head (`final_argmax`) | the CPU head path | `test-vk-dsv4-ops` (winner MATCH, value OK) |
| attention core (sigmoid sink + shared latent KV + window/compressed pools + inverse-RoPE) | `coli_v4_sparse_attention_ref` | `tests/test_dsv4_sparse_attn_vk.c` (`test-vk-dsv4-attn`: cached + idx + ring wrap; max diff = one bf16 ulp) |
| mHC pre/post (sinkhorn) | `normalized_hc_pre` / `coli_v4_hc_post` | `tests/test_dsv4_mhc_vk.c` (`test-vk-dsv4-mhc`; post/comb/out BITWISE, batch + comp bf16 matmuls) |
| fp4 routed experts (mirrors, group, fused phases, prefill bank) | `coli_fp4_matvec_ref` + `coli_v4_expert_forward_ref` composition | `tests/test_dsv4_vk_experts.c` (`test-vk-dsv4-experts`: matvec ~6e-7, refill, 0/255-code NaN-row edge, group + fused phases BITWISE, bank batch, OPS gating, L4 hook) |

Per-op contract: bf16 rounding points **bitwise** (storage-only, must round
at the same points), fp32 sums **thresholded** (maxrel), top-k **bitwise
where the CUDA tier is bitwise**. Deterministic seeds, real shapes, synthetic
activations.

### L3 — Engine parity, holistic
- **Greedy token-for-token byte-identical** vs the CPU chain within the
  documented **parity envelope**: the tier is thresholded (shader matmuls are
  fp32/fp64-cross-block tree reductions vs the CPU's exact serial order), so
  a borderline greedy logit CAN flip beyond a verified generation length —
  the same within-config-regression class as the CUDA tier. Verified
  byte-identical to ≤16 generated tokens on the tested prompts (the fp64
  cross-block matmul fix moved the first borderline flip on the long-test
  prompt 18→21); the envelope is prompt- and build-dependent — `scripts/
  vk_parity.sh` defaults to 8 tokens and can run longer (VK_MAX_TOKENS) to
  see it. `make deepseek-v4-tiny-check` covers the no-checkpoint tiny oracle.
- Run with `COLI_DSV4_VK_OPS` to isolate: start all-CPU, then flip one op
  group to GPU per run; each must stay byte-identical to the all-CPU run.
- Cover: decode + prefill batch, window/compressed boundary crossing
  (position 0 of a compressed layer, compressed pool growth), the spec-decode
  path (stays CPU).
- Pass: byte-identical greedy output within the envelope; any divergence is
  a blocker for that op group.

### L4 — Fallback & fault injection
- Contract (D6): "a wedged GPU can slow a run, never corrupt it."
- Cases (`tests/test_dsv4_vk_fallback.c` + fault hook `COLI_DSV4_VK_FAIL`):
  1. no ICD / broken ICD → tier unavailable, output identical;
  2. forced op failure mid-run (`COLI_DSV4_VK_FAIL=wo`) → that op falls back
     to CPU, output identical, no crash — pre-drop this is a plain per-op
     fallback with no reload;
  3. post-D7 (after the dense-RAM drop): first VK failure → one-time ~7.3 GiB
     reload, **permanent CPU mode**, loud warn; output identical and no
     flip-flopping (exactly one reload per run);
  4. VRAM pressure / allocation failure during upload → graceful degrade.
- Pass: output identical, exit 0, expected warnings present.

### L5 — Performance A/B, holistic
- Not parity: dense-off/on × dense-RAM held/cleared (`COLI_DSV4_VK_DROP=0`
  isolates bandwidth-relief from capacity-relief) × loader lanes. Phases stay
  sequential (the loop is serial by dependency, not by implementation).
- Benchmark discipline (mirrors CONTRIBUTING): 8 interleaved runs, disclose
  cpuset/OMP/power state; headline = TTFT + prefill tok/s (batch mode),
  decode secondary. `VK_PROF` per-op timing attributes the win.

## 3. Parity contract summary

| Op / stage | Contract | Bitwise where |
|---|---|---|
| E8M0 → fp32 scale expansion (upload) | equal to CPU e8 decode | **bitwise** (same table) |
| activation QDQ (GPU vs CPU) | equal to `coli_fp8_activation_qdq_ref` | **bitwise** (incl. the `_pre` hoist) |
| head: bf16 decode + norm + argmax | vs the CPU head path | bf16 decode bitwise; argmax exact |
| bf16 rounding points | round at the same points as CPU | **bitwise** |
| fp8 GEMV/GEMM (production shader) | maxrel ~1e-5..2e-3 (fp32/fp64 reduction order) | — |
| fp8 matmul (verification replica) | `ds4vk_fp8_ref_matmul` | **bitwise** (serial order) |
| fp4 expert group / fused phases / bank | vs the CPU per-expert chain | **bitwise** (tiny + real shapes) |
| attention core | vs `coli_v4_sparse_attention_ref` (threshold) + engine text identity | indexer scoring where CUDA is bitwise |
| route top-k | vs CPU | **bitwise** (ids) |
| engine greedy decode / prefill | token-for-token byte-identical within the parity envelope | text identity (within-config regression, not cross-config proof) |

## 4. New & extended files

| File | Layer | Note |
|---|---|---|
| `backend_vulkan.c` (`-DVK_TEST` main) | L1 | fmt=8/fmt=10 branches + dsv4 helper surface |
| `backend_vulkan_dsv4.c/.h` | all | the `dsv4_cuda_*` ABI over the Vulkan backend; per-op enable env |
| `tests/test_vk_dsv4_fp8.c` | L1 | template `test_vk_mxfp4.c`; incl. scale-expansion + QDQ oracle |
| `tests/test_dsv4_vk_ops.c` | L2 | qkv/wo/route/head vs engine CPU functions |
| `tests/test_dsv4_sparse_attn_vk.c` | L2 | attention core (cached + idx + ring wrap) |
| `tests/test_dsv4_mhc_vk.c` | L2 | mHC pre/post + batched variants + comp bf16 matmuls |
| `tests/test_dsv4_vk_experts.c` | L2 | fp4 tier: matvec/upload/refill/group/fused-phases/bank + 0/255 edge + OPS gating + L4 hook |
| `tests/test_dsv4_vk_fallback.c` | L4 | fault injection (needs the `COLI_DSV4_VK_FAIL` test hook) |
| `Makefile.deepseek-v4` | — | `test-vk-dsv4*` targets (VK=1); `check`/`test-c` stay green with VK=0 |
| `scripts/vk_parity.sh` | L3/L5 | orchestrator: build → L1/L2/L4 suites → oracle (tiny + real) → per-op A/B |
| `scripts/vk_longgen_ab.py`, `scripts/vk_agentic_ab.py`, `scripts/vk_kvsave_ab.sh` | L5 | bench reproducibility for the perf claims |

## 5. Running the tests

```bash
# CI / any box (no checkpoint required)
make -C c check              # existing gate stays green (VK=0)
make -C c deepseek-v4 VK=1   # the new CI compile gate

# L1/L2 synthetic suites (VK=1 build; each test skips exit-0 without an ICD)
make -f Makefile.deepseek-v4 test-vk-dsv4 VK=1         # L1: fmt=8 matmul + QDQ + exactness
make -f Makefile.deepseek-v4 test-vk-dsv4-ops VK=1     # L2 (M3a): qkv + grouped wo + route + head
make -f Makefile.deepseek-v4 test-vk-dsv4-attn VK=1    # L2 (M3b): cached sparse attention + idx + wrap
make -f Makefile.deepseek-v4 test-vk-dsv4-mhc VK=1     # L2 (M3c): mHC pre/post (sinkhorn) chain
make -f Makefile.deepseek-v4 test-vk-dsv4-experts VK=1 # L2 (M5): fp4 routed-expert tier
make -f Makefile.deepseek-v4 test-vk-dsv4-fallback VK=1 # L4 (M3d): COLI_DSV4_VK_FAIL fault injection

# L3/L5 on a GPU host (checkpoint + ICD)
./scripts/vk_parity.sh       # build → tiny oracle → real oracle → per-op A/B → benches

# L4 fault injection (test build)
COLI_DSV4_VK_FAIL=wo ./deepseek_v4 …   # forced failure → CPU fallback, identical output
```

## 6. Anti-goals

- Not a benchmark methodology (that is the engine's own; the bench section
  here only names the A/B axes and the discipline).
- Not a general engine test plan — upstream `make check`/`test-c`/oracle
  targets stay the engine's own suite; TEST.md only covers the delta the
  Vulkan tier introduces.
- No new test framework, no golden-file sprawl: one synthetic harness per
  layer, one oracle path, skip-if-unavailable semantics everywhere.
