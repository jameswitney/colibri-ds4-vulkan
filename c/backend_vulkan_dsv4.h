#ifndef COLIBRI_BACKEND_VULKAN_DSV4_H
#define COLIBRI_BACKEND_VULKAN_DSV4_H

/* dsv4 Vulkan op surface — backend selection header (M1-3 seam spike).
 *
 * Mechanism decision (PLAN D9 / REVIEW C2, spike 2026-08-28): **same-named
 * ABI + guarded include swap**. The engine's COLI_V4_UNIT_GPU glue talks to
 * whichever backend is linked through the `dsv4_cuda_*` ABI in
 * backend_cuda_dsv4.h; the Vulkan backend implements that same ABI (its
 * internal `ds4vk_*` surface lands with the real ops in M2-M3), so:
 *
 *   - the GPU unit's call sites and the BLOCK_HYBRID unit's extern
 *     declarations compile and link unchanged (zero upstream call-site edits),
 *   - the only engine change is a guarded include swap (COLI_V4_GPU_TIER_VK
 *     selects this header; CUDA builds keep backend_cuda_dsv4.h byte-identical),
 *   - rebase-fail-fast is automatic: a new upstream dsv4_cuda_* call is
 *     declared by this header (it re-exports the ABI) but has no definition
 *     in backend_vulkan_dsv4.c until the backend implements it -> loud
 *     undefined-reference link error (proven in the M1-3 spike).
 *
 * The -D rename and a fork of the prototype list were rejected: a 45-81
 * entry rename map is implicit and brittle, and a duplicated header would
 * drift from upstream's. One ABI definition, one implementation file.
 *
 * The implementation is the M1-3 seam stub grown into the M2-3 real dense
 * tier: init brings up the Vulkan backend and shaders; the fp8 uploads land
 * the resident dense set in the backend's host-visible arenas (raw e4m3 +
 * UE8M0->fp32 scale expansion at upload, PLAN D4) with real byte accounting,
 * and the backend falls back to system-RAM host-visible memory when the
 * Resizable-BAR window is exhausted. Every compute op still fails so the
 * engine falls back to CPU per-op (D6) — output stays token-for-token
 * identical to a pure-CPU run. M3 replaces the compute bodies with real
 * Vulkan ops (decode path), M4 the batched prefill ops.
 */
#include "backend_cuda_dsv4.h"

/* ---- ds4vk_* exactness surface (M2-4) ----
 * ds4vk_fp8_ref_matmul: serial-order CPU replica of the CUDA
 * dsv4_cuda_fp8_ref_matmul kernels (row-major: fp32-sequential-within-block +
 * fp64-across-blocks; rows8-packed: (x*v)*scale fp32 sequential) — the BITWISE
 * reference for the production fmt=8 shader (TEST L1 oracle) and a bitwise
 * cross-check against the engine's own CPU kernels. Returns 1 on success. */
int ds4vk_fp8_ref_matmul(const uint8_t *w, const float *bscale,
                         int rows, int cols, int packed_rows8,
                         const float *x, int tokens, float *y);

#endif /* COLIBRI_BACKEND_VULKAN_DSV4_H */
