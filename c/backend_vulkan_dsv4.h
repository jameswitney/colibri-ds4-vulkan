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
 * The current implementation is the M1-3 **stub**: it satisfies every
 * dsv4_cuda_* symbol so the seam compiles and links on any box (no ICD), logs
 * the per-layer dense-set uploads (REVIEW G7 inventory), and fails every
 * compute op so the engine falls back to CPU per-op (D6) — output stays
 * token-for-token identical to a pure-CPU run. M2 replaces the bodies with
 * real Vulkan ops (fmt=8 matmul first), M3 wires the decode path.
 */
#include "backend_cuda_dsv4.h"

#endif /* COLIBRI_BACKEND_VULKAN_DSV4_H */
