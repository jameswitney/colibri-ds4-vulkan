/* backend_vulkan_dsv4.c — dsv4 Vulkan op surface.
 *
 * Implements the full `dsv4_cuda_*` ABI (backend_cuda_dsv4.h) over the shared
 * GLM Vulkan plumbing (backend_vulkan.c). Mechanism (backend_vulkan_dsv4.h):
 *
 *   - M2-3: real backend init + REAL dense-set uploads. dsv4_cuda_init brings
 *     up the Vulkan device and committed shaders; the fp8 uploads
 *     (dsv4_cuda_upload_fp8 / _bf16) copy the resident dense weights into the
 *     backend's host-visible arenas (fmt=8: raw e4m3 bytes, UE8M0 codes
 *     expanded to fp32 at upload, PLAN D4) via coli_vk_tensor_ensure. The
 *     handles carry a ColiVkTensor* that the M3a compute ops will drive.
 *     The engine's per-layer plan dump (coli_v4_gpu_layer_upload) still logs
 *     the G7 inventory names; this backend logs the real bytes + device.
 *   - Budget + ReBAR (PLAN M2-3): dsv4_cuda_mem_free_mb reports real free
 *     VRAM from VK_EXT_memory_budget; the backend's arena falls back to
 *     system-RAM host-visible memory when the Resizable-BAR window is
 *     exhausted (256 MB on this host), so the ~5.5 GiB dense set still
 *     uploads on a ReBAR-disabled card (correct, slower reads over PCIe).
 *   - bf16/f32 uploads deliberately fail (log "deferred"): their consumers
 *     (route, mHC, head argmax, batched attention) are M3a/M3c/M4 ops; the
 *     tensors stay CPU-only in v1, exactly like the fp4 expert mirrors (M5).
 *   - every compute op still returns failure -> the engine's per-op dispatch
 *     falls back to the CPU reference (D6), so output is token-for-token
 *     identical to a pure-CPU run. M3a replaces the matmul-family bodies.
 */
#include "backend_vulkan_dsv4.h"
#include "backend_vulkan.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>
#include <unistd.h>

/* ---- opaque handle definitions (ABI types are forward-declared in the
 *      header; the engine only passes these pointers around) ---- */

typedef struct Dsv4CudaTensor {
    int device;
    long long bytes;   /* uploaded weight bytes (real accounting) */
    int fmt;           /* 8=fp8-e4m3, 9=fp8-bf16-rounded, 4=fp4, 16=bf16, 32=f32, 0=f32-as-raw */
    int O, I;
    ColiVkTensor *vk;  /* the arena handle (M2-3: real uploads; M3a compute) */
    /* M3a: per-op gating group (TEST §0.5 COLI_DSV4_VK_OPS). Set by
     * ds4vk_tensor_set_op at upload (the engine knows the tensor name; the
     * ABI does not carry one). */
    int op;
    /* M3a grouped matvec (wo_a): per-group sub-tensors sliced from the full
     * block-diagonal upload on first grouped use (groups of o_rank x
     * group_width stacked vertically — the resident layout, PLAN G7). */
    int groups;
    ColiVkTensor **gvk;
    /* M3e-2: cached HOST copy of small f32 tensors (scale/base/norm/bias).
     * The arena mapping is write-combined ReBAR; scalar CPU reads from it are
     * uncached and cost ~0.8 us each (measured: 4096 norm reads = 3.2 ms/call
     * in the mHC pre host path). Uploads <= 256 KiB get a plain malloc'd copy
     * so the mHC/route host math reads cached RAM instead. */
    float *whost_cache;
} Dsv4CudaTensor;

/* Batch mHC constants (M4-1): the batch ABI carries no eps/iters — mirror the
 * CUDA tier's hardcoded values = the DeepSeek-V4-Flash-0731 config the prefill
 * path runs under (rms_norm_eps 1e-6, hc_eps 1e-6, hc_sinkhorn_iters 20,
 * post_mult 2.0). The engine wrapper restricts the batch path to
 * hc==4 && hidden==4096, so these are the only values that occur. */
#define DS4VK_MHC_EPS 1e-6f
#define DS4VK_MHC_POST_MULT 2.0f
enum { DS4VK_MHC_BATCH_ITERS = 20 };

/* ---- per-op gating (TEST §0.5) ----
 * COLI_DSV4_VK_OPS=qkv,wo,route,head,shared — comma list; unset/empty =
 * every hooked op runs on the GPU. A disabled op returns failure, so the
 * engine's per-op dispatch falls back to the CPU reference (D6) — this turns
 * the whole of L3 into per-op A/B isolation (flip one group at a time and
 * compare against the all-CPU run). The groups mirror the M3a op surface:
 * qkv (wq_a/wq_b/wkv matvecs), wo (wo_a grouped + wo_b), route (ffn gate
 * logits + top-6), head (head.weight bf16 matvec + argmax), shared
 * (ffn.shared_experts.w1/w2/w3 fp8 matvecs), attn (the decode attention
 * core over the device KV ring). */
typedef enum {
    DS4VK_OP_NONE = 0,
    DS4VK_OP_QKV,
    DS4VK_OP_WO,
    DS4VK_OP_ROUTE,
    DS4VK_OP_HEAD,
    DS4VK_OP_SHARED,
    DS4VK_OP_ATTN,
    DS4VK_OP_MHC,
    DS4VK_OP_COMP,
    DS4VK_OP_EXPERT,   /* M5-1a: routed-expert fp4 group (mirrors + group) */
    DS4VK_OP_COUNT
} Ds4vkOp;

static const char *ds4vk_op_name(Ds4vkOp op) {
    switch (op) {
    case DS4VK_OP_QKV: return "qkv";
    case DS4VK_OP_WO: return "wo";
    case DS4VK_OP_ROUTE: return "route";
    case DS4VK_OP_HEAD: return "head";
    case DS4VK_OP_SHARED: return "shared";
    case DS4VK_OP_ATTN: return "attn";
    case DS4VK_OP_MHC: return "mhc";
    case DS4VK_OP_COMP: return "comp";
    case DS4VK_OP_EXPERT: return "expert";
    default: return "none";
    }
}

static Ds4vkOp ds4vk_op_from_name(const char *name) {
    if (!name || !*name) return DS4VK_OP_NONE;
    for (int op = 1; op < DS4VK_OP_COUNT; op++)
        if (strcmp(name, ds4vk_op_name((Ds4vkOp)op)) == 0) return (Ds4vkOp)op;
    return DS4VK_OP_NONE;
}

/* Parsed once: bitmask of enabled op groups. -1 = env unset/empty (ALL hooked
 * ops enabled — TEST §0.5); otherwise the bitmask (0 = none valid -> all CPU). */
static int ds4vk_op_mask(void) {
    static int mask = -2;
    if (mask == -2) {
        const char *env = getenv("COLI_DSV4_VK_OPS");
        if (!env || !*env) {
            mask = -1;
        } else {
            mask = 0;
            char *copy = strdup(env);
            if (copy) {
                for (char *tok = strtok(copy, ","); tok; tok = strtok(NULL, ",")) {
                    while (*tok == ' ' || *tok == '\t') tok++;
                    Ds4vkOp op = ds4vk_op_from_name(tok);
                    if (op != DS4VK_OP_NONE) mask |= 1 << op;
                }
                free(copy);
            }
        }
    }
    return mask;
}

int ds4vk_tensor_set_op(Dsv4CudaTensor *t, const char *group) {
    if (!t) return 0;
    t->op = ds4vk_op_from_name(group);
    return t->op != DS4VK_OP_NONE;
}

/* The op gate: enabled when COLI_DSV4_VK_OPS is unset/empty (default = all
 * hooked) or the tensor's group is listed. A gated-off op returns 0 from the
 * compute entry so the engine falls back to CPU (D6). */
static int ds4vk_op_enabled(const Dsv4CudaTensor *t) {
    if (!t) return 0;
    int mask = ds4vk_op_mask();
    if (mask < 0) return t->op != DS4VK_OP_NONE;
    return t->op != DS4VK_OP_NONE && ((mask >> t->op) & 1);
}

/* The attention core has no per-tensor handle (it runs over the device KV
 * ring) — gate its group directly. */
static int ds4vk_op_enabled_g(void) {
    int mask = ds4vk_op_mask();
    if (mask < 0) return 1;
    return (mask >> DS4VK_OP_ATTN) & 1;
}

/* mHC post has no weight handle either (post/comb ride in the state
 * activation) — gate its group directly, like the attention core. */
static int ds4vk_op_enabled_g_mhc(void) {
    int mask = ds4vk_op_mask();
    if (mask < 0) return 1;
    return (mask >> DS4VK_OP_MHC) & 1;
}

/* The routed-expert group (M5-1a) runs over the mirror handles directly
 * (no single tensor to tag) — gate its group like attention/mHC. */
static int ds4vk_op_enabled_g_expert(void) {
    int mask = ds4vk_op_mask();
    if (mask < 0) return 1;
    return (mask >> DS4VK_OP_EXPERT) & 1;
}

/* ---- L4 fault injection (TEST.md §4, M3d-1) ----
 * COLI_DSV4_VK_FAIL=qkv,wo,route,head,shared,attn,mhc,upload — comma list;
 * every listed group is FORCED to fail (compute entries return 0), so the
 * engine's per-op dispatch falls back to CPU — and, once the D7 dense-RAM
 * drop is active, triggers the one-time reload + permanent-CPU mode (no
 * flip-flopping). Unset/empty = no faults. "upload" is a pseudo-group that
 * fails the upload entries (L4 case 4: graceful degrade, no boot hang). */
#define DS4VK_FAIL_UPLOAD (1 << DS4VK_OP_COUNT)
static int ds4vk_fault_mask(void) {
    static int mask = -2;
    if (mask == -2) {
        const char *env = getenv("COLI_DSV4_VK_FAIL");
        if (!env || !*env) {
            mask = 0;
        } else {
            mask = 0;
            char *copy = strdup(env);
            if (copy) {
                for (char *tok = strtok(copy, ","); tok;
                     tok = strtok(NULL, ",")) {
                    while (*tok == ' ' || *tok == '\t') tok++;
                    if (!strcmp(tok, "upload")) {
                        mask |= DS4VK_FAIL_UPLOAD;
                        continue;
                    }
                    Ds4vkOp op = ds4vk_op_from_name(tok);
                    if (op != DS4VK_OP_NONE) mask |= 1 << op;
                }
                free(copy);
            }
        }
    }
    return mask;
}
static int ds4vk_fault_hit(const Dsv4CudaTensor *t) {
    int mask = ds4vk_fault_mask();
    if (!mask) return 0;
    return t && t->op > DS4VK_OP_NONE && t->op < DS4VK_OP_COUNT &&
           ((mask >> t->op) & 1);
}
static int ds4vk_fault_hit_g(int group) {
    int mask = ds4vk_fault_mask();
    if (!mask) return 0;
    return group > DS4VK_OP_NONE && group < DS4VK_OP_COUNT &&
           ((mask >> group) & 1);
}
static int ds4vk_fault_hit_upload(void) {
    return (ds4vk_fault_mask() & DS4VK_FAIL_UPLOAD) != 0;
}
static void ds4vk_fault_log(const char *op) {
    fprintf(stderr, "[VK dsv4] fault-injection: op=%s forced failure\n", op);
}

/* ---- per-op wall timing (VK_PROF=1, the backend's own gate) ---- */
static double g_op_ms[DS4VK_OP_COUNT];
static long g_op_n[DS4VK_OP_COUNT];
static void ds4vk_prof_tick(const Dsv4CudaTensor *t, double t0) {
    const char *prof = getenv("VK_PROF");
    if (!prof || atoi(prof) == 0) return;
    int op = t && t->op > DS4VK_OP_NONE && t->op < DS4VK_OP_COUNT ? t->op : 0;
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    double now = ts.tv_sec + ts.tv_nsec / 1e9;
    g_op_ms[op] += (now - t0) * 1000.0;
    if ((++g_op_n[op] & 4095) == 0)
        fprintf(stderr, "[VK_PROF dsv4] op=%s n=%ld total=%.0f ms\n",
                ds4vk_op_name((Ds4vkOp)op), g_op_n[op], g_op_ms[op]);
}

typedef struct Dsv4CudaActivation {
    int device;
    long long elements;
    float *data;       /* host copy (M3a: route reads it directly; batched ops M4) */
} Dsv4CudaActivation;

typedef struct Dsv4CudaKvCache {
    int device;
} Dsv4CudaKvCache;

typedef struct Dsv4CudaExpertSet {
    int device;
} Dsv4CudaExpertSet;

typedef struct Dsv4CudaGraph {
    int device;
} Dsv4CudaGraph;

static const char *stub_fmt_name(int fmt) {
    switch (fmt) {
    case 0:  return "f32";
    case 8:  return "fp8-e4m3";
    case 9:  return "fp8-bf16";
    case 4:  return "fp4";
    case 7:  return "fp4-mxfp4";   /* GLM mxfp4 (per-group scale) */
    case 10: return "fp4-elem";     /* M5-1a: dsv4 fp4, per-element scale */
    case 16: return "bf16";
    case 32: return "f32";
    default: return "?";
    }
}

/* ---- fp8 exactness replica (TEST L1 bitwise oracle, M2-4) -----------------
 * ds4vk_fp8_ref_matmul is a serial-order C mirror of the CUDA
 * dsv4_cuda_fp8_ref_matmul kernels (backend_cuda_dsv4.cu:982/1004): the
 * row-major path accumulates fp32 SEQUENTIALLY within each 128-block
 * (__fadd_rn/__fmul_rn, no FMA) and fp64 across blocks (__dadd_rn/__dmul_rn,
 * __double2float_rn at the end); the rows8-packed path accumulates
 * (x*v)*scale fp32 sequentially over ALL columns. Both are exactly the CUDA
 * source semantics, so the replica is the BITWISE reference for the
 * production shader (which is fp32 tree-order, thresholded against it) and a
 * bitwise cross-check against the engine's own CPU kernels (the AVX2 rows8
 * compute uses the same (x*v)*scale order). fp-contract is disabled so a*b+c
 * never fuses into FMA (CUDA's __fadd_rn/__fmul_rn are explicitly unfused). */
#pragma GCC push_options
#pragma GCC optimize ("fp-contract=off")
static float ds4vk_e4m3(uint8_t b) {   /* mirror of e4m3() / coli_e4m3fn_decode */
    int sign = b >> 7, e = (b >> 3) & 15, m = b & 7;
    float v;
    if (!e) v = ldexpf((float)m, -9);
    else if (e == 15) v = m == 7 ? NAN : ldexpf(1.f + m / 8.f, 8);
    else v = ldexpf(1.f + m / 8.f, e - 7);
    return sign ? -v : v;
}
int ds4vk_fp8_ref_matmul(const uint8_t *w, const float *bscale,
                         int rows, int cols, int packed_rows8,
                         const float *x, int tokens, float *y) {
    if (!w || !bscale || !x || !y || rows < 1 || cols < 1 || cols % 128 ||
        tokens < 1 || (packed_rows8 && rows % 8)) return 0;
    if (packed_rows8) {
        int nblk = cols / 128;
        for (int t = 0; t < tokens; t++) {
            const float *xs = x + (long long)t * cols;
            for (int o = 0; o < rows; o++) {
                const float *scl = bscale + (long long)(o / 128) * nblk;
                long long tile = o >> 3; int r = o & 7;
                float sum = 0.0f;
                for (int base = 0; base < cols; base += 128) {
                    float sc = scl[base / 128];
                    for (int i = base; i < base + 128; i++) {
                        float v = ds4vk_e4m3(w[((tile * cols) + i) * 8 + r]);
                        sum = sum + (xs[i] * v) * sc;
                    }
                }
                y[(long long)t * rows + o] = sum;
            }
        }
    } else {
        int nblk = (cols + 127) / 128;
        for (int t = 0; t < tokens; t++) {
            const float *xs = x + (long long)t * cols;
            for (int o = 0; o < rows; o++) {
                const uint8_t *wr = w + (long long)o * cols;
                const float *scl = bscale + (long long)(o / 128) * nblk;
                double a = 0.0;
                for (int bi = 0; bi * 128 < cols; bi++) {
                    int base = bi * 128;
                    int blen = (cols - base < 128) ? cols - base : 128;
                    float acc = 0.0f;
                    for (int i = base; i < base + blen; i++)
                        acc = acc + ds4vk_e4m3(wr[i]) * xs[i];
                    a = a + (double)acc * (double)scl[bi];
                }
                y[(long long)t * rows + o] = (float)a;
            }
        }
    }
    return 1;
}
#pragma GCC pop_options

static void stub_upload_log(const char *what, int device, int O, int I,
                            int fmt, long long bytes) {
    fprintf(stderr, "v4_gpu vk upload %s device=%d rows=%d cols=%d fmt=%s "
                    "bytes=%lld\n",
            what, device, O, I, stub_fmt_name(fmt), bytes);
}

/* Mirror the engine's v4_vk_resolve_spv (deepseek_v4.c, COLI_VULKAN probe):
 * COLI_VK_SHADERS may be qmatmul.spv itself or a directory; unset, look
 * alongside the binary (<exedir>/shaders/) before the CWD fallback. */
static const char *ds4vk_resolve_spv(char *buf, size_t n) {
    const char *env = getenv("COLI_VK_SHADERS");
    struct stat st;
    if (env && *env) {
        if (!stat(env, &st) && S_ISDIR(st.st_mode)) {
            snprintf(buf, n, "%s/qmatmul.spv", env);
            return buf;
        }
        return env;
    }
#ifdef __linux__
    ssize_t k = readlink("/proc/self/exe", buf, n - 1);
    if (k > 0) {
        buf[k] = 0;
        char *sl = strrchr(buf, '/');
        if (sl && (size_t)(sl + 1 - buf) + sizeof("shaders/qmatmul.spv") <= n) {
            strcpy(sl + 1, "shaders/qmatmul.spv");
            if (!stat(buf, &st)) return buf;
        }
    }
#endif
    return "shaders/qmatmul.spv";
}

/* ---- lifecycle / identity ---- */

int dsv4_cuda_init(const int *devices, int count) {
    int device = (devices && count > 0) ? devices[0] : 0;
    char spv[1024];
    const char *path = ds4vk_resolve_spv(spv, sizeof(spv));
    if (!coli_vk_init(path)) {
        fprintf(stderr, "v4_gpu vk backend=init-failed (%s); continuing-CPU "
                        "(need libvulkan + committed shaders; set COLI_VK_SHADERS)\n",
                        path);
        return 0;
    }
    fprintf(stderr,
            "v4_gpu vk backend=vulkan-dsv4 device=%d (M2-3: dense set uploads "
            "live; compute ops land M3a, fall back to CPU per D6)\n",
            device);
    return 1;
}

void dsv4_cuda_shutdown(void) {
    const char *prof = getenv("VK_PROF");
    if (prof && atoi(prof) != 0) {
        for (int op = 1; op < DS4VK_OP_COUNT; op++)
            if (g_op_n[op])
                fprintf(stderr, "[VK_PROF dsv4] op=%-6s n=%-6ld total=%.1f ms avg=%.3f ms\n",
                        ds4vk_op_name((Ds4vkOp)op), g_op_n[op], g_op_ms[op],
                        g_op_ms[op] / (double)g_op_n[op]);
    }
    coli_vk_shutdown();
}

int dsv4_cuda_backend_arch_ok(int device) { (void)device; return coli_vk_available(); }

const char *dsv4_cuda_backend_name(void) { return "vulkan-dsv4"; }

long long dsv4_cuda_mem_free_mb(int device) {
    (void)device;
    double used = 0, budget = 0;
    if (coli_vk_mem_budget(&used, &budget) && budget > 0) {
        /* coli_vk_mem_budget reports GB (heapUsage/1e9); convert to MiB. A
         * bytes/GB mixup here reported ~0 MB free and made the expert mirror
         * cache's VRAM growth guard (v4_gpu_expert_attach_cached, reserve 600
         * MB) recycle ONE slot for every expert — all routed mirrors shared a
         * single tensor (M5-1a parity run: every expert produced the same
         * gate/up/down result). */
        double free_gb = budget - used;
        return free_gb > 0 ? (long long)(free_gb * 1024.0) : 0;
    }
    return 16128; /* RX 6900 XT VRAM; VK_EXT_memory_budget absent */
}

/* ---- uploads (M2-3: real; the engine's per-layer plan dump names them) ----
 * Both fp8 flavours land in the same fmt=8 arena layout (raw e4m3 bytes +
 * UE8M0 codes expanded to fp32 at upload). fmt=9 (fp8-bf16) only differs in
 * matmul-time bf16 rounding for the batched attention output path (M4); the
 * decode grouped matvec uses fmt-8 numerics either way. */

int dsv4_cuda_upload_fp8(Dsv4CudaTensor **t, const uint8_t *w,
                         const uint8_t *scale, int O, int I, int device) {
    if (!t) return 0;
    if (ds4vk_fault_hit_upload()) {
        ds4vk_fault_log("upload");
        *t = NULL;
        return 0;
    }
    ColiVkTensor *vk = NULL;
    int ok = device == 0
        ? coli_vk_tensor_ensure(&vk, w, (const float *)scale, 8, I, O, 0)
        : coli_vk_dev2_available()
            ? coli_vk_tensor_ensure2(&vk, w, (const float *)scale, 8, I, O, 0)
            : 0;
    if (!ok) { *t = NULL; return 0; }
    Dsv4CudaTensor *h = calloc(1, sizeof(*h));
    if (!h) { coli_vk_tensor_free(vk); *t = NULL; return 0; }
    h->device = device; h->fmt = 8; h->O = O; h->I = I;
    h->vk = vk; h->bytes = (long long)coli_vk_tensor_bytes(vk);
    *t = h;
    stub_upload_log("fp8", device, O, I, 8, h->bytes);
    return 1;
}

int dsv4_cuda_upload_fp8_bf16(Dsv4CudaTensor **t, const uint8_t *w,
                              const uint8_t *scale, int O, int I, int device) {
    if (!t) return 0;
    if (ds4vk_fault_hit_upload()) {
        ds4vk_fault_log("upload");
        *t = NULL;
        return 0;
    }
    ColiVkTensor *vk = NULL;
    int ok = device == 0
        ? coli_vk_tensor_ensure(&vk, w, (const float *)scale, 8, I, O, 0)
        : coli_vk_dev2_available()
            ? coli_vk_tensor_ensure2(&vk, w, (const float *)scale, 8, I, O, 0)
            : 0;
    if (!ok) { *t = NULL; return 0; }
    Dsv4CudaTensor *h = calloc(1, sizeof(*h));
    if (!h) { coli_vk_tensor_free(vk); *t = NULL; return 0; }
    h->device = device; h->fmt = 9; h->O = O; h->I = I;
    h->vk = vk; h->bytes = (long long)coli_vk_tensor_bytes(vk);
    *t = h;
    stub_upload_log("fp8-bf16", device, O, I, 9, h->bytes);
    return 1;
}

/* ---- M5-1a: fp4 expert tier (routed experts, decode path) ----
 * The dsv4 routed-expert format (COLI_TENSOR_FP4_NATIVE_BLOCK +
 * COLI_SCALE_UE8M0, deepseek_v4.c coli_fp4_matvec_ref): e2m1 nibbles packed
 * 2/byte (LOW nibble = even column, bit3 = sign, bits 0..2 index
 * {0,.5,1,1.5,2,3,4,6}) + one UE8M0 byte per 32-input group. That is exactly
 * the mxfp4 (fmt=7) layout the shared qmatmul.comp already decodes (mx4() +
 * per-gs scale, PLAN: GLM's Kimi K3 tier ships the same format), so the fp4
 * tier needs NO shader change — the upload expands UE8M0 -> f32 (the shader
 * is float-only) and stores a fmt=7 tensor with gs=32, mirroring the fp8
 * tier's E8M0 expansion (D4). The matvec chain replicates coli_fp4_matvec_ref:
 * activation QDQ (block=128) + fmt=7 matmul on the dequantized activations. */

/* UE8M0 scale: replicate the engine's e8 TABLE semantics exactly (the CPU
 * fp4 reference — coli_fp4_matvec_rows16_order / rows16_v10 — decodes via
 * coli_e8m0_table: 0xff -> NaN, else 2^(value-127)). NOT the quant.h
 * mx4_scale bit trick (0 -> +0, 255 -> +inf): the real checkpoint's expert
 * gate scales carry codes 0 and 255, and the swiglu fminf/fmaxf clamps
 * REPAIR the NaN rows (fminf(NaN, limit)=limit) while +inf survives into
 * -inf -> NaN poison. This divergence was found on the real checkpoint
 * (M5-1a, first parity run); the L2 suite now pins the 0/255 codes. */
static float ds4vk_ue8m0(uint8_t s) {
    if (s == 0xff) return NAN;
    return ldexpf(1.0f, (int)s - 127);
}

int dsv4_cuda_upload_fp4(Dsv4CudaTensor **t, const uint8_t *w,
                         const uint8_t *scale, int O, int I, int device) {
    /* Real fp4 upload (M5-1a): packed e2m1 nibbles (I/2 bytes/row) + UE8M0
     * per-32-group scales expanded to f32 at upload, fmt=10 gs=32. fmt=10 is
     * the per-ELEMENT scale shader branch (M5-1a parity fix) — the engine's
     * rows16 kernel multiplies the group scale per element, which is
     * load-bearing for the extreme UE8M0 codes (0xfe/0xff) the real
     * checkpoint's gate scales carry. The mirror cache
     * (v4_gpu_expert_attach_cached_ex) drives the shape; any failure leaves
     * *t NULL and the expert stays on the CPU fp4 path (D6). */
    if (!t || !w || !scale || O < 1 || I < 1 || I % 32) return 0;
    if (ds4vk_fault_hit_upload()) {
        ds4vk_fault_log("upload");
        *t = NULL;
        return 0;
    }
    int ng = (I + 31) / 32;
    size_t ns = (size_t)O * (size_t)ng;
    float *s = malloc(ns * sizeof(float));
    if (!s) return 0;
    for (size_t i = 0; i < ns; i++) s[i] = ds4vk_ue8m0(scale[i]);
    ColiVkTensor *vk = NULL;
    int ok = device == 0
        ? coli_vk_tensor_ensure(&vk, w, s, 10, I, O, 32)
        : coli_vk_dev2_available()
            ? coli_vk_tensor_ensure2(&vk, w, s, 10, I, O, 32)
            : 0;
    free(s);
    if (!ok) { *t = NULL; return 0; }
    Dsv4CudaTensor *h = calloc(1, sizeof(*h));
    if (!h) { coli_vk_tensor_free(vk); *t = NULL; return 0; }
    h->device = device; h->fmt = 10; h->O = O; h->I = I;
    h->vk = vk; h->bytes = (long long)coli_vk_tensor_bytes(vk);
    h->op = DS4VK_OP_EXPERT;   /* COLI_DSV4_VK_OPS=expert gate (M5-1a) */
    *t = h;
    stub_upload_log("fp4", device, O, I, 10, h->bytes);
    return 1;
}

int dsv4_cuda_tensor_refill_fp4(Dsv4CudaTensor *t, const uint8_t *w,
                                const uint8_t *scale, int O, int I, int sync) {
    /* Same-shape in-place refill of an fp4 mirror (LRU recycle in
     * v4_gpu_expert_attach_cached_ex): overwrite the arena row bytes + re-
     * expand the UE8M0 scales. Host-visible arena writes are visible to the
     * device on the next queue submit, so the caller's following compute is
     * always ordered after the new bytes (sync is a no-op — the VK tier has
     * no async stream). */
    (void)sync;
    if (!t || !t->vk || !w || !scale || t->fmt != 10) return 0;
    ColiVkTensor *vk = t->vk;
    if (vk->O != O || vk->I != I) return 0;
    size_t stride = (size_t)vk->rowWords * 4;      /* padded row bytes */
    size_t cpu_rb = (size_t)(I + 1) / 2;           /* packed nibbles/row */
    int ng = (I + 31) / 32;
    uint8_t *wptr = (uint8_t *)vk->whost;
    float *sptr = (float *)vk->shost;
    if (!wptr || !sptr) return 0;
    for (int o = 0; o < O; o++) {
        memcpy(wptr + (size_t)o * stride, w + (size_t)o * cpu_rb, cpu_rb);
        for (int g = 0; g < ng; g++)
            sptr[(size_t)o * ng + (size_t)g] =
                ds4vk_ue8m0(scale[(size_t)o * ng + (size_t)g]);
    }
    return 1;
}

int dsv4_cuda_upload_bf16(Dsv4CudaTensor **t, const uint16_t *w, int O, int I,
                          int device) {
    /* M3a: real bf16 upload (fmt=16, unquantized — the head.weight matvec
     * decodes bf16 in-shader; the compressor/indexer mirrors stay deferred
     * (M4 batched path). The head consumes the raw resident bf16 bytes, so
     * upload is byte-identical and the decode happens at matvec time. */
    if (!t || !w) return 0;
    if (ds4vk_fault_hit_upload()) {
        ds4vk_fault_log("upload");
        *t = NULL;
        return 0;
    }
    ColiVkTensor *vk = NULL;
    int ok = device == 0
        ? coli_vk_tensor_ensure(&vk, w, NULL, 16, I, O, 0)
        : coli_vk_dev2_available()
            ? coli_vk_tensor_ensure2(&vk, w, NULL, 16, I, O, 0)
            : 0;
    if (!ok) { *t = NULL; return 0; }
    Dsv4CudaTensor *h = calloc(1, sizeof(*h));
    if (!h) { coli_vk_tensor_free(vk); *t = NULL; return 0; }
    h->device = device; h->fmt = 16; h->O = O; h->I = I;
    h->vk = vk; h->bytes = (long long)coli_vk_tensor_bytes(vk);
    h->op = DS4VK_OP_HEAD;   /* only the output head uses bf16 in v1 */
    *t = h;
    stub_upload_log("bf16", device, O, I, 16, h->bytes);
    return 1;
}

int dsv4_cuda_upload_f32(Dsv4CudaTensor **t, const float *w, int O, int I,
                         int device) {
    /* M3a: real f32 upload (fmt=0, one f32 per word in the matmul shader).
     * Consumers in v1: the route gate (256 x hidden) + bias (O=256, I=1) and
     * the mHC fn/scale/base mirrors (batched path, M4). The engine decodes
     * the resident bf16 gate to f32 at upload (v4_gpu_upload_gate), so the
     * bytes here are the exact f32 mirror — no conversion. */
    if (!t || !w) return 0;
    if (ds4vk_fault_hit_upload()) {
        ds4vk_fault_log("upload");
        *t = NULL;
        return 0;
    }
    ColiVkTensor *vk = NULL;
    int ok = device == 0
        ? coli_vk_tensor_ensure(&vk, w, NULL, 0, I, O, 0)
        : coli_vk_dev2_available()
            ? coli_vk_tensor_ensure2(&vk, w, NULL, 0, I, O, 0)
            : 0;
    if (!ok) { *t = NULL; return 0; }
    Dsv4CudaTensor *h = calloc(1, sizeof(*h));
    if (!h) { coli_vk_tensor_free(vk); *t = NULL; return 0; }
    h->device = device; h->fmt = 0; h->O = O; h->I = I;
    h->vk = vk; h->bytes = (long long)coli_vk_tensor_bytes(vk);
    h->op = DS4VK_OP_ROUTE;   /* route gate/bias are the only f32 uploads in v1 */
    /* M3e-2: cache small f32 uploads' host bytes (WC-mapping read fix). The
     * fn tensor (1.5 MiB) and gate (4 MiB) are read only by the GPU, so they
     * are skipped; scale/base/norm/bias are read by host math. */
    if ((size_t)O * (size_t)I * sizeof(float) <= 256u * 1024u) {
        h->whost_cache = malloc((size_t)O * (size_t)I * sizeof(float));
        if (h->whost_cache)
            memcpy(h->whost_cache, w, (size_t)O * (size_t)I * sizeof(float));
    }
    *t = h;
    stub_upload_log("f32", device, O, I, 0, h->bytes);
    return 1;
}

/* ---- tensor/activation handles ---- */

void dsv4_cuda_tensor_free(Dsv4CudaTensor *t) {
    if (!t) return;
    for (int g = 0; t->gvk && g < t->groups; g++)
        if (t->gvk[g]) coli_vk_tensor_free(t->gvk[g]);
    free(t->gvk);
    if (t->vk) coli_vk_tensor_free(t->vk);
    free(t->whost_cache);
    free(t);
}

long long dsv4_cuda_tensor_bytes(const Dsv4CudaTensor *t) {
    return t ? t->bytes : 0;
}

int dsv4_cuda_tensor_device(const Dsv4CudaTensor *t) {
    return t ? t->device : -1;
}

Dsv4CudaActivation *dsv4_cuda_activation_create(int device, long long elements) {
    Dsv4CudaActivation *a = calloc(1, sizeof(*a));
    if (!a) return NULL;
    a->device = device;
    a->elements = elements;
    return a;
}

void dsv4_cuda_activation_free(Dsv4CudaActivation *a) {
    if (!a) return;
    free(a->data);
    free(a);
}

int dsv4_cuda_activation_upload(Dsv4CudaActivation *a, const float *x,
                                long long elements) {
    /* M3a: host-side mirror (the decode path crosses host pointers; the
     * batched prefill ops will move the bytes to device scratch in M4).
     * Capacity semantics (M4-1 fix): a->elements is the mirror's CAPACITY
     * (the engine's v4_gpu_mhc_mirror grows it monotonically); the data
     * buffer is allocated to that capacity ONCE and never shrunk — a
     * realloc to the uploaded count allowed a later larger call to write
     * past the shrunken buffer (ASAN: heap-buffer-overflow in
     * ds4vk_mhc_sinkhorn after a 106-token post shrank the state buffer
     * that a 128-token pre then wrote into). */
    if (!a || !x || elements < 1) return 0;
    if (a->elements < elements) return 0;
    if (!a->data) {
        a->data = malloc((size_t)a->elements * sizeof(float));
        if (!a->data) return 0;
    }
    memcpy(a->data, x, (size_t)elements * sizeof(*x));
    return 1;
}

int dsv4_cuda_activation_download(float *x, const Dsv4CudaActivation *a,
                                  long long elements) {
    if (!x || !a || !a->data || elements < 1 || a->elements < elements) return 0;
    memcpy(x, a->data, (size_t)elements * sizeof(*x));
    return 1;
}

int dsv4_cuda_activation_copy(Dsv4CudaActivation *dst,
                              const Dsv4CudaActivation *src,
                              long long elements) {
    if (!dst || !src || !src->data || elements < 1 || dst->elements < elements ||
        src->elements < elements)
        return 0;
    if (!dst->data) {
        dst->data = malloc((size_t)dst->elements * sizeof(float));
        if (!dst->data) return 0;
    }
    memcpy(dst->data, src->data, (size_t)elements * sizeof(*dst->data));
    return 1;
}

int dsv4_cuda_activation_copy_range(Dsv4CudaActivation *dst,
                                    long long dst_offset,
                                    const Dsv4CudaActivation *src,
                                    long long src_offset, long long elements) {
    if (!dst || !src || !src->data || elements < 1 || dst_offset < 0 ||
        src_offset < 0 || dst->elements < dst_offset + elements ||
        src->elements < src_offset + elements)
        return 0;
    if (!dst->data) {
        dst->data = malloc((size_t)dst->elements * sizeof(float));
        if (!dst->data) return 0;
    }
    memcpy(dst->data + dst_offset, src->data + src_offset,
           (size_t)elements * sizeof(*dst->data));
    return 1;
}

int dsv4_cuda_activation_sync(const Dsv4CudaActivation *a) {
    (void)a;
    return 1;   /* host copy — nothing pending */
}

int dsv4_cuda_activation_device(const Dsv4CudaActivation *a) {
    return a ? a->device : -1;
}

int dsv4_cuda_decode_state_set(int device, int token, int position) {
    (void)device; (void)token; (void)position;
    return 0;
}

/* ---- compute ops (M3a-1: decode-path matmul family) ----
 * Each op reproduces the CPU chain the engine's per-op dispatch replaces
 * (TEST L2): the fp8 matvecs run QDQ (block=128, the `_pre` hoist contract
 * — bitwise vs coli_fp8_activation_qdq_ref, M2-2) then the fmt=8 matmul on
 * the dequantized activation (thresholded vs the fp64-across-blocks CPU
 * accumulate, M2-4); wo_a runs as `groups` independent matvecs over the
 * block-diagonal resident layout; route runs the f32 gate logits matvec
 * then the host-side top-6 (256 scores — exact same selection as the CPU
 * route, including the strict-greater tie-break); head runs the bf16 matvec
 * then a host argmax. Every entry returns 0 on ANY failure so the engine
 * falls back to the CPU reference (D6) — a wedged GPU can slow a run,
 * never corrupt it. */

/* fp8 matvec: one QDQ pass (the _pre hoist contract) + one fmt=8 matmul on
 * the dequantized activations — the exact GPU chain M2-4 pinned at global
 * maxabs/maxref 1.4e-7..1.3e-6 vs the REAL coli_fp8_matvec_ref. */
static int ds4vk_fp8_matvec(Dsv4CudaTensor *t, float *y, const float *x) {
    size_t n = (size_t)t->I;
    float *act = malloc(n * sizeof(float));
    uint8_t *actsc = malloc((n + 127) / 128);
    if (!act || !actsc) { free(actsc); free(act); return 0; }
    int ok = coli_vk_activation_qdq(act, actsc, x, 1, t->I, 128) &&
             coli_vk_matmul(&t->vk, y, act, NULL, NULL, 8, 1, t->I, t->O, 1);
    free(actsc); free(act);
    return ok;
}

/* fp4 matvec (M5-1a): replicates coli_fp4_matvec_ref — the input is
 * dynamically QDQ'd to fp8 (block=128) first, then the fmt=10 (e2m1,
 * per-element group scale) matmul consumes the dequantized activations (the
 * CPU chain's matmul_mxfp4 / rows16_order see exactly these bytes). QDQ
 * bitwise (M2-2), matmul thresholded — the same contract as the fp8 tier. */
static int ds4vk_fp4_matvec(Dsv4CudaTensor *t, float *y, const float *x) {
    size_t n = (size_t)t->I;
    float *act = malloc(n * sizeof(float));
    uint8_t *actsc = malloc((n + 127) / 128);
    if (!act || !actsc) { free(actsc); free(act); return 0; }
    int ok = coli_vk_activation_qdq(act, actsc, x, 1, t->I, 128) &&
             coli_vk_matmul(&t->vk, y, act, NULL, NULL, 10, 1, t->I, t->O, 32);
    free(actsc); free(act);
    return ok;
}

/* Grouped matvec (wo_a): the resident weight is the block-diagonal matrix
 * stacked vertically — groups of o_rank rows x group_width columns. The full
 * upload (O=groups*o_rank, I=group_width) is sliced into per-group tensors
 * on first use (the group weight/scale rows are contiguous in the arena
 * buffers); each group then runs the standard fmt=8 matvec against its own
 * input slice x + g*group_width. */
static int ds4vk_grouped_matvec(Dsv4CudaTensor *t, float *y, const float *x,
                                int groups) {
    /* M3e-3 (perf): the block-diagonal wo_a's per-group slices are read
     * DIRECTLY from the arena via offset views and ALL groups run as ONE
     * batched command buffer (1 QDQ over the packed rows + `groups` matmuls,
     * one submit + one fence) — the per-group loop was 2 submits/group (16/
     * layer for o_groups=8). The packed-row QDQ is per-row independent
     * (qdq.comp workgroup per (block,row)), so the numerics are bitwise
     * unchanged (L2 + L3 re-verified); the M3e-1 view keeps the same bytes
     * and the same shader math. */
    if (!t || !t->vk || groups < 1 || t->O % groups || t->I % groups) return 0;
    return coli_vk_matmul_grouped_batch(t->vk, y, x, groups, 8);
}

/* f32 matvec (route gate logits; also the generic fmt-0 path). */
static int ds4vk_f32_matvec(Dsv4CudaTensor *t, float *y, const float *x) {
    return coli_vk_matmul(&t->vk, y, x, NULL, NULL, 0, 1, t->I, t->O, 1);
}

/* bf16 matvec (head.weight): fmt=16 in the same shader — in-shader bf16
 * decode is bitwise (f32 bits = b<<16), accumulation thresholded. */
static int ds4vk_bf16_matvec(Dsv4CudaTensor *t, float *y, const float *x) {
    return coli_vk_matmul(&t->vk, y, x, NULL, NULL, 16, 1, t->I, t->O, 1);
}

/* The CPU route's top-6 (coli_v4_route_bf16, deepseek_v4.c): scores =
 * sqrt(softplus(logit)), selection = score + bias, pick the topk unselected
 * with STRICTLY-greater replacement (smallest index wins ties — matches
 * route_top6_serial's route_better), weights = score/total*scale. The GPU
 * only computes the 256 logits; this host-side selection is bit-exact vs
 * the CPU route given the same logits. */
static int ds4vk_route_top6(int *ids, float *weights, const float *logits,
                            const float *bias, const int *fixed, int topk,
                            float scale) {
    float scores[256], selection[256];
    for (int e = 0; e < 256; e++) {
        float l = logits[e];
        scores[e] = sqrtf(fmaxf(l, 0.0f) + log1pf(expf(-fabsf(l))));
        selection[e] = scores[e] + (bias ? bias[e] : 0.0f);
    }
    if (fixed) {
        for (int k = 0; k < topk; k++) ids[k] = fixed[k];
    } else {
        int selected[256]; memset(selected, 0, sizeof(selected));
        for (int k = 0; k < topk; k++) {
            int best = -1;
            for (int e = 0; e < 256; e++)
                if (!selected[e] && (best < 0 || selection[e] > selection[best]))
                    best = e;
            ids[k] = best; selected[best] = 1;
        }
    }
    float total = 0.0f;
    for (int k = 0; k < topk; k++) total += scores[ids[k]];
    if (!(total > 0.0f)) return 0;
    for (int k = 0; k < topk; k++) weights[k] = scores[ids[k]] / total * scale;
    return 1;
}

int dsv4_cuda_matvec(Dsv4CudaTensor *t, float *y, const float *x) {
    if (!t || !t->vk || !y || !x || !ds4vk_op_enabled(t)) return 0;
    if (ds4vk_fault_hit(t)) {
        ds4vk_fault_log(ds4vk_op_name((Ds4vkOp)t->op));
        return 0;
    }
    struct timespec ts0; clock_gettime(CLOCK_MONOTONIC, &ts0);
    double t0 = ts0.tv_sec + ts0.tv_nsec / 1e9;
    int ok = t->fmt == 8 || t->fmt == 9 ? ds4vk_fp8_matvec(t, y, x)
          : t->fmt == 10 ? ds4vk_fp4_matvec(t, y, x)   /* M5-1a: fp4 experts */
          : t->fmt == 0 ? ds4vk_f32_matvec(t, y, x)
          : t->fmt == 16 ? ds4vk_bf16_matvec(t, y, x) : 0;
    if (ok) ds4vk_prof_tick(t, t0);
    return ok;
}

int dsv4_cuda_matvec_grouped(Dsv4CudaTensor *t, float *y, const float *x,
                             int groups) {
    if (!t || !t->vk || !y || !x || groups < 1 || !ds4vk_op_enabled(t)) return 0;
    if (ds4vk_fault_hit(t)) {
        ds4vk_fault_log(ds4vk_op_name((Ds4vkOp)t->op));
        return 0;
    }
    struct timespec ts0; clock_gettime(CLOCK_MONOTONIC, &ts0);
    double t0 = ts0.tv_sec + ts0.tv_nsec / 1e9;
    int ok = (t->fmt == 8 || t->fmt == 9) ? ds4vk_grouped_matvec(t, y, x, groups)
            : groups == 1 ? dsv4_cuda_matvec(t, y, x) : 0;
    if (ok) ds4vk_prof_tick(t, t0);
    return ok;
}

int dsv4_cuda_matmul_batch(Dsv4CudaTensor *t, const Dsv4CudaActivation *input,
                           int tokens, Dsv4CudaActivation *output) {
    /* M3d-1 enabler (minimal M4 slice the D7 drop requires): the prefill
     * batch matmuls (qkv, wo_b, shared experts) must run on the GPU — their
     * CPU fallbacks read the resident fp8 weight RAM, which the D7 boot flow
     * returns to the OS after the VRAM upload; a stub here would silently
     * compute on zero pages (or force the one-time reload on every prefill
     * chunk). Same chain as the decode matvec (ds4vk_fp8_matvec): QDQ the
     * whole batch (S=tokens, block=128, the _pre hoist contract — bitwise
     * vs coli_fp8_activation_qdq_ref) then one fmt=8 matmul over all tokens
     * (M2-4-verified at S=1 and S=128). fmt=9 (fp8-bf16) computes as fmt=8
     * like the decode path (numerics verified L3). Batched
     * compressor/indexer/mHC projections stay CPU-first (their bf16 mirrors
     * are never dropped) — full M4 remains. */
    if (!t || !t->vk || !input || !input->data || tokens < 1 || !output ||
        !ds4vk_op_enabled(t))
        return 0;
    if (ds4vk_fault_hit(t)) {
        ds4vk_fault_log(ds4vk_op_name((Ds4vkOp)t->op));
        return 0;
    }
    if (t->fmt != 8 && t->fmt != 9) return 0;
    long long in_elements = (long long)tokens * t->I;
    long long out_elements = (long long)tokens * t->O;
    if (input->elements < in_elements || output->elements < out_elements)
        return 0;
    /* activations allocate their host copy lazily (the mHC path's pattern):
     * the output mirror is never uploaded, so its data must be allocated
     * here. */
    if (!output->data) {
        output->data = calloc((size_t)output->elements, sizeof(float));
        if (!output->data) return 0;
    }
    size_t act_bytes = (size_t)in_elements * sizeof(float);
    size_t sc_bytes = (size_t)tokens * (size_t)((t->I + 127) / 128);
    float *act = malloc(act_bytes);
    uint8_t *actsc = malloc(sc_bytes);
    if (!act || !actsc) { free(actsc); free(act); return 0; }
    int ok = coli_vk_activation_qdq(act, actsc, input->data, tokens, t->I, 128) &&
             coli_vk_matmul(&t->vk, output->data, act, NULL, NULL, 8, tokens,
                            t->I, t->O, 1);
    free(actsc); free(act);
    return ok;
}

int dsv4_cuda_matmul_bf16_batch(Dsv4CudaTensor *t, const float *x, int tokens,
                                float *y) {
    /* M4-1: the compressor / indexer-compressor wkv+wgate projections for the
     * prefill batch path (coli_v4_gpu_compressor_project_batch): one fmt=16
     * matmul over all tokens (bf16 weights decode in-shader — the same
     * fmt=16 branch the decode head matvec uses, bitwise decode + thresholded
     * fp32-tree accumulate). The engine tags these mirrors "comp" so the
     * COLI_DSV4_VK_OPS A/B gate isolates them; a disabled op returns 0 and
     * the engine's compressor_step runs its per-token CPU projection (D6). */
    if (!t || !t->vk || !x || tokens < 1 || !y || !ds4vk_op_enabled(t)) return 0;
    if (ds4vk_fault_hit(t)) {
        ds4vk_fault_log(ds4vk_op_name((Ds4vkOp)t->op));
        return 0;
    }
    if (t->fmt != 16) return 0;
    struct timespec ts0; clock_gettime(CLOCK_MONOTONIC, &ts0);
    double t0 = ts0.tv_sec + ts0.tv_nsec / 1e9;
    int ok = coli_vk_matmul(&t->vk, y, x, NULL, NULL, 16, tokens, t->I, t->O, 1);
    if (ok) ds4vk_prof_tick(t, t0);
    return ok;
}

int dsv4_cuda_qkv(Dsv4CudaTensor *q_a, Dsv4CudaTensor *q_norm,
                  Dsv4CudaTensor *q_b, Dsv4CudaTensor *kv, float eps,
                  float *q_out, float *kv_out, const float *x) {
    (void)q_a; (void)q_norm; (void)q_b; (void)kv; (void)eps;
    (void)q_out; (void)kv_out; (void)x;
    return 0;
}

int dsv4_cuda_wo(Dsv4CudaTensor *wo_a, Dsv4CudaTensor *wo_b, int groups,
                 float *out, const float *context) {
    (void)wo_a; (void)wo_b; (void)groups; (void)out; (void)context;
    return 0;
}

int dsv4_cuda_route(const Dsv4CudaActivation *input, Dsv4CudaTensor *gate,
                    Dsv4CudaTensor *bias, const int *fixed_ids,
                    float routed_scale, int ids[6], float weights[6]) {
    if (!input || !gate || !ids || !weights || !ds4vk_op_enabled(gate))
        return 0;
    if (ds4vk_fault_hit(gate)) {
        ds4vk_fault_log(ds4vk_op_name((Ds4vkOp)gate->op));
        return 0;
    }
    if (gate->fmt != 0 || gate->O != 256 || gate->I < 1 ||
        (size_t)gate->I > (size_t)input->elements)
        return 0;
    if (bias && (bias->fmt != 0 || bias->O < 256 || bias->I < 1)) return 0;
    struct timespec ts0; clock_gettime(CLOCK_MONOTONIC, &ts0);
    double t0 = ts0.tv_sec + ts0.tv_nsec / 1e9;
    float *logits = malloc(256 * sizeof(*logits));
    if (!logits) return 0;
    int ok = dsv4_cuda_matvec(gate, logits, (const float *)input->data)
          && ds4vk_route_top6(ids, weights, logits,
                              bias ? (bias->whost_cache
                                          ? bias->whost_cache
                                          : (const float *)coli_vk_tensor_wptr(bias->vk))
                                   : NULL,
                              fixed_ids, 6, routed_scale);
    free(logits);
    if (ok) ds4vk_prof_tick(gate, t0);
    return ok;
}

int dsv4_cuda_head_argmax(Dsv4CudaTensor *t, const float *x, int *id,
                          float *value) {
    /* The engine's CPU head path (head_argmax, deepseek_v4.c): scores[row] =
     * sum_i bf16(head[row,i]) * x[i] (fp32 accumulate), winner = strictly-
     * greatest score (first row wins ties). The GPU computes the 129280-row
     * bf16 matvec once; the 516 KB score readback + host argmax is exact. */
    if (!t || !t->vk || !x || !id || !value || t->fmt != 16 ||
        !ds4vk_op_enabled(t))
        return 0;
    if (ds4vk_fault_hit(t)) {
        ds4vk_fault_log(ds4vk_op_name((Ds4vkOp)t->op));
        return 0;
    }
    struct timespec ts0; clock_gettime(CLOCK_MONOTONIC, &ts0);
    double t0 = ts0.tv_sec + ts0.tv_nsec / 1e9;
    float *scores = malloc((size_t)t->O * sizeof(*scores));
    if (!scores) return 0;
    int ok = coli_vk_matmul(&t->vk, scores, x, NULL, NULL, 16, 1, t->I, t->O, 1);
    if (ok) {
        int winner = -1;
        float maximum = -INFINITY;
        for (int row = 0; row < t->O; row++)
            if (scores[row] > maximum) { maximum = scores[row]; winner = row; }
        ok = winner >= 0;
        if (ok) { *id = winner; *value = maximum; }
    }
    free(scores);
    if (ok) ds4vk_prof_tick(t, t0);
    return ok;
}

int dsv4_cuda_final_argmax(const Dsv4CudaActivation *residual,
                           Dsv4CudaTensor *fn, Dsv4CudaTensor *scale,
                           Dsv4CudaTensor *base, Dsv4CudaTensor *norm,
                           Dsv4CudaTensor *head, int M, int H, float eps,
                           float pre_eps, int *id, float *value) {
    /* Fused mHC-mix + rmsnorm + head matvec + argmax (the CUDA DeepGEMM
     * surface). The v1 decode path runs final_hidden (mHC mix + rmsnorm) on
     * the CPU and calls dsv4_cuda_head_argmax directly, so this fused entry
     * is not wired; keep it failing so the engine stays on its CPU path. */
    (void)residual; (void)fn; (void)scale; (void)base; (void)norm; (void)head;
    (void)M; (void)H; (void)eps; (void)pre_eps; (void)id; (void)value;
    return 0;
}

int dsv4_cuda_rmsnorm(Dsv4CudaActivation *x, Dsv4CudaTensor *weight, float eps,
                      int elements) {
    (void)x; (void)weight; (void)eps; (void)elements;
    return 0;
}

int dsv4_cuda_sparse_attn_batch(int device, const float *q, const float *vals,
                                const float *sinks, const int *meta,
                                int value_rows, int comp_base, int heads,
                                int dim, int tokens, float scale, float *out) {
    (void)device; (void)q; (void)vals; (void)sinks; (void)meta;
    (void)value_rows; (void)comp_base; (void)heads; (void)dim; (void)tokens;
    (void)scale; (void)out;
    return 0;
}

int dsv4_cuda_sparse_attn_batch_cached(int device, int layer, const float *q,
                                       const float *chunk, int chunk_start,
                                       const float *sinks, const int *meta,
                                       int abs_base, int comp_limit, int heads,
                                       int dim, int tokens, float scale,
                                       float *out) {
    /* M3b: the decode attention core — sigmoid sink + window ring + compressed
     * pool + bf16 probability/value (sparse_attn.comp), the same math as the
     * engine's coli_v4_sparse_attention_ref. Gated per-op like the matvecs
     * (COLI_DSV4_VK_OPS=attn); any failure -> the engine's CPU attention runs
     * (D6, the block at deepseek_v4.c:2151 falls through to the reference). */
    (void)device;
    if (!ds4vk_op_enabled_g()) return 0;
    if (ds4vk_fault_hit_g(DS4VK_OP_ATTN)) {
        ds4vk_fault_log("attn");
        return 0;
    }
    struct timespec ts0; clock_gettime(CLOCK_MONOTONIC, &ts0);
    double t0 = ts0.tv_sec + ts0.tv_nsec / 1e9;
    int ok = coli_vk_dsv4_attn_cached(
        layer, out, q, chunk, chunk_start, sinks, meta, abs_base,
        comp_limit, heads, dim, tokens, scale);
    if (ok) {
        Dsv4CudaTensor fake; memset(&fake, 0, sizeof(fake)); fake.op = DS4VK_OP_ATTN;
        ds4vk_prof_tick(&fake, t0);
    }
    return ok;
}

int dsv4_cuda_sparse_attn_batch_cached_idx(int device, int layer,
                                           const float *q, const float *chunk,
                                           int chunk_start, const float *sinks,
                                           const int *meta, const int *sel,
                                           int selstride, int abs_base,
                                           int comp_limit, int heads, int dim,
                                           int tokens, float scale, float *out) {
    (void)device;
    if (!ds4vk_op_enabled_g()) return 0;
    if (ds4vk_fault_hit_g(DS4VK_OP_ATTN)) {
        ds4vk_fault_log("attn");
        return 0;
    }
    struct timespec ts0; clock_gettime(CLOCK_MONOTONIC, &ts0);
    double t0 = ts0.tv_sec + ts0.tv_nsec / 1e9;
    int ok = coli_vk_dsv4_attn_cached_idx(
        layer, out, q, chunk, chunk_start, sinks, meta, sel, selstride,
        abs_base, comp_limit, heads, dim, tokens, scale);
    if (ok) {
        Dsv4CudaTensor fake; memset(&fake, 0, sizeof(fake)); fake.op = DS4VK_OP_ATTN;
        ds4vk_prof_tick(&fake, t0);
    }
    return ok;
}

int dsv4_cuda_indexer_score_batch(int device, const float *queries,
                                  const float *keys, const float *head_w,
                                  const int *counts, int tokens, int heads,
                                  int dim, int count, float *scores) {
    (void)device; (void)queries; (void)keys; (void)head_w; (void)counts;
    (void)tokens; (void)heads; (void)dim; (void)count; (void)scores;
    return 0;
}

int dsv4_cuda_fp8_ref_matmul(int device, const uint8_t *w, const float *bscale,
                             int rows, int cols, int packed_rows8,
                             const float *x, int tokens, float *y) {
    (void)device; (void)w; (void)bscale; (void)rows; (void)cols;
    (void)packed_rows8; (void)x; (void)tokens; (void)y;
    return 0;
}

int dsv4_cuda_kv_ring_append(int device, int layer, const float *rows,
                             int start_pos, int count, int window, int dim) {
    /* M3b: the per-layer device window ring (slot = position % window),
     * seeded/appended by the engine's coli_v4_gpu_kv_cache_sync/advance. */
    (void)device;
    return coli_vk_dsv4_kv_ring_append(layer, rows, start_pos, count, window, dim);
}

int dsv4_cuda_kv_comp_append(int device, int layer, const float *rows,
                             int start_idx, int count, int dim) {
    /* M3b: the per-layer append-only compressed pool (grown on demand). */
    (void)device;
    return coli_vk_dsv4_kv_comp_append(layer, rows, start_idx, count, dim);
}

/* ---- mHC pre/post (M3c-1): the per-layer hyper-connection, decode path ----
 * Each op mirrors the CPU chain the engine's block_token_impl/pipeline calls
 * (deepseek_v4.c block_token_*): normalized_hc_pre = coli_v4_hc_pre (fn
 * matvec + RMS inv + sinkhorn pre/post/comb) + bf16 round + rmsnorm with the
 * branch norm + bf16 round; coli_v4_hc_post = post[j]*x[h] +
 * comb[i*M+j]*residual[i*H+h], bf16-rounded. The fn matvec is the ONLY heavy
 * term (N x M*H f32, ~1.5 MB per layer at N=24/M=4/H=4096) — it runs on the
 * GPU through the shared fmt=0 qmatmul path (the M2-4-verified matmul
 * primitive, thresholded vs the CPU's sequential fp32, same contract as the
 * CUDA tier's split accumulation). Everything downstream (the RMS inverse,
 * the sinkhorn iterations, the pre*residual / post*residual sums, the bf16
 * rounding points, the norm rmsnorm) is host-side sequential fp32 — BITWISE
 * vs the CPU reference, exactly like the engine's own CPU code (the CUDA
 * tier computes these in shaders and is thresholded there; the VK tier
 * matches the CPU chain more faithfully, the same choice as G9's QDQ).
 * state layout matches the CUDA ABI: [post M][comb M*M][pre M]. */

/* bf16 round-to-nearest-even, mirroring coli_bf16_round (deepseek_v4.c) —
 * identical bit arithmetic including the inf/nan pass-through. */
static float ds4vk_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7f800000u) != 0x7f800000u) {
        uint32_t tie = (bits >> 16) & 1u;
        bits += 0x7fffu + tie;
    }
    bits &= 0xffff0000u;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

/* sigmoidf_stable (deepseek_v4.c MATH unit): expf in the same branches, so
 * the result is bitwise vs the engine given the same input. */
static float ds4vk_sigmoid(float value) {
    if (value >= 0.0f) {
        float decay = expf(-value);
        return 1.0f / (1.0f + decay);
    }
    float growth = expf(value);
    return growth / (1.0f + growth);
}

/* Port of coli_v4_hc_split_sinkhorn (deepseek_v4.c MATH unit) — the same
 * sequential fp32 ops in the same order (softmax per row with max
 * subtraction, +eps, one column normalize, then (iterations-1) rounds of
 * row+column normalize), so pre/post/comb are bitwise vs the CPU given the
 * same mixes/scale/base. pre_eps/post_mult follow the CUDA ABI (the CPU
 * chain calls it with hc_eps / 2.0f — the engine wrapper passes exactly
 * those, so the results are identical). M <= 16 fits the stack array (the
 * config loader validates hc_mult <= 16; the real model uses 4). */
static int ds4vk_mhc_sinkhorn(float *pre, float *post, float *comb,
                              const float *mixes, const float scale[3],
                              const float *base, int M, int iterations,
                              float pre_eps, float post_mult, float eps) {
    if (!pre || !post || !comb || !mixes || !scale || !base ||
        M < 1 || M > 16 || iterations < 1 || eps < 0.0f)
        return 0;
    for (int index = 0; index < M; index++) {
        pre[index] = ds4vk_sigmoid(mixes[index] * scale[0] + base[index]) +
                     pre_eps;
        post[index] = post_mult *
                      ds4vk_sigmoid(mixes[M + index] * scale[1] +
                                    base[M + index]);
    }
    int offset = 2 * M;
    for (int row = 0; row < M; row++) {
        float maximum = -INFINITY;
        for (int column = 0; column < M; column++) {
            int index = offset + row * M + column;
            float value = mixes[index] * scale[2] + base[index];
            comb[row * M + column] = value;
            if (value > maximum) maximum = value;
        }
        float sum = 0.0f;
        for (int column = 0; column < M; column++) {
            float value = expf(comb[row * M + column] - maximum);
            comb[row * M + column] = value;
            sum += value;
        }
        for (int column = 0; column < M; column++)
            comb[row * M + column] = comb[row * M + column] / sum + eps;
    }
    float sums[16];
    for (int column = 0; column < M; column++) {
        float sum = 0.0f;
        for (int row = 0; row < M; row++)
            sum += comb[row * M + column];
        sums[column] = sum;
    }
    for (int row = 0; row < M; row++)
        for (int column = 0; column < M; column++)
            comb[row * M + column] /= sums[column] + eps;
    for (int iteration = 1; iteration < iterations; iteration++) {
        for (int row = 0; row < M; row++) {
            float sum = 0.0f;
            for (int column = 0; column < M; column++)
                sum += comb[row * M + column];
            sums[row] = sum;
        }
        for (int row = 0; row < M; row++)
            for (int column = 0; column < M; column++)
                comb[row * M + column] /= sums[row] + eps;
        for (int column = 0; column < M; column++) {
            float sum = 0.0f;
            for (int row = 0; row < M; row++)
                sum += comb[row * M + column];
            sums[column] = sum;
        }
        for (int row = 0; row < M; row++)
            for (int column = 0; column < M; column++)
                comb[row * M + column] /= sums[column] + eps;
    }
    return 1;
}

/* Shared body of dsv4_cuda_mhc_pre/_pre_norm: fn matvec on the GPU (fmt=0),
 * then host-side inv/scale/base + sinkhorn into state (post+comb+pre) and
 * the pre*residual input. norm == NULL -> plain input (bf16 round only);
 * norm != NULL -> input_norm (bf16 round, then rmsnorm with the branch norm
 * weight, then bf16 round) — exactly normalized_hc_pre. */
static int ds4vk_mhc_pre_common(const Dsv4CudaActivation *residual,
                                Dsv4CudaTensor *fn, Dsv4CudaTensor *scale,
                                Dsv4CudaTensor *base, Dsv4CudaTensor *norm,
                                int M, int H, float rms_eps, float pre_eps,
                                float sink_eps, float post_mult,
                                int sink_iters, float norm_eps,
                                Dsv4CudaActivation *state,
                                Dsv4CudaActivation *input) {
    int N = 2 * M + M * M, MH = M * H;
    if (!residual || !residual->data || !fn || !scale || !base || !state ||
        !input || M < 1 || H < 1 || N < 1 || MH < 1 ||
        residual->elements < MH || state->elements < M + M * M + M ||
        input->elements < H ||
        fn->fmt != 0 || scale->fmt != 0 || base->fmt != 0 ||
        (norm && norm->fmt != 0) || fn->O != N || fn->I != MH ||
        scale->O * scale->I < 3 || base->O * base->I < N ||
        (norm && norm->O * norm->I < H) || !fn->vk || !ds4vk_op_enabled(fn))
        return 0;
    if (ds4vk_fault_hit(fn)) {
        ds4vk_fault_log("mhc");
        return 0;
    }
    struct timespec ts0; clock_gettime(CLOCK_MONOTONIC, &ts0);
    double t0 = ts0.tv_sec + ts0.tv_nsec / 1e9;
    /* The VK backend's activations hold a HOST copy (allocated lazily on
     * upload); the pre writes state/input directly, so ensure the buffers
     * exist when the engine did not upload them (the CUDA tier allocates
     * device storage at create — same contract). */
    if (!state->data) {
        state->data = calloc((size_t)state->elements, sizeof(float));
        if (!state->data) return 0;
    }
    if (!input->data) {
        input->data = calloc((size_t)input->elements, sizeof(float));
        if (!input->data) return 0;
    }
    /* fn matvec: mix_raw[n] = sum_i fn[n][i]*residual[i] (GPU, fp32
     * tree/warp order — the same contract as the M2-4 matmul; the CUDA tier
     * splits the same sum over 8 parts). */
    float *mixes = malloc((size_t)N * sizeof(*mixes));
    if (!mixes) return 0;
    int ok = coli_vk_matmul(&fn->vk, mixes, residual->data, NULL, NULL, 0, 1,
                            MH, N, 1);
    if (ok) {
        /* RMS inverse + scale/base, bitwise vs coli_v4_hc_pre (sequential
         * fp32 mean-square, 1/sqrtf, ((sum*inv)*scale[group])+base). */
        const float *sc = scale->whost_cache
                             ? scale->whost_cache
                             : coli_vk_tensor_wptr(scale->vk);
        const float *bs = base->whost_cache
                            ? base->whost_cache
                            : coli_vk_tensor_wptr(base->vk);
        if (!sc || !bs) {
            ok = 0;
        } else {
            float mean_square = 0.0f;
            for (int i = 0; i < MH; i++)
                mean_square += residual->data[i] * residual->data[i];
            float inverse_rms = 1.0f / sqrtf(mean_square / MH + rms_eps);
            /* coli_v4_hc_pre: mixes[row] = sum * inverse_rms (NO scale/base
             * yet — the sinkhorn applies them, exactly like the CPU chain). */
            for (int n = 0; n < N; n++)
                mixes[n] = mixes[n] * inverse_rms;
            float *post = state->data;
            float *comb = state->data + M;
            float *pre = state->data + M + M * M;
            ok = ds4vk_mhc_sinkhorn(pre, post, comb, mixes, sc, bs, M,
                                    sink_iters, pre_eps, post_mult,
                                    sink_eps);
        }
    }
    if (ok) {
        /* input[h] = bf16(sum_i pre[i]*residual[i*H+h]) — sequential fp32 in
         * the CPU's order (coli_v4_hc_pre), then the bf16 round point. */
        const float *pre = state->data + M + M * M;
        float *reduced = input->data;
        for (int h = 0; h < H; h++) {
            float v = 0.0f;
            for (int i = 0; i < M; i++)
                v += pre[i] * residual->data[(size_t)i * H + h];
            reduced[h] = ds4vk_bf16(v);
        }
        if (norm) {
            /* normalized_hc_pre's rmsnorm: mean-square over the bf16-rounded
             * reduced values, inv = 1/sqrt(ss/H + norm_eps), out =
             * bf16(reduced*inv*norm) — same ops as coli_v4_rmsnorm +
             * coli_bf16_round_array. */
            const float *w = norm->whost_cache
                                ? norm->whost_cache
                                : coli_vk_tensor_wptr(norm->vk);
            if (!w) {
                ok = 0;
            } else {
                float mean_square = 0.0f;
                for (int h = 0; h < H; h++)
                    mean_square += reduced[h] * reduced[h];
                float inverse_rms = 1.0f / sqrtf(mean_square / H + norm_eps);
                for (int h = 0; h < H; h++)
                    reduced[h] = ds4vk_bf16(reduced[h] * inverse_rms * w[h]);
            }
        }
    }
    free(mixes);
    if (ok) ds4vk_prof_tick(fn, t0);
    return ok;
}

int dsv4_cuda_mhc_pre(const Dsv4CudaActivation *residual, Dsv4CudaTensor *fn,
                      Dsv4CudaTensor *scale, Dsv4CudaTensor *base, int M, int H,
                      float rms_eps, float pre_eps, float sink_eps,
                      float post_mult, int sink_iters, Dsv4CudaActivation *state,
                      Dsv4CudaActivation *input) {
    /* plain mHC pre: the raw input (bf16-rounded pre*residual), no norm —
     * the CUDA mhc_input contract. The v1 decode path uses the _norm
     * variant (normalized_hc_pre); this entry exists for ABI completeness
     * and the CUDA-test surface. pre_eps/post_mult mirror the CUDA coeff
     * kernel's handling (post_mult multiplies post; pre_eps is unused by
     * the CPU chain — sigmoid + hc_eps). */
    return ds4vk_mhc_pre_common(residual, fn, scale, base, NULL, M, H,
                                rms_eps, pre_eps, sink_eps, post_mult,
                                sink_iters, 0.0f, state, input);
}

int dsv4_cuda_mhc_pre_norm(const Dsv4CudaActivation *residual,
                           Dsv4CudaTensor *fn, Dsv4CudaTensor *scale,
                           Dsv4CudaTensor *base, Dsv4CudaTensor *norm, int M,
                           int H, float rms_eps, float pre_eps, float sink_eps,
                           float post_mult, int sink_iters, float norm_eps,
                           Dsv4CudaActivation *state, Dsv4CudaActivation *input) {
    return ds4vk_mhc_pre_common(residual, fn, scale, base, norm, M, H,
                                rms_eps, pre_eps, sink_eps, post_mult,
                                sink_iters, norm_eps, state, input);
}

/* M4-1: shared body of dsv4_cuda_mhc_pre_norm_batch / dsv4_cuda_mhc_pre_batch
 * — one fn matvec over ALL tokens (fmt=0, S=tokens), then the per-token
 * host-side chain exactly like the single-token ds4vk_mhc_pre_common: RMS
 * inverse (sequential mean-square over the token's M*H slice), scale/base +
 * sinkhorn, pre*residual sum + bf16 round, and (norm != NULL) the branch-
 * norm rmsnorm + bf16 round — bitwise vs the engine's per-token
 * normalized_hc_pre. State layout matches the CUDA batch contract (header
 * comment on dsv4_cuda_mhc_post_pre_norm_batch): post_mix[tokens][M] then
 * comb_mix[tokens][M*M]; the pre array is NOT stored (the post call reads
 * only post/comb) — the input computation uses a local copy. M is not in the
 * ABI; it is derived from the fn output rows N = 2M + M^2 and re-verified
 * against the residual/state element counts. */
static int ds4vk_mhc_pre_batch_common(const Dsv4CudaActivation *residual,
                                      Dsv4CudaTensor *fn, Dsv4CudaTensor *scale,
                                      Dsv4CudaTensor *base, Dsv4CudaTensor *norm,
                                      int tokens, int H, float rms_eps,
                                      float pre_eps, float sink_eps,
                                      float post_mult, int sink_iters,
                                      float norm_eps,
                                      Dsv4CudaActivation *state,
                                      Dsv4CudaActivation *input) {
    if (!residual || !residual->data || !fn || !scale || !base || !state ||
        !input || tokens < 1 || H < 1 || !fn->vk || !ds4vk_op_enabled(fn))
        return 0;
    if (ds4vk_fault_hit(fn)) {
        ds4vk_fault_log("mhc");
        return 0;
    }
    if (fn->fmt != 0 || scale->fmt != 0 || base->fmt != 0 ||
        (norm && norm->fmt != 0))
        return 0;
    int N = fn->O;
    long long Mll = (long long)(sqrtf((float)(1 + N)) + 0.5f) - 1;
    if (Mll < 1 || Mll > 16 || 2 * Mll + Mll * Mll != N) return 0;
    int M = (int)Mll, MH = M * H;
    long long sh = (long long)tokens * (M + M * M);
    long long xh = (long long)tokens * H;
    if (fn->I != MH || residual->elements < (long long)tokens * MH ||
        state->elements < sh || input->elements < xh ||
        scale->O * scale->I < 3 || base->O * base->I < N ||
        (norm && norm->O * norm->I < H))
        return 0;
    if (!state->data) {
        state->data = calloc((size_t)state->elements, sizeof(float));
        if (!state->data) return 0;
    }
    if (!input->data) {
        input->data = calloc((size_t)input->elements, sizeof(float));
        if (!input->data) return 0;
    }
    struct timespec ts0; clock_gettime(CLOCK_MONOTONIC, &ts0);
    double t0 = ts0.tv_sec + ts0.tv_nsec / 1e9;
    /* mix_raw[t*N+n] = sum_i fn[n][i]*residual[t*MH+i] — one fmt=0 matmul
     * over all tokens (fp32 tree/warp order, same contract as the decode
     * fn matvec). */
    float *mixes = malloc((size_t)tokens * N * sizeof(*mixes));
    float *pre_all = malloc((size_t)tokens * M * sizeof(*pre_all));
    if (!mixes || !pre_all) { free(pre_all); free(mixes); return 0; }
    int ok = coli_vk_matmul(&fn->vk, mixes, residual->data, NULL, NULL, 0,
                            tokens, MH, N, 1);
    if (ok) {
        const float *sc = scale->whost_cache
                            ? scale->whost_cache
                            : coli_vk_tensor_wptr(scale->vk);
        const float *bs = base->whost_cache
                            ? base->whost_cache
                            : coli_vk_tensor_wptr(base->vk);
        const float *w = norm
            ? (norm->whost_cache ? norm->whost_cache
                                 : coli_vk_tensor_wptr(norm->vk))
            : NULL;
        if (!sc || !bs || (norm && !w)) {
            ok = 0;
        } else {
            for (int t = 0; ok && t < tokens; t++) {
                const float *res = residual->data + (size_t)t * MH;
                float mean_square = 0.0f;
                for (int i = 0; i < MH; i++)
                    mean_square += res[i] * res[i];
                float inverse_rms = 1.0f / sqrtf(mean_square / MH + rms_eps);
                float *mx = mixes + (size_t)t * N;
                for (int n = 0; n < N; n++) mx[n] *= inverse_rms;
                float *post = state->data + (size_t)t * M;
                float *comb =
                    state->data + (size_t)tokens * M + (size_t)t * M * M;
                float *pre = pre_all + (size_t)t * M;
                if (!ds4vk_mhc_sinkhorn(pre, post, comb, mx, sc, bs, M,
                                        sink_iters, pre_eps, post_mult,
                                        sink_eps))
                    ok = 0;
            }
        }
    }
    if (ok) {
        for (int t = 0; t < tokens; t++) {
            const float *res = residual->data + (size_t)t * MH;
            const float *pre = pre_all + (size_t)t * M;
            float *red = input->data + (size_t)t * H;
            for (int h = 0; h < H; h++) {
                float v = 0.0f;
                for (int i = 0; i < M; i++)
                    v += pre[i] * res[(size_t)i * H + h];
                red[h] = ds4vk_bf16(v);
            }
            if (norm) {
                const float *w = norm->whost_cache
                                   ? norm->whost_cache
                                   : coli_vk_tensor_wptr(norm->vk);
                if (!w) { ok = 0; break; }
                float mean_square = 0.0f;
                for (int h = 0; h < H; h++)
                    mean_square += red[h] * red[h];
                float inverse_rms = 1.0f / sqrtf(mean_square / H + norm_eps);
                for (int h = 0; h < H; h++)
                    red[h] = ds4vk_bf16(red[h] * inverse_rms * w[h]);
            }
        }
    }
    free(pre_all); free(mixes);
    if (ok) ds4vk_prof_tick(fn, t0);
    return ok;
}

int dsv4_cuda_mhc_pre_norm_batch(const Dsv4CudaActivation *residual,
                                 Dsv4CudaTensor *fn, Dsv4CudaTensor *scale,
                                 Dsv4CudaTensor *base, Dsv4CudaTensor *norm,
                                 int tokens, int H, Dsv4CudaActivation *state,
                                 Dsv4CudaActivation *input) {
    /* M4-1: the batched normalized_hc_pre (prefill block phase hc1/hc2). The
     * engine wrapper (coli_v4_gpu_mhc_pre_norm_batch) uploaded the whole
     * chunk's residual (tokens*hc*hidden) and downloads posts/combs/normalized
     * after the call; any refusal -> the per-token CPU loop (D6). The ABI
     * carries no eps/iters — mirror the CUDA tier and the engine's config:
     * rms_norm_eps=1e-6, hc_eps=1e-6, hc_sinkhorn_iters=20, post_mult=2.0
     * (the batch path only runs on the real model: the engine wrapper
     * requires hc==4 && hidden==4096). */
    return ds4vk_mhc_pre_batch_common(
        residual, fn, scale, base, norm, tokens, H, DS4VK_MHC_EPS,
        DS4VK_MHC_EPS, DS4VK_MHC_EPS, DS4VK_MHC_POST_MULT,
        DS4VK_MHC_BATCH_ITERS, DS4VK_MHC_EPS, state, input);
}

int dsv4_cuda_mhc_pre_batch(const Dsv4CudaActivation *residual,
                            Dsv4CudaTensor *fn, Dsv4CudaTensor *scale,
                            Dsv4CudaTensor *base, int tokens, int H,
                            Dsv4CudaActivation *state, Dsv4CudaActivation *input) {
    /* plain batched pre: the raw bf16-rounded pre*residual input (no norm
     * rmsnorm) — the CUDA mhc_input batch contract; ABI completeness in the
     * VK build (the engine uses the _norm variant). Same constants. */
    return ds4vk_mhc_pre_batch_common(
        residual, fn, scale, base, NULL, tokens, H, DS4VK_MHC_EPS,
        DS4VK_MHC_EPS, DS4VK_MHC_EPS, DS4VK_MHC_POST_MULT,
        DS4VK_MHC_BATCH_ITERS, 0.0f, state, input);
}

int dsv4_cuda_mhc_post(const Dsv4CudaActivation *x,
                       const Dsv4CudaActivation *residual,
                       const Dsv4CudaActivation *state, int M, int H,
                       Dsv4CudaActivation *out) {
    /* coli_v4_hc_post: out[j*H+h] = bf16(post[j]*x[h] + sum_i
     * comb[i*M+j]*residual[i*H+h]) — sequential fp32 in the CPU's order,
     * bf16 round at the same point (the engine's extra coli_bf16_round_array
     * after the call is a no-op on the already-rounded values). post/comb
     * come from state ([post M][comb M*M]); gated on the op group via a
     * synthetic tensor anchored to DS4VK_OP_MHC (the op has no weight
     * handle — same pattern as the attention core). */
    if (!x || !x->data || !residual || !residual->data || !state ||
        !state->data || !out || M < 1 || H < 1 ||
        x->elements < H || residual->elements < M * H ||
        state->elements < M + M * M + M || out->elements < M * H ||
        !ds4vk_op_enabled_g_mhc())
        return 0;
    if (ds4vk_fault_hit_g(DS4VK_OP_MHC)) {
        ds4vk_fault_log("mhc");
        return 0;
    }
    if (!out->data) {
        out->data = calloc((size_t)out->elements, sizeof(float));
        if (!out->data) return 0;
    }
    struct timespec ts0; clock_gettime(CLOCK_MONOTONIC, &ts0);
    double t0 = ts0.tv_sec + ts0.tv_nsec / 1e9;
    const float *post = state->data;
    const float *comb = state->data + M;
    for (int j = 0; j < M; j++) {
        for (int h = 0; h < H; h++) {
            float v = post[j] * x->data[h];
            for (int i = 0; i < M; i++)
                v += comb[i * M + j] * residual->data[(size_t)i * H + h];
            out->data[(size_t)j * H + h] = ds4vk_bf16(v);
        }
    }
    Dsv4CudaTensor fake; memset(&fake, 0, sizeof(fake)); fake.op = DS4VK_OP_MHC;
    ds4vk_prof_tick(&fake, t0);
    return 1;
}

int dsv4_cuda_mhc_post_pre(const Dsv4CudaActivation *x,
                           const Dsv4CudaActivation *residual,
                           Dsv4CudaActivation *state, int M, int H,
                           Dsv4CudaActivation *out, Dsv4CudaTensor *fn,
                           Dsv4CudaTensor *scale, Dsv4CudaTensor *base,
                           float rms_eps, float pre_eps, float sink_eps,
                           float post_mult, int sink_iters,
                           Dsv4CudaActivation *input) {
    (void)x; (void)residual; (void)state; (void)M; (void)H; (void)out; (void)fn;
    (void)scale; (void)base; (void)rms_eps; (void)pre_eps; (void)sink_eps;
    (void)post_mult; (void)sink_iters; (void)input;
    return 0;
}

int dsv4_cuda_mhc_post_pre_norm(const Dsv4CudaActivation *x,
                                const Dsv4CudaActivation *residual,
                                Dsv4CudaActivation *state, int M, int H,
                                Dsv4CudaActivation *out, Dsv4CudaTensor *fn,
                                Dsv4CudaTensor *scale, Dsv4CudaTensor *base,
                                Dsv4CudaTensor *norm, float rms_eps,
                                float pre_eps, float sink_eps, float post_mult,
                                int sink_iters, float norm_eps,
                                Dsv4CudaActivation *input) {
    (void)x; (void)residual; (void)state; (void)M; (void)H; (void)out; (void)fn;
    (void)scale; (void)base; (void)norm; (void)rms_eps; (void)pre_eps;
    (void)sink_eps; (void)post_mult; (void)sink_iters; (void)norm_eps;
    (void)input;
    return 0;
}

int dsv4_cuda_mhc_post_pre_norm_batch(const Dsv4CudaActivation *x,
                                      const Dsv4CudaActivation *residual,
                                      Dsv4CudaActivation *state, int tokens,
                                      int H, Dsv4CudaActivation *out,
                                      Dsv4CudaTensor *fn, Dsv4CudaTensor *scale,
                                      Dsv4CudaTensor *base, Dsv4CudaTensor *norm,
                                      Dsv4CudaActivation *input) {
    /* M4-1: the fused batch post -> next pre_norm (CUDA-tier composition; the
     * engine's VK prefill path calls post_batch + pre_norm_batch separately,
     * this entry is ABI completeness): hc_post from the state into out, then
     * normalized_hc_pre over out (the post output is the next residual). */
    if (!ds4vk_op_enabled_g_mhc()) return 0;
    return dsv4_cuda_mhc_post_batch(x, residual, state, tokens, H, out) &&
           ds4vk_mhc_pre_batch_common(
               out, fn, scale, base, norm, tokens, H, DS4VK_MHC_EPS,
               DS4VK_MHC_EPS, DS4VK_MHC_EPS, DS4VK_MHC_POST_MULT,
               DS4VK_MHC_BATCH_ITERS, DS4VK_MHC_EPS, state, input);
}

int dsv4_cuda_mhc_post_batch(const Dsv4CudaActivation *x,
                             const Dsv4CudaActivation *residual,
                             const Dsv4CudaActivation *state, int tokens, int H,
                             Dsv4CudaActivation *out) {
    /* M4-1: the batched coli_v4_hc_post — out[t][j][h] =
     * bf16(post[t][j]*x[t][h] + sum_i comb[t][i][j]*residual[t][i][h]) in the
     * CPU's per-token order, bf16-rounded at the same point (the engine's
     * extra coli_bf16_round_array after the call is a no-op on the already-
     * rounded values). post/comb come from the state activation in the CUDA
     * batch layout [post (tokens*M)][comb (tokens*M*M)]; M is derived from
     * state->elements / tokens (M^2 + M - E/t = 0). Gated on the mhc group
     * directly (the op has no weight handle — same pattern as the decode
     * post). */
    if (!x || !x->data || !residual || !residual->data || !state ||
        !state->data || !out || tokens < 1 || H < 1 ||
        !ds4vk_op_enabled_g_mhc())
        return 0;
    if (ds4vk_fault_hit_g(DS4VK_OP_MHC)) {
        ds4vk_fault_log("mhc");
        return 0;
    }
    long long Ept = state->elements / tokens;   /* M + M*M */
    long long Mll = (long long)((sqrtf(1.0f + 4.0f * (float)Ept) - 1.0f) / 2.0f + 0.5f);
    if (Mll < 1 || Mll > 16 || Mll + Mll * Mll != Ept) return 0;
    int M = (int)Mll, MH = M * H;
    if (x->elements < (long long)tokens * H ||
        residual->elements < (long long)tokens * MH ||
        out->elements < (long long)tokens * MH)
        return 0;
    if (!out->data) {
        out->data = calloc((size_t)out->elements, sizeof(float));
        if (!out->data) return 0;
    }
    struct timespec ts0; clock_gettime(CLOCK_MONOTONIC, &ts0);
    double t0 = ts0.tv_sec + ts0.tv_nsec / 1e9;
    const float *post_base = state->data;
    const float *comb_base = state->data + (size_t)tokens * M;
    for (int t = 0; t < tokens; t++) {
        const float *br = x->data + (size_t)t * H;
        const float *res = residual->data + (size_t)t * MH;
        const float *post = post_base + (size_t)t * M;
        const float *comb = comb_base + (size_t)t * M * M;
        float *o = out->data + (size_t)t * MH;
        for (int j = 0; j < M; j++) {
            for (int h = 0; h < H; h++) {
                float v = post[j] * br[h];
                for (int i = 0; i < M; i++)
                    v += comb[(size_t)i * M + j] * res[(size_t)i * H + h];
                o[(size_t)j * H + h] = ds4vk_bf16(v);
            }
        }
    }
    Dsv4CudaTensor fake; memset(&fake, 0, sizeof(fake)); fake.op = DS4VK_OP_MHC;
    ds4vk_prof_tick(&fake, t0);
    return 1;
}

int dsv4_cuda_attention_first(const Dsv4CudaActivation *input,
                              Dsv4CudaTensor *attn_norm, Dsv4CudaTensor *q_a,
                              Dsv4CudaTensor *q_norm, Dsv4CudaTensor *q_b,
                              Dsv4CudaTensor *wkv, Dsv4CudaTensor *kv_norm,
                              Dsv4CudaTensor *sink, Dsv4CudaTensor *wo_a,
                              Dsv4CudaTensor *wo_b, int heads, int head_dim,
                              int qk_rope, int groups, float eps,
                              Dsv4CudaActivation *output) {
    (void)input; (void)attn_norm; (void)q_a; (void)q_norm; (void)q_b; (void)wkv;
    (void)kv_norm; (void)sink; (void)wo_a; (void)wo_b; (void)heads;
    (void)head_dim; (void)qk_rope; (void)groups; (void)eps; (void)output;
    return 0;
}

int dsv4_cuda_attention_window(
    const Dsv4CudaActivation *input, Dsv4CudaTensor *attn_norm,
    Dsv4CudaTensor *q_a, Dsv4CudaTensor *q_norm, Dsv4CudaTensor *q_b,
    Dsv4CudaTensor *wkv, Dsv4CudaTensor *kv_norm, Dsv4CudaTensor *sink,
    Dsv4CudaTensor *wo_a, Dsv4CudaTensor *wo_b, Dsv4CudaTensor *compress_wkv,
    Dsv4CudaTensor *compress_wgate, Dsv4CudaTensor *compress_ape,
    Dsv4CudaTensor *compress_norm, int compress_ratio, int heads, int head_dim,
    int qk_rope, int groups, int pos, float eps, Dsv4CudaKvCache *cache,
    Dsv4CudaActivation *output) {
    (void)input; (void)attn_norm; (void)q_a; (void)q_norm; (void)q_b; (void)wkv;
    (void)kv_norm; (void)sink; (void)wo_a; (void)wo_b; (void)compress_wkv;
    (void)compress_wgate; (void)compress_ape; (void)compress_norm;
    (void)compress_ratio; (void)heads; (void)head_dim; (void)qk_rope;
    (void)groups; (void)pos; (void)eps; (void)cache; (void)output;
    return 0;
}

int dsv4_cuda_attention_sparse_batch(
    const Dsv4CudaActivation *input, Dsv4CudaTensor *attn_norm,
    Dsv4CudaTensor *qkv, Dsv4CudaTensor *q_norm, Dsv4CudaTensor *q_b,
    Dsv4CudaTensor *kv_norm, Dsv4CudaTensor *sink, int heads, int head_dim,
    int start_pos, int tokens, float eps, Dsv4CudaKvCache *cache,
    Dsv4CudaActivation *context) {
    (void)input; (void)attn_norm; (void)qkv; (void)q_norm; (void)q_b;
    (void)kv_norm; (void)sink; (void)heads; (void)head_dim; (void)start_pos;
    (void)tokens; (void)eps; (void)cache; (void)context;
    return 0;
}

int dsv4_cuda_attention_output_batch(const Dsv4CudaActivation *context,
                                     Dsv4CudaTensor *wo_a, Dsv4CudaTensor *wo_b,
                                     int groups, int tokens,
                                     Dsv4CudaActivation *output) {
    /* M3d-1 enabler: the batched attention-output chain (grouped wo_a +
     * wo_b). The batch attention unit's grouped wo_a CPU loop reads the
     * resident fp8 RAM that the D7 boot flow returns to the OS, so this
     * must run on the GPU or the prefill computes on zero pages. Mirrors
     * the CPU reference exactly: per group, oa[g] = wo_a[g] (o_rank x
     * group_width slice of the block-diagonal) times the group's context
     * columns, then bf16-round the full oa (ds4vk_bf16, the same point as
     * the CPU's coli_bf16_round_array), then out = wo_b · oa (the engine
     * rounds the final outputs, matching the CPU). Reuses the decode
     * grouped-matvec's per-group tensor slices and the shared batch QDQ /
     * fmt=8 matmul (M2-4-verified at S>1). fmt=9 (fp8-bf16 wo_a mirror)
     * computes as fmt=8 like the decode path. */
    if (!context || !context->data || !wo_a || !wo_a->vk || !wo_b ||
        !wo_b->vk || !output || groups < 1 || tokens < 1 ||
        !ds4vk_op_enabled(wo_a) || !ds4vk_op_enabled(wo_b))
        return 0;
    if (ds4vk_fault_hit(wo_a) || ds4vk_fault_hit(wo_b)) {
        ds4vk_fault_log(ds4vk_op_name((Ds4vkOp)wo_a->op));
        return 0;
    }
    if ((wo_a->fmt != 8 && wo_a->fmt != 9) || wo_b->fmt != 8 ||
        wo_a->O % groups)
        return 0;
    int Og = wo_a->O / groups;              /* o_rank per group */
    int Ig = wo_a->I;                       /* group_width (columns per group) */
    int q_width = groups * Ig;              /* context columns (heads*head_dim) */
    long long in_elements = (long long)tokens * q_width;
    long long oa_elements = (long long)tokens * (long long)wo_a->O;
    long long out_elements = (long long)tokens * (long long)wo_b->O;
    if (context->elements < in_elements || output->elements < out_elements)
        return 0;
    if (!output->data) {
        output->data = calloc((size_t)output->elements, sizeof(float));
        if (!output->data) return 0;
    }
    if (wo_a->O % groups || wo_a->I < 1) return 0;
    float *group_in = malloc((size_t)tokens * (size_t)Ig * sizeof(float));
    float *act = malloc((size_t)tokens * (size_t)Ig * sizeof(float));
    uint8_t *actsc = malloc((size_t)tokens * (size_t)((Ig + 127) / 128));
    float *group_out = malloc((size_t)tokens * (size_t)Og * sizeof(float));
    float *oa = malloc((size_t)oa_elements * sizeof(float));
    if (!group_in || !act || !actsc || !group_out || !oa) {
        free(oa); free(group_out); free(actsc); free(act); free(group_in);
        return 0;
    }
    int ok = 1;
    for (int g = 0; g < groups && ok; g++) {
        /* gather the group's columns (strided per token) — the CPU loop's
         * per-item memcpy */
        for (int t = 0; t < tokens; t++)
            memcpy(group_in + (size_t)t * Ig,
                   context->data + (size_t)t * q_width + (size_t)g * Ig,
                   (size_t)Ig * sizeof(float));
        /* Per-group matmul into a CONTIGUOUS temp (the matmul writes
         * y[S,O] contiguously); oa is tokens x o_width row-major, so the
         * group's block must be SCATTERED at stride o_width — passing
         * oa + g*Og directly would overlap groups (the decode grouped
         * matvec gets away with it only because S=1). */
        ColiVkTensor view = *wo_a->vk;   /* group slice via arena offsets */
        ColiVkTensor *vp = &view;
        view.fmt = 8;   /* fmt=9 (fp8-bf16) computes as fmt=8, M3a */
        view.O = Og;    /* per-group rows; rowWords stays the full stride */
        view.woff = (int)((size_t)g * (size_t)Og * (size_t)view.rowWords);
        view.soff = (int)((size_t)g * (size_t)((Og + 127) / 128) *
                          (size_t)((Ig + 127) / 128));
        ok = coli_vk_activation_qdq(act, actsc, group_in, tokens, Ig, 128) &&
             coli_vk_matmul(&vp, group_out, act, NULL, NULL,
                            8, tokens, Ig, Og, 1);
        if (ok)
            for (int t = 0; t < tokens; t++)
                memcpy(oa + (size_t)t * wo_a->O + (size_t)g * Og,
                       group_out + (size_t)t * Og,
                       (size_t)Og * sizeof(float));
    }
    if (ok) {
        for (size_t i = 0; i < (size_t)oa_elements; i++)
            oa[i] = ds4vk_bf16(oa[i]);
        ok = coli_vk_matmul(&wo_b->vk, output->data, oa, NULL, NULL, 8, tokens,
                            wo_a->O, wo_b->O, 1);

    }
    free(oa); free(group_out); free(actsc); free(act); free(group_in);
    return ok;
}

int dsv4_cuda_attention_window_tp2(
    const Dsv4CudaActivation *input, Dsv4CudaActivation *peer_input,
    const Dsv4CudaAttentionWeights *primary, const Dsv4CudaAttentionWeights *peer,
    int compress_ratio, int heads, int head_dim, int qk_rope, int groups,
    int pos, float eps, Dsv4CudaKvCache *cache, Dsv4CudaKvCache *peer_cache,
    Dsv4CudaActivation *output, Dsv4CudaActivation *peer_output) {
    (void)input; (void)peer_input; (void)primary; (void)peer;
    (void)compress_ratio; (void)heads; (void)head_dim; (void)qk_rope;
    (void)groups; (void)pos; (void)eps; (void)cache; (void)peer_cache;
    (void)output; (void)peer_output;
    return 0;
}

Dsv4CudaKvCache *dsv4_cuda_kv_create(int device, int window, int head_dim,
                                     int max_tokens, int rope_pairs,
                                     const float *rope_cos,
                                     const float *rope_sin,
                                     const float *compress_cos,
                                     const float *compress_sin) {
    (void)device; (void)window; (void)head_dim; (void)max_tokens; (void)rope_pairs;
    (void)rope_cos; (void)rope_sin; (void)compress_cos; (void)compress_sin;
    /* NULL: the kv-cache path fails fast and the engine stays CPU (the
     * attention batch path is env-gated anyway; M3b implements the real ring). */
    return NULL;
}

void dsv4_cuda_kv_free(Dsv4CudaKvCache *cache) { free(cache); }

/* ---- expert / MoE (fp4 mirrors fail -> hybrid MoE block unreachable) ---- */

int dsv4_cuda_expert_group(Dsv4CudaTensor *const *gate,
                           Dsv4CudaTensor *const *up,
                           Dsv4CudaTensor *const *down, const float *weights,
                           int count, float limit, float *y, const float *x) {
    /* M5-1b: decode-path routed-expert group, batched — TWO submits per layer
     * per token instead of ~3 per expert (~19 for top-6). The engine's fused
     * path calls this when ALL of a token's selected experts carry GPU
     * mirrors. CPU reference: the per-expert coli_v4_expert_forward_ref
     * accumulation the engine falls back to (gate/up fp4 matvecs -> bf16
     * round -> swiglu(limit) -> *route_weight -> bf16 round -> down fp4
     * matvec -> bf16 round -> sum). The VK tier replicates the CPU chain's
     * rounding points BITWISE host-side (ds4vk_bf16 / ds4vk_sigmoid are the
     * engine twins, proven by M3c); only the fp4 matvecs are GPU (thresholded,
     * same contract as the fp8 tier). Phase 1 = QDQ(x) + ALL gate/up matmuls
     * in ONE command buffer (they share one QDQ — the CPU QDQs per call,
     * deterministic, so a single pass is bitwise identical); phase 2 =
     * QDQ(hid rows, S=count) + ALL down matmuls in ONE command buffer. */
    if (!y || !x || count < 1 || count > 64 || !gate || !up || !down ||
        limit < 0.0f)
        return 0;
    /* op gate + L4 fault injection: the group has no single tensor handle,
     * gate the group pseudo-op directly (the mirrors carry it too). */
    if (!ds4vk_op_enabled_g_expert()) return 0;
    if (ds4vk_fault_hit_g(DS4VK_OP_EXPERT)) {
        ds4vk_fault_log("expert");
        return 0;
    }
    Dsv4CudaTensor *g0 = gate[0];
    if (!g0 || !g0->vk) return 0;
    int H = g0->I, I = g0->O;   /* H = input dim, I = moe_intermediate */
    if (H < 1 || I < 1 || I % 32 || H % 32) return 0;
    for (int e = 0; e < count; e++)
        if (!gate[e] || !up[e] || !down[e] || !gate[e]->vk || !up[e]->vk ||
            !down[e]->vk || gate[e]->fmt != 10 || up[e]->fmt != 10 ||
            down[e]->fmt != 10 || gate[e]->I != H || gate[e]->O != I ||
            up[e]->I != H || up[e]->O != I || down[e]->I != I ||
            down[e]->O != H)
            return 0;
    struct timespec ts0; clock_gettime(CLOCK_MONOTONIC, &ts0);
    double t0 = ts0.tv_sec + ts0.tv_nsec / 1e9;
    ColiVkTensor *gv[64], *uv[64], *dv[64];
    for (int e = 0; e < count; e++) {
        gv[e] = gate[e]->vk;
        uv[e] = up[e]->vk;
        dv[e] = down[e]->vk;
    }
    float *gatebuf = malloc((size_t)count * (size_t)I * sizeof(float));
    float *upbuf = malloc((size_t)count * (size_t)I * sizeof(float));
    float *hid = malloc((size_t)count * (size_t)I * sizeof(float));
    float *out = malloc((size_t)count * (size_t)H * sizeof(float));
    if (!gatebuf || !upbuf || !hid || !out) {
        free(out); free(hid); free(upbuf); free(gatebuf);
        return 0;
    }
    /* Phase 1: one submit for QDQ(x) + every expert's gate/up. */
    int ok = coli_vk_expert_dsv4_phase1(gv, uv, gatebuf, upbuf, x,
                                        count, H, I);
    /* host rounding chain — bitwise vs coli_v4_expert_forward_ref:
     * bf16(gate) / bf16(up) -> limit clamps -> sigmoid -> * weight -> bf16.
     * The per-expert down QDQ uses a fresh activation (hid). */
    if (ok)
        for (int e = 0; e < count && ok; e++) {
            const float *g = gatebuf + (size_t)e * I;
            const float *u = upbuf + (size_t)e * I;
            float *h = hid + (size_t)e * I;
            for (int i = 0; i < I; i++) {
                float gv_ = ds4vk_bf16(g[i]);
                float uv_ = ds4vk_bf16(u[i]);
                if (limit > 0.0f) {
                    gv_ = fminf(gv_, limit);
                    uv_ = fmaxf(-limit, fminf(uv_, limit));
                }
                h[i] = ds4vk_bf16(gv_ * ds4vk_sigmoid(gv_) * uv_ *
                                   weights[e]);
            }
        }
    /* Phase 2: one submit for QDQ(hid, S=count) + every expert's down. */
    if (ok) ok = coli_vk_expert_dsv4_phase2(dv, out, hid, count, H, I);
    if (ok) {
        memset(y, 0, (size_t)H * sizeof(float));
        for (int e = 0; e < count; e++) {
            const float *o = out + (size_t)e * H;
            for (int i = 0; i < H; i++) y[i] += ds4vk_bf16(o[i]);
        }
    }
    free(out); free(hid); free(upbuf); free(gatebuf);
    if (ok) ds4vk_prof_tick(g0, t0);
    return ok;
}

int dsv4_cuda_expert_fp8(Dsv4CudaTensor *gate, Dsv4CudaTensor *up,
                         Dsv4CudaTensor *down, float limit, float *y,
                         const float *x) {
    (void)gate; (void)up; (void)down; (void)limit; (void)y; (void)x;
    return 0;
}

int dsv4_cuda_moe(Dsv4CudaTensor *const *gate, Dsv4CudaTensor *const *up,
                  Dsv4CudaTensor *const *down, const float *weights, int count,
                  Dsv4CudaTensor *shared_gate, Dsv4CudaTensor *shared_up,
                  Dsv4CudaTensor *shared_down, float limit, float *y,
                  const float *x) {
    (void)gate; (void)up; (void)down; (void)weights; (void)count;
    (void)shared_gate; (void)shared_up; (void)shared_down; (void)limit; (void)y;
    (void)x;
    return 0;
}

int dsv4_cuda_moe_activation(Dsv4CudaTensor *const *gate,
                             Dsv4CudaTensor *const *up,
                             Dsv4CudaTensor *const *down, const float *weights,
                             int count, Dsv4CudaTensor *shared_gate,
                             Dsv4CudaTensor *shared_up,
                             Dsv4CudaTensor *shared_down, float limit,
                             const Dsv4CudaActivation *input,
                             Dsv4CudaActivation *output) {
    (void)gate; (void)up; (void)down; (void)weights; (void)count;
    (void)shared_gate; (void)shared_up; (void)shared_down; (void)limit;
    (void)input; (void)output;
    return 0;
}

Dsv4CudaExpertSet *dsv4_cuda_expert_set_create(
    Dsv4CudaTensor *const *gate, Dsv4CudaTensor *const *up,
    Dsv4CudaTensor *const *down, int count, Dsv4CudaTensor *shared_gate,
    Dsv4CudaTensor *shared_up, Dsv4CudaTensor *shared_down) {
    (void)gate; (void)up; (void)down; (void)count; (void)shared_gate;
    (void)shared_up; (void)shared_down;
    return NULL;
}

Dsv4CudaExpertSet *dsv4_cuda_expert_bank_create(int count, int hidden,
                                                int intermediate, int device,
                                                Dsv4CudaTensor *shared_gate,
                                                Dsv4CudaTensor *shared_up,
                                                Dsv4CudaTensor *shared_down) {
    (void)count; (void)hidden; (void)intermediate; (void)shared_gate;
    (void)shared_up; (void)shared_down;
    Dsv4CudaExpertSet *s = calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->device = device;
    return s;
}

int dsv4_cuda_expert_bank_upload(Dsv4CudaExpertSet *set, int expert,
                                 const uint8_t *gate_weight,
                                 const uint8_t *gate_scale,
                                 const uint8_t *up_weight,
                                 const uint8_t *up_scale,
                                 const uint8_t *down_weight,
                                 const uint8_t *down_scale,
                                 Dsv4CudaTensor **gate, Dsv4CudaTensor **up,
                                 Dsv4CudaTensor **down) {
    (void)set; (void)expert; (void)gate_weight; (void)gate_scale;
    (void)up_weight; (void)up_scale; (void)down_weight; (void)down_scale;
    if (gate) *gate = NULL;
    if (up) *up = NULL;
    if (down) *down = NULL;
    return 0;
}

int dsv4_cuda_expert_bank_set_shared(Dsv4CudaExpertSet *set, Dsv4CudaTensor *sg,
                                     Dsv4CudaTensor *su, Dsv4CudaTensor *sd) {
    (void)set; (void)sg; (void)su; (void)sd;
    return 0;
}

int dsv4_cuda_expert_bank_upload_aux(Dsv4CudaExpertSet *set, int expert,
                                     const uint8_t *gw, const uint8_t *gs,
                                     const uint8_t *uw, const uint8_t *us,
                                     const uint8_t *dw, const uint8_t *ds,
                                     Dsv4CudaTensor **gate, Dsv4CudaTensor **up,
                                     Dsv4CudaTensor **down) {
    (void)set; (void)expert; (void)gw; (void)gs; (void)uw; (void)us; (void)dw;
    (void)ds;
    if (gate) *gate = NULL;
    if (up) *up = NULL;
    if (down) *down = NULL;
    return 0;
}

int dsv4_cuda_expert_bank_upload_tp2(Dsv4CudaExpertSet *set, int expert,
                                     int rank, const uint8_t *gate_weight,
                                     const uint8_t *gate_scale,
                                     const uint8_t *up_weight,
                                     const uint8_t *up_scale,
                                     const uint8_t *down_weight,
                                     const uint8_t *down_scale) {
    (void)set; (void)expert; (void)rank; (void)gate_weight; (void)gate_scale;
    (void)up_weight; (void)up_scale; (void)down_weight; (void)down_scale;
    return 0;
}

void dsv4_cuda_expert_set_free(Dsv4CudaExpertSet *set) { free(set); }

int dsv4_cuda_expert_set_upload_hash(Dsv4CudaExpertSet *set, const int64_t *map,
                                     int vocab, int topk) {
    (void)set; (void)map; (void)vocab; (void)topk;
    return 0;
}

int dsv4_cuda_route_moe(const Dsv4CudaActivation *input, Dsv4CudaTensor *gate,
                        Dsv4CudaTensor *bias, int token, float routed_scale,
                        Dsv4CudaExpertSet *experts, float limit,
                        Dsv4CudaActivation *output) {
    (void)input; (void)gate; (void)bias; (void)token; (void)routed_scale;
    (void)experts; (void)limit; (void)output;
    return 0;
}

int dsv4_cuda_route_moe_batch(const Dsv4CudaActivation *input,
                              Dsv4CudaTensor *gate, Dsv4CudaTensor *bias,
                              const int *tokens, int count, float routed_scale,
                              Dsv4CudaExpertSet *experts, float limit,
                              Dsv4CudaActivation *output) {
    (void)input; (void)gate; (void)bias; (void)tokens; (void)count;
    (void)routed_scale; (void)experts; (void)limit; (void)output;
    return 0;
}

int dsv4_cuda_route_top6_batch(const Dsv4CudaActivation *input,
                               Dsv4CudaTensor *gate, Dsv4CudaTensor *bias,
                               int count, float routed_scale, int *ids,
                               float *weights) {
    (void)input; (void)gate; (void)bias; (void)count; (void)routed_scale;
    (void)ids; (void)weights;
    return 0;
}

int dsv4_cuda_route_moe_ids_batch(const Dsv4CudaActivation *input,
                                  const int *ids, const float *weights,
                                  int count, Dsv4CudaExpertSet *experts,
                                  float limit, Dsv4CudaActivation *output) {
    (void)input; (void)ids; (void)weights; (void)count; (void)experts;
    (void)limit; (void)output;
    return 0;
}

int dsv4_cuda_route_moe_ep2(const Dsv4CudaActivation *input,
                            Dsv4CudaTensor *gate, Dsv4CudaTensor *bias,
                            const Dsv4CudaActivation *peer_input,
                            Dsv4CudaTensor *peer_gate, Dsv4CudaTensor *peer_bias,
                            int token, float routed_scale,
                            Dsv4CudaExpertSet *local, Dsv4CudaExpertSet *peer,
                            float limit, Dsv4CudaActivation *output,
                            Dsv4CudaActivation *peer_output) {
    (void)input; (void)gate; (void)bias; (void)peer_input; (void)peer_gate;
    (void)peer_bias; (void)token; (void)routed_scale; (void)local; (void)peer;
    (void)limit; (void)output; (void)peer_output;
    return 0;
}

/* ---- CUDA graphs / profiler / stream (no-ops in the stub) ---- */

int dsv4_cuda_graph_begin(int device) { (void)device; return 0; }

Dsv4CudaGraph *dsv4_cuda_graph_end(int device) {
    (void)device;
    return NULL;
}

int dsv4_cuda_graph_end_pair(int primary, int peer,
                             Dsv4CudaGraph **primary_graph,
                             Dsv4CudaGraph **peer_graph) {
    (void)primary; (void)peer;
    if (primary_graph) *primary_graph = NULL;
    if (peer_graph) *peer_graph = NULL;
    return 0;
}

int dsv4_cuda_graph_launch(Dsv4CudaGraph *graph) {
    (void)graph;
    return 0;
}

void dsv4_cuda_graph_free(Dsv4CudaGraph *graph) { free(graph); }

void dsv4_cuda_profiler_start(void) {}
void dsv4_cuda_profiler_stop(void) {}

int dsv4_cuda_stream_drain(int device) { (void)device; return 0; }
