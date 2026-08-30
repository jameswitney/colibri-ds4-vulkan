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
} Dsv4CudaTensor;

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
        double free_mb = budget - used;
        return free_mb > 0 ? (long long)(free_mb / 1e6) : 0;
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

int dsv4_cuda_upload_fp4(Dsv4CudaTensor **t, const uint8_t *w,
                         const uint8_t *scale, int O, int I, int device) {
    /* Deliberately fails: fp4 expert mirrors are the M5 expert tier. Keeping
     * them NULL keeps the hybrid MoE block unreachable (its failure path is
     * token-fatal, not a D6 fallback) and the run pure-CPU on experts. */
    (void)w; (void)scale; (void)O; (void)I; (void)device;
    if (t) *t = NULL;
    return 0;
}

int dsv4_cuda_upload_bf16(Dsv4CudaTensor **t, const uint16_t *w, int O, int I,
                          int device) {
    /* M3a: real bf16 upload (fmt=16, unquantized — the head.weight matvec
     * decodes bf16 in-shader; the compressor/indexer mirrors stay deferred
     * (M4 batched path). The head consumes the raw resident bf16 bytes, so
     * upload is byte-identical and the decode happens at matvec time. */
    if (!t || !w) return 0;
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
    *t = h;
    stub_upload_log("f32", device, O, I, 0, h->bytes);
    return 1;
}

int dsv4_cuda_tensor_refill_fp4(Dsv4CudaTensor *t, const uint8_t *w,
                                const uint8_t *scale, int O, int I, int sync) {
    (void)t; (void)w; (void)scale; (void)O; (void)I; (void)sync;
    return 0; /* no fp4 mirrors in v1 (M5) */
}

/* ---- tensor/activation handles ---- */

void dsv4_cuda_tensor_free(Dsv4CudaTensor *t) {
    if (!t) return;
    for (int g = 0; t->gvk && g < t->groups; g++)
        if (t->gvk[g]) coli_vk_tensor_free(t->gvk[g]);
    free(t->gvk);
    if (t->vk) coli_vk_tensor_free(t->vk);
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
     * batched prefill ops will move the bytes to device scratch in M4). */
    if (!a || !x || elements < 1) return 0;
    if (a->elements < elements) return 0;
    float *copy = realloc(a->data, (size_t)elements * sizeof(*copy));
    if (!copy) return 0;
    memcpy(copy, x, (size_t)elements * sizeof(*copy));
    a->data = copy;
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
    float *copy = realloc(dst->data, (size_t)elements * sizeof(*copy));
    if (!copy) return 0;
    memcpy(copy, src->data, (size_t)elements * sizeof(*copy));
    dst->data = copy;
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
    float *copy = realloc(dst->data, (size_t)(dst_offset + elements) * sizeof(*copy));
    if (!copy) return 0;
    dst->data = copy;
    memcpy(copy + dst_offset, src->data + src_offset,
           (size_t)elements * sizeof(*copy));
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

/* Grouped matvec (wo_a): the resident weight is the block-diagonal matrix
 * stacked vertically — groups of o_rank rows x group_width columns. The full
 * upload (O=groups*o_rank, I=group_width) is sliced into per-group tensors
 * on first use (the group weight/scale rows are contiguous in the arena
 * buffers); each group then runs the standard fmt=8 matvec against its own
 * input slice x + g*group_width. */
static int ds4vk_grouped_matvec_ensure(Dsv4CudaTensor *t, int groups) {
    if (groups < 1 || t->O % groups || t->I % groups) return 0;
    if (t->groups == groups && t->gvk) return 1;
    for (int g = 0; t->gvk && g < t->groups; g++)
        if (t->gvk[g]) coli_vk_tensor_free(t->gvk[g]);
    free(t->gvk);
    t->gvk = calloc((size_t)groups, sizeof(*t->gvk));
    t->groups = groups;
    if (!t->gvk || !t->vk) return 0;
    int Og = t->O / groups, Ig = t->I;      /* per-group: o_rank x group_width */
    const uint8_t *w = coli_vk_tensor_wptr(t->vk);
    const float *s = coli_vk_tensor_sptr(t->vk);
    if (!w || !s) return 0;                  /* backend down (no host-visible arena) */
    size_t wrow = (size_t)((Ig + 3) / 4) * 4;   /* padded row stride, fmt=8 (I%4==0 -> I) */
    size_t srow = (size_t)((Og + 127) / 128) * ((Ig + 127) / 128);
    /* The arena holds the EXPANDED fp32 scales; upload_tensor fmt=8 wants raw
     * E8M0 codes (it expands again). Re-encode the group's slice like the
     * engine's v4_gpu_upload_fp8_fmt does (power-of-two scales: exponent+127,
     * NaN -> 0xff). */
    uint8_t *e8 = malloc(srow);
    if (!e8) return 0;
    for (int g = 0; g < groups; g++) {
        const uint8_t *wg = w + (size_t)g * Og * wrow;
        const float *sg = s + (size_t)g * srow;
        for (size_t b = 0; b < srow; b++) {
            float value = sg[b];
            if (isnan(value) || !isfinite(value)) { e8[b] = 0xff; continue; }
            e8[b] = (uint8_t)(ilogbf(value) + 127);
        }
        if (!coli_vk_tensor_ensure(&t->gvk[g], wg, (const float *)e8, 8, Ig, Og, 0)) {
            free(e8);
            return 0;
        }
    }
    free(e8);
    return 1;
}

static int ds4vk_grouped_matvec(Dsv4CudaTensor *t, float *y, const float *x,
                                int groups) {
    if (!ds4vk_grouped_matvec_ensure(t, groups)) return 0;
    int Og = t->O / groups, Ig = t->I;
    size_t n = (size_t)Ig;
    float *act = malloc(n * sizeof(float));
    uint8_t *actsc = malloc((n + 127) / 128);
    if (!act || !actsc) { free(actsc); free(act); return 0; }
    int ok = 1;
    for (int g = 0; g < groups && ok; g++) {
        ok = coli_vk_activation_qdq(act, actsc, x + (size_t)g * Ig, 1, Ig, 128) &&
             coli_vk_matmul(&t->gvk[g], y + (size_t)g * Og, act, NULL, NULL,
                            8, 1, Ig, Og, 1);
    }
    free(actsc); free(act);
    return ok;
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
    struct timespec ts0; clock_gettime(CLOCK_MONOTONIC, &ts0);
    double t0 = ts0.tv_sec + ts0.tv_nsec / 1e9;
    int ok = t->fmt == 8 || t->fmt == 9 ? ds4vk_fp8_matvec(t, y, x)
          : t->fmt == 0 ? ds4vk_f32_matvec(t, y, x)
          : t->fmt == 16 ? ds4vk_bf16_matvec(t, y, x) : 0;
    if (ok) ds4vk_prof_tick(t, t0);
    return ok;
}

int dsv4_cuda_matvec_grouped(Dsv4CudaTensor *t, float *y, const float *x,
                             int groups) {
    if (!t || !t->vk || !y || !x || groups < 1 || !ds4vk_op_enabled(t)) return 0;
    struct timespec ts0; clock_gettime(CLOCK_MONOTONIC, &ts0);
    double t0 = ts0.tv_sec + ts0.tv_nsec / 1e9;
    int ok = (t->fmt == 8 || t->fmt == 9) ? ds4vk_grouped_matvec(t, y, x, groups)
            : groups == 1 ? dsv4_cuda_matvec(t, y, x) : 0;
    if (ok) ds4vk_prof_tick(t, t0);
    return ok;
}

int dsv4_cuda_matmul_batch(Dsv4CudaTensor *t, const Dsv4CudaActivation *input,
                           int tokens, Dsv4CudaActivation *output) {
    (void)t; (void)input; (void)tokens; (void)output;
    return 0;
}

int dsv4_cuda_matmul_bf16_batch(Dsv4CudaTensor *t, const float *x, int tokens,
                                float *y) {
    (void)t; (void)x; (void)tokens; (void)y;
    return 0;
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
                              bias ? (const float *)coli_vk_tensor_wptr(bias->vk) : NULL,
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
        const float *sc = coli_vk_tensor_wptr(scale->vk);
        const float *bs = coli_vk_tensor_wptr(base->vk);
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
            const float *w = coli_vk_tensor_wptr(norm->vk);
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

int dsv4_cuda_mhc_pre_norm_batch(const Dsv4CudaActivation *residual,
                                 Dsv4CudaTensor *fn, Dsv4CudaTensor *scale,
                                 Dsv4CudaTensor *base, Dsv4CudaTensor *norm,
                                 int tokens, int H, Dsv4CudaActivation *state,
                                 Dsv4CudaActivation *input) {
    (void)residual; (void)fn; (void)scale; (void)base; (void)norm; (void)tokens;
    (void)H; (void)state; (void)input;
    return 0;
}

int dsv4_cuda_mhc_pre_batch(const Dsv4CudaActivation *residual,
                            Dsv4CudaTensor *fn, Dsv4CudaTensor *scale,
                            Dsv4CudaTensor *base, int tokens, int H,
                            Dsv4CudaActivation *state, Dsv4CudaActivation *input) {
    (void)residual; (void)fn; (void)scale; (void)base; (void)tokens; (void)H;
    (void)state; (void)input;
    return 0;
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
    (void)x; (void)residual; (void)state; (void)tokens; (void)H; (void)out;
    (void)fn; (void)scale; (void)base; (void)norm; (void)input;
    return 0;
}

int dsv4_cuda_mhc_post_batch(const Dsv4CudaActivation *x,
                             const Dsv4CudaActivation *residual,
                             const Dsv4CudaActivation *state, int tokens, int H,
                             Dsv4CudaActivation *out) {
    (void)x; (void)residual; (void)state; (void)tokens; (void)H; (void)out;
    return 0;
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
    (void)context; (void)wo_a; (void)wo_b; (void)groups; (void)tokens;
    (void)output;
    return 0;
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
    (void)gate; (void)up; (void)down; (void)weights; (void)count; (void)limit;
    (void)y; (void)x;
    return 0;
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
