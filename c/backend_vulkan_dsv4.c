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
#include <sys/stat.h>
#include <unistd.h>

/* ---- opaque handle definitions (ABI types are forward-declared in the
 *      header; the engine only passes these pointers around) ---- */

typedef struct Dsv4CudaTensor {
    int device;
    long long bytes;   /* uploaded weight bytes (real accounting) */
    int fmt;           /* 8=fp8-e4m3, 9=fp8-bf16-rounded, 4=fp4, 16=bf16, 32=f32 */
    int O, I;
    ColiVkTensor *vk;  /* the arena handle (M2-3: real uploads; M3a compute) */
} Dsv4CudaTensor;

typedef struct Dsv4CudaActivation {
    int device;
    long long elements;
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

void dsv4_cuda_shutdown(void) { coli_vk_shutdown(); }

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
    /* Deferred: bf16 mirrors (compressor / indexer-compressor projections)
     * only upload under COLI_CUDA_ATTN_BATCH=1, whose batched ops are M4. */
    (void)w; (void)O; (void)I; (void)device;
    if (t) *t = NULL;
    fprintf(stderr, "v4_gpu vk upload bf16 deferred (batched attention ops are "
                    "M4); tensor stays CPU\n");
    return 0;
}

int dsv4_cuda_upload_f32(Dsv4CudaTensor **t, const float *w, int O, int I,
                         int device) {
    /* Deferred: f32 mirrors (route gate, mHC fn/scale/base, head branch
     * norms) belong to the M3a/M3c ops; the tensors stay CPU until then. */
    (void)w; (void)O; (void)I; (void)device;
    if (t) *t = NULL;
    fprintf(stderr, "v4_gpu vk upload f32 deferred (route/mHC/head ops land "
                    "M3a-M3c); tensor stays CPU\n");
    return 0;
}

int dsv4_cuda_tensor_refill_fp4(Dsv4CudaTensor *t, const uint8_t *w,
                                const uint8_t *scale, int O, int I, int sync) {
    (void)t; (void)w; (void)scale; (void)O; (void)I; (void)sync;
    return 0; /* no fp4 mirrors in v1 (M5) */
}

/* ---- tensor/activation handles ---- */

void dsv4_cuda_tensor_free(Dsv4CudaTensor *t) {
    if (!t) return;
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

void dsv4_cuda_activation_free(Dsv4CudaActivation *a) { free(a); }

int dsv4_cuda_activation_upload(Dsv4CudaActivation *a, const float *x,
                                long long elements) {
    (void)a; (void)x; (void)elements;
    return 0;
}

int dsv4_cuda_activation_download(float *x, const Dsv4CudaActivation *a,
                                  long long elements) {
    (void)x; (void)a; (void)elements;
    return 0;
}

int dsv4_cuda_activation_copy(Dsv4CudaActivation *dst,
                              const Dsv4CudaActivation *src,
                              long long elements) {
    (void)dst; (void)src; (void)elements;
    return 0;
}

int dsv4_cuda_activation_copy_range(Dsv4CudaActivation *dst,
                                    long long dst_offset,
                                    const Dsv4CudaActivation *src,
                                    long long src_offset, long long elements) {
    (void)dst; (void)dst_offset; (void)src; (void)src_offset; (void)elements;
    return 0;
}

int dsv4_cuda_activation_sync(const Dsv4CudaActivation *a) {
    (void)a;
    return 0;
}

int dsv4_cuda_activation_device(const Dsv4CudaActivation *a) {
    return a ? a->device : -1;
}

int dsv4_cuda_decode_state_set(int device, int token, int position) {
    (void)device; (void)token; (void)position;
    return 0;
}

/* ---- compute ops: every one fails -> engine falls back to CPU (D6) ---- */

int dsv4_cuda_matvec(Dsv4CudaTensor *t, float *y, const float *x) {
    (void)t; (void)y; (void)x;
    return 0;
}

int dsv4_cuda_matvec_grouped(Dsv4CudaTensor *t, float *y, const float *x,
                             int groups) {
    (void)t; (void)y; (void)x; (void)groups;
    return 0;
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
    (void)input; (void)gate; (void)bias; (void)fixed_ids; (void)routed_scale;
    (void)ids; (void)weights;
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
    (void)device; (void)layer; (void)q; (void)chunk; (void)chunk_start;
    (void)sinks; (void)meta; (void)abs_base; (void)comp_limit; (void)heads;
    (void)dim; (void)tokens; (void)scale; (void)out;
    return 0;
}

int dsv4_cuda_sparse_attn_batch_cached_idx(int device, int layer,
                                           const float *q, const float *chunk,
                                           int chunk_start, const float *sinks,
                                           const int *meta, const int *sel,
                                           int selstride, int abs_base,
                                           int comp_limit, int heads, int dim,
                                           int tokens, float scale, float *out) {
    (void)device; (void)layer; (void)q; (void)chunk; (void)chunk_start;
    (void)sinks; (void)meta; (void)sel; (void)selstride; (void)abs_base;
    (void)comp_limit; (void)heads; (void)dim; (void)tokens; (void)scale;
    (void)out;
    return 0;
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
    (void)device; (void)layer; (void)rows; (void)start_pos; (void)count;
    (void)window; (void)dim;
    return 0;
}

int dsv4_cuda_kv_comp_append(int device, int layer, const float *rows,
                             int start_idx, int count, int dim) {
    (void)device; (void)layer; (void)rows; (void)start_idx; (void)count;
    (void)dim;
    return 0;
}

int dsv4_cuda_head_argmax(Dsv4CudaTensor *t, const float *x, int *id,
                          float *value) {
    (void)t; (void)x; (void)id; (void)value;
    return 0;
}

int dsv4_cuda_final_argmax(const Dsv4CudaActivation *residual,
                           Dsv4CudaTensor *fn, Dsv4CudaTensor *scale,
                           Dsv4CudaTensor *base, Dsv4CudaTensor *norm,
                           Dsv4CudaTensor *head, int M, int H, float eps,
                           float pre_eps, int *id, float *value) {
    (void)residual; (void)fn; (void)scale; (void)base; (void)norm; (void)head;
    (void)M; (void)H; (void)eps; (void)pre_eps; (void)id; (void)value;
    return 0;
}

int dsv4_cuda_mhc_pre(const Dsv4CudaActivation *residual, Dsv4CudaTensor *fn,
                      Dsv4CudaTensor *scale, Dsv4CudaTensor *base, int M, int H,
                      float rms_eps, float pre_eps, float sink_eps,
                      float post_mult, int sink_iters, Dsv4CudaActivation *state,
                      Dsv4CudaActivation *input) {
    (void)residual; (void)fn; (void)scale; (void)base; (void)M; (void)H;
    (void)rms_eps; (void)pre_eps; (void)sink_eps; (void)post_mult;
    (void)sink_iters; (void)state; (void)input;
    return 0;
}

int dsv4_cuda_mhc_pre_norm(const Dsv4CudaActivation *residual,
                           Dsv4CudaTensor *fn, Dsv4CudaTensor *scale,
                           Dsv4CudaTensor *base, Dsv4CudaTensor *norm, int M,
                           int H, float rms_eps, float pre_eps, float sink_eps,
                           float post_mult, int sink_iters, float norm_eps,
                           Dsv4CudaActivation *state, Dsv4CudaActivation *input) {
    (void)residual; (void)fn; (void)scale; (void)base; (void)norm; (void)M;
    (void)H; (void)rms_eps; (void)pre_eps; (void)sink_eps; (void)post_mult;
    (void)sink_iters; (void)norm_eps; (void)state; (void)input;
    return 0;
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
    (void)x; (void)residual; (void)state; (void)M; (void)H; (void)out;
    return 0;
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
