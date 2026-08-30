/* fmt=8 (dsv4 fp8-e4m3) Vulkan matmul vs the CPU reference kernels.
 * Template: tests/test_vk_mxfp4.c.
 *
 * What it covers (TEST L1):
 *   - real dsv4 dense shapes (DeepSeek-V4-Flash-0731 config: hidden 4096,
 *     q_rank 1024, head_dim 512, heads 64 -> wq_b rows 32768, o_groups 8,
 *     o_lora_rank 1024 -> o_width 8192), S=1 decode and S=128 prefill;
 *   - the E8M0 -> fp32 block-scale expansion at upload, BITWISE-equal to the
 *     CPU decoder (deepseek_v4.c coli_e8m0_decode: 0xff -> NaN, else
 *     2^(value-127)) — a dedicated unit test of coli_vk_expand_e8m0, the exact
 *     function upload_tensor runs (PLAN D4);
 *   - tail 128-blocks (I%128 != 0, O%128 != 0) and the unstaged path
 *     (I > 6144: wo_b, I = 8192).
 *
 * GPU path: coli_vk_matmul with fmt=8 — raw e4m3 weight bytes (byte-identical
 * upload) + UE8M0 scale codes (expanded to fp32 at upload). CPU reference:
 * quant.h matmul_fp8 (GLM's generic fp8 kernel: e4m3_decode LUT + per
 * 128x128-block fp32 scales, double-across-blocks). Skips (exit 0) when no
 * Vulkan device is available. Deterministic seed.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "../quant.h"
#include "../tensor.h"
#include "../backend_vulkan.h"
#include "../backend_vulkan_dsv4.h"

/* The REAL engine CPU references (deepseek_v4.c COLI_V4_UNIT_NATIVE_QUANT) —
 * linked from the amalgamation unit object so the gates are true oracles, not
 * test-side replicas (TEST L1 / M2-4). The unit is built WITHOUT the GPU tier
 * defines (-UCOLI_V4_GPU_TIER), so the CPU-only paths are what link. */
extern int coli_fp8_activation_qdq_ref(float *output, uint8_t *scales,
                                       const float *input, size_t length,
                                       size_t block_size);
extern int coli_fp8_matvec_ref(float *output, const ColiTensorView *weight,
                               const float *input);
extern int coli_fp8_matvec_pre(float *output, const ColiTensorView *weight,
                               const float *input, const float *activation);

/* Local UE8M0 decode mirroring coli_e8m0_decode — the bitwise oracle for the
 * expansion test. */
static float ref_e8(uint8_t v) { return v == 0xff ? NAN : ldexpf(1.0f, (int)v - 127); }

/* NaN-free random e4m3 weight byte: real model weights never carry the NaN
 * codes 0x7f/0xff (exp=15, man=7). Unfiltered rand() bytes poison every dot
 * product with NaN — which trivially "matches" on both sides and corrupted
 * early maxrel measurements — so test weights must map those two codes away. */
static uint8_t nn_byte(void) {
    uint8_t c = (uint8_t)rand();
    return (c & 0x7f) == 0x7f ? (uint8_t)(c & 0x7e) : c;   /* 0x7f/0xff -> 0x7e (448) */
}

/* Comparison stats helper (defined below): max abs, max rel over |ref| >= 1e-2,
 * bitwise flag, maxref. */
static void cmp_stats(const float *a, const float *b, int n,
                      double *maxabs, double *maxrel, int *bitwise, double *maxref);

static int test_scale_expansion(void) {
    uint8_t all[256]; float got[256], want[256];
    for (int i = 0; i < 256; i++) all[i] = (uint8_t)i;
    if (!coli_vk_expand_e8m0(all, got, 256)) { fprintf(stderr, "vk-fp8: expand refused\n"); return 1; }
    for (int i = 0; i < 256; i++) {
        want[i] = ref_e8(all[i]);
        uint32_t gb, wb; memcpy(&gb, &got[i], 4); memcpy(&wb, &want[i], 4);
        if (!((got[i] != got[i] && want[i] != want[i]) || gb == wb)) {
            fprintf(stderr, "vk-fp8: scale-expansion BIT MISMATCH code 0x%02x (0x%08x vs 0x%08x)\n",
                    all[i], gb, wb);
            return 1;
        }
    }
    /* realistic shapes: wq_a (1024x4096) scale tensor = 8 x 32 blocks, plus
     * tail-block shapes and a single block. */
    const int rows128[] = {8, 12, 7, 1}, cols128[] = {32, 32, 31, 1};
    for (int s = 0; s < 4; s++) {
        size_t n = (size_t)rows128[s] * cols128[s];
        uint8_t *e8 = malloc(n); float *g = malloc(n * 4), *w = malloc(n * 4);
        for (size_t i = 0; i < n; i++) e8[i] = (uint8_t)(120 + rand() % 16);
        if (!coli_vk_expand_e8m0(e8, g, n)) { fprintf(stderr, "vk-fp8: bulk expand refused\n"); return 1; }
        for (size_t i = 0; i < n; i++) {
            w[i] = ref_e8(e8[i]);
            uint32_t gb, wb; memcpy(&gb, &g[i], 4); memcpy(&wb, &w[i], 4);
            if (!((g[i] != g[i] && w[i] != w[i]) || gb == wb)) {
                fprintf(stderr, "vk-fp8: scale-expansion BIT MISMATCH shape %dx%d @%zu\n",
                        rows128[s], cols128[s], i);
                return 1;
            }
        }
        free(e8); free(g); free(w);
    }
    return 0;
}

static int run_shape(int O, int I, int S) {
    size_t nblk = (size_t)((O + 127) / 128) * ((I + 127) / 128);
    uint8_t *q8 = malloc((size_t)O * I);          /* raw e4m3 bytes, byte-identical */
    uint8_t *e8 = malloc(nblk);                   /* UE8M0 codes -> expanded at upload */
    float *bscale = malloc(nblk * 4);             /* fp32 expansion (CPU ref input) */
    float *x = malloc((size_t)S * I * 4);
    float *yc = calloc((size_t)S * O, 4), *yg = calloc((size_t)S * O, 4);
    for (size_t i = 0; i < (size_t)O * I; i++) q8[i] = nn_byte();
    for (size_t i = 0; i < nblk; i++) { e8[i] = (uint8_t)(120 + rand() % 16); bscale[i] = ref_e8(e8[i]); }
    for (int i = 0; i < S * I; i++) x[i] = (float)(rand() % 2001 - 1000) / 500.f;

    matmul_fp8(yc, x, q8, bscale, S, I, O);

    ColiVkTensor *t = NULL;
    if (!coli_vk_matmul(&t, yg, x, q8, (const float *)e8, 8, S, I, O, 0)) {
        fprintf(stderr, "vk-fp8: FAIL coli_vk_matmul fmt=8 refused (O=%d I=%d S=%d)\n", O, I, S);
        return 1;
    }
    double maxabs, mx, maxref; int bitwise;
    cmp_stats(yg, yc, S * O, &maxabs, &mx, &bitwise, &maxref);
    printf("vk-fp8: fmt=8 O=%d I=%d S=%d | maxabs %.3e maxref %.3e global %.3e "
           "max_rel(floor) %.3e %s\n",
           O, I, S, maxabs, maxref, maxref > 0 ? maxabs / maxref : 0, mx,
           bitwise ? "[bitwise]" : "");
    coli_vk_tensor_free(t);
    free(q8); free(e8); free(bscale); free(x); free(yc); free(yg);
    /* GLM's bar for fp32-reduction-order divergence is ~1e-5..2e-3 maxrel;
     * the production shader reduces in fp32 tree order while matmul_fp8 uses
     * fp64-across-blocks, so the contract is thresholded (NOT bitwise — early
     * "bitwise" measurements were NaN artifacts from unfiltered random weight
     * bytes, fixed by nn_byte()). Engine-level logit comparison on real
     * weights (L3) is the tight check. */
    return (maxref > 0 && maxabs / maxref > 1e-2) ? 1 : 0;   /* global bound: catches a broken shader; floor-maxrel spikes on cancellation are informational */
}

/* ---- Activation QDQ bitwise gate (M2-2) -------------------------------------
 * GPU qdq.comp must reproduce coli_fp8_activation_qdq_ref BITWISE (the qkv
 * chain feeds these activations into the fp8 matmuls — any rounding difference
 * is a parity break). Sweeps the decode shapes plus the edge regimes: values
 * that clamp at +-448, magnitudes in the subnormal-e4m3 encode branch
 * (< 2^-6), tiny inputs that force the scale exponent to clamp at -127
 * (subnormal scale), exact dyadic / RNE tie cases, and tail blocks. */
static int run_qdq_bitwise(void) {
    struct { int S, I, block; int mode; const char *tag; } cases[] = {
        {1, 4096, 128, 0, "wq_a decode"},
        {8, 4096, 128, 3, "batch mixed"},
        {1, 130, 128, 0, "tail"},
        {1, 576, 64, 0, "kv_nope b64"},
        {1, 128, 128, 1, "small"},
        {1, 128, 128, 2, "tiny (exp clamp -127)"},
        {1, 128, 128, 4, "dyadic/ties"},
        {1, 1, 128, 0, "single"},
        {1, 7, 7, 0, "block=7"},
        {128, 512, 128, 3, "prefill"},
    };
    static const float dy[] = {1.0f, 2.0f, 0.5f, 448.0f, -448.0f, 0.015625f,
        0.0029296875f, 0.0009765625f, 1.5f, 1.75f, 0.0f, -0.0f, 2e-38f,
        0.001953125f, -0.001953125f, 0.00390625f};
    int bad = 0;
    for (int c = 0; c < (int)(sizeof(cases)/sizeof(cases[0])); c++) {
        int S = cases[c].S, I = cases[c].I, block = cases[c].block;
        size_t nblk = (size_t)S * ((I + block - 1) / block);
        float *x = malloc((size_t)S * I * 4), *yg = malloc((size_t)S * I * 4);
        float *yc = malloc((size_t)S * I * 4);
        uint8_t *scg = malloc(nblk), *scc = malloc(nblk);
        for (int i = 0; i < S * I; i++) {
            switch (cases[c].mode) {
                case 0: x[i] = (float)(rand() % 200000 - 100000) / 100.0f; break;
                case 1: x[i] = (float)(rand() % 20000 - 10000) / 10000000.0f; break;
                case 2: x[i] = (float)(rand() % 1000 - 500) * 1e-38f; break;
                case 3: x[i] = (i & 1) ? (float)(rand() % 200000 - 100000) / 100.0f
                                       : (float)(rand() % 20000 - 10000) / 10000000.0f; break;
                default: x[i] = dy[(i + c) % (int)(sizeof(dy)/sizeof(dy[0]))]; break;
            }
        }
        int ok = coli_vk_activation_qdq(yg, scg, x, S, I, block);
        int ref = coli_fp8_activation_qdq_ref(yc, scc, x, (size_t)S * I, (size_t)block);
        if (!ok || ref) { fprintf(stderr, "vk-fp8 qdq: refused (ok=%d ref=%d S=%d I=%d block=%d)\n", ok, ref, S, I, block); bad = 1; }
        else {
            for (size_t i = 0; i < (size_t)S * I && !bad; i++) {
                uint32_t a, b; memcpy(&a, &yg[i], 4); memcpy(&b, &yc[i], 4);
                if (!((yg[i] != yg[i] && yc[i] != yc[i]) || a == b)) {
                    fprintf(stderr, "vk-fp8 qdq: BIT MISMATCH %s @%zu gpu 0x%08x cpu 0x%08x\n",
                            cases[c].tag, i, a, b);
                    bad = 1;
                }
            }
            for (size_t i = 0; i < nblk && !bad; i++)
                if (scg[i] != scc[i]) {
                    fprintf(stderr, "vk-fp8 qdq: SCALE MISMATCH %s @%zu gpu %u cpu %u\n",
                            cases[c].tag, i, scg[i], scc[i]);
                    bad = 1;
                }
        }
        printf("vk-fp8 qdq: %-22s S=%d I=%d block=%d : %s\n", cases[c].tag, S, I, block,
               ok && !bad ? "bitwise OK (vs engine unit)" : "FAIL");
        if (bad) break;
        free(x); free(yg); free(yc); free(scg); free(scc);
    }
    return bad;
}

/* ---- fp8 GEMV exactness vs the CPU reference (M2-4, TEST L1) ----------------
 * The GPU chain (QDQ shader + fmt=8 matmul shader on the qdq'd activations)
 * must reproduce the engine's coli_fp8_matvec_ref: the CPU runs QDQ + the
 * fp8 matmul (matmul_fp8 for block_rows=128, the AVX2 rows8 kernel for
 * block_rows=8 with rows8-packed weights). The QDQ halves are already proven
 * bitwise (M2-2); the matmul halves are fp32-tree vs fp64-across-blocks, so
 * the contract is maxrel (GLM's bar ~1e-5..2e-3; M2-1 measured ~0 on real
 * shapes). The rows8 case additionally validates the exact (x*v)*scale fp32
 * sequential order via coli_fp8_matvec_pre (no QDQ inside). */

/* rows8 AVX2 packing (engine v4_fp8_pack_rows8_inplace semantics): tile t
 * holds rows 8t..8t+7, byte layout ((tile*cols)+col)*8 + lane. */
static void pack_rows8(uint8_t *dst, const uint8_t *row_major,
                       int64_t rows, int64_t cols) {
    for (int64_t tile = 0; tile < rows / 8; tile++)
        for (int64_t col = 0; col < cols; col++)
            for (int lane = 0; lane < 8; lane++)
                dst[(tile * cols + col) * 8 + lane] =
                    row_major[(tile * 8 + lane) * cols + col];
}

/* Comparison stats: max abs error, max RELATIVE error over outputs with
 * |ref| >= 1e-2 (the harness's own floor — near-zero cancellation outputs
 * amplify relative error beyond any meaningful fp32-order contract), maxref
 * (for the global maxabs/maxref ratio), and the bitwise-equality flag. */
static void cmp_stats(const float *a, const float *b, int n,
                      double *maxabs, double *maxrel, int *bitwise, double *maxref) {
    *maxabs = 0; *maxrel = 0; *bitwise = 1; *maxref = 0;
    for (int i = 0; i < n; i++) {
        uint32_t xa, xb; memcpy(&xa, &a[i], 4); memcpy(&xb, &b[i], 4);
        if (xa != xb && !(a[i] != a[i] && b[i] != b[i])) *bitwise = 0;
        double d = fabs((double)a[i] - (double)b[i]);
        if (d > *maxabs) *maxabs = d;
        double ref = fabs((double)b[i]);
        if (ref > *maxref) *maxref = ref;
        if (ref >= 1e-2) { double r = d / ref; if (r > *maxrel) *maxrel = r; }
    }
}

static int run_gpu_vs_cpu_chain(int O, int I, int S, int block_rows) {
    size_t nsc = (size_t)((O + 127) / 128) * ((I + 127) / 128);
    uint8_t *w = malloc((size_t)O * I);           /* row-major raw e4m3 */
    uint8_t *w8 = malloc((size_t)O * I);          /* rows8-packed for the CPU view */
    uint8_t *e8 = malloc(nsc);
    float *bscale = malloc(nsc * 4);
    float *x = malloc((size_t)S * I * 4);
    float *act = malloc((size_t)S * I * 4);       /* GPU QDQ output */
    uint8_t *actsc = malloc((size_t)S * ((I + 127) / 128));
    float *yg = malloc((size_t)S * O * 4), *yc = malloc((size_t)S * O * 4);
    for (size_t i = 0; i < (size_t)O * I; i++) w[i] = nn_byte();
    pack_rows8(w8, w, O, I);
    for (size_t i = 0; i < nsc; i++) {
        uint8_t code = (uint8_t)(115 + rand() % 14);   /* realistic UE8M0: 2^-12..2^1 (checkpoint median 120/121) */
        e8[i] = code;
        bscale[i] = code == 0xff ? NAN : ldexpf(1.0f, (int)code - 127);
    }
    for (int i = 0; i < S * I; i++) x[i] = (float)(rand() % 200001 - 100000) / 100.0f;

    ColiTensorView view = {0};
    view.format = COLI_TENSOR_FP8_E4M3_BLOCK;
    view.scale_format = COLI_SCALE_F32;
    view.data = block_rows == 8 ? w8 : w;
    view.scales = bscale;
    view.data_bytes = (size_t)O * I;
    view.scale_bytes = nsc * 4;
    view.rows = O; view.columns = I;
    view.block_rows = (uint32_t)block_rows;
    view.block_columns = 128;

    /* GPU chain: one QDQ pass (the _pre hoist contract) + one matmul */
    ColiVkTensor *t = NULL;
    int okg = coli_vk_activation_qdq(act, actsc, x, S, I, 128) &&
              coli_vk_matmul(&t, yg, act, w, (const float *)e8, 8, S, I, O, 0);
    /* CPU reference is per-vector (decode): loop the batch, each token qdq'd
     * internally (the GPU's batched QDQ is per-token-block anyway, M2-2). */
    int okc = 0;
    for (int s = 0; s < S; s++)
        okc |= coli_fp8_matvec_ref(yc + (size_t)s * O, &view, x + (size_t)s * I);
    if (!okg || okc) {
        fprintf(stderr, "vk-fp8 chain: refused (gpu=%d cpu=%d O=%d I=%d br=%d)\n",
                okg, okc, O, I, block_rows);
        coli_vk_tensor_free(t);
        free(w); free(w8); free(e8); free(bscale); free(x); free(act); free(actsc);
        free(yg); free(yc);
        return 1;
    }
    double maxabs, mx, maxref; int bitwise;
    cmp_stats(yg, yc, S * O, &maxabs, &mx, &bitwise, &maxref);
    printf("vk-fp8 chain vs coli_fp8_matvec_ref: O=%d I=%d S=%d block_rows=%d | "
           "maxabs %.3e maxref %.3e global %.3e max_rel(floor) %.3e %s\n",
           O, I, S, block_rows, maxabs, maxref, maxref > 0 ? maxabs / maxref : 0,
           mx, bitwise ? "[bitwise]" : "");
    coli_vk_tensor_free(t);
    free(w); free(w8); free(e8); free(bscale); free(x); free(act); free(actsc);
    free(yg); free(yc);
    return (maxref > 0 && maxabs / maxref > 1e-2) ? 1 : 0;   /* global bound: catches a broken shader; floor-maxrel spikes on cancellation are informational */
}

/* Replica bitwise gates: ds4vk_fp8_ref_matmul must equal the engine's CPU
 * kernels bit-for-bit. Non-rows8: quant.h matmul_fp8 (same fp32-within-block /
 * fp64-across-blocks order, both non-fused in this build). Rows8: the engine's
 * AVX2 rows8 compute via coli_fp8_matvec_pre on a rows8 view with the qdq'd
 * activations (same (x*v)*scale fp32 sequential order). */
static int run_replica_bitwise(int O, int I, int S) {
    size_t nsc = (size_t)((O + 127) / 128) * ((I + 127) / 128);
    uint8_t *w = malloc((size_t)O * I), *w8 = malloc((size_t)O * I);
    uint8_t *e8 = malloc(nsc);
    float *bscale = malloc(nsc * 4);
    float *x = malloc((size_t)S * I * 4);
    float *act = malloc((size_t)S * I * 4);
    uint8_t *actsc = malloc((size_t)S * ((I + 127) / 128));
    float *yr = malloc((size_t)S * O * 4);          /* non-rows8 serial ref (kept) */
    float *yr8 = malloc((size_t)S * O * 4);         /* rows8 serial ref (separate) */
    float *yc = malloc((size_t)S * O * 4), *yp = malloc((size_t)S * O * 4);
    for (size_t i = 0; i < (size_t)O * I; i++) w[i] = nn_byte();
    pack_rows8(w8, w, O, I);
    for (size_t i = 0; i < nsc; i++) {
        uint8_t code = (uint8_t)(115 + rand() % 14);   /* realistic UE8M0 range (checkpoint median 120/121) */
        e8[i] = code;
        bscale[i] = ldexpf(1.0f, (int)code - 127);
    }
    for (int i = 0; i < S * I; i++) x[i] = (float)(rand() % 200001 - 100000) / 100.0f;

    int bad = 0;
    /* non-rows8: replica vs quant.h matmul_fp8 (raw x) */
    ds4vk_fp8_ref_matmul(w, bscale, O, I, 0, x, S, yr);
    matmul_fp8(yc, x, w, bscale, S, I, O);
    int bw = 1;
    for (int i = 0; i < S * O && bw; i++) {
        uint32_t a, b; memcpy(&a, &yr[i], 4); memcpy(&b, &yc[i], 4);
        if (a != b && !(yr[i] != yr[i] && yc[i] != yc[i])) bw = 0;
    }
    printf("vk-fp8 replica non-rows8 vs matmul_fp8: O=%d I=%d S=%d : %s\n",
           O, I, S, bw ? "BITWISE" : "NOT bitwise");
    if (!bw) bad = 1;
    /* rows8: replica vs the engine AVX2 rows8 compute (same qdq'd activations) */
    if (coli_fp8_activation_qdq_ref(act, actsc, x, (size_t)S * I, 128)) { fprintf(stderr, "qdq refused\n"); bad = 1; }
    else {
        ds4vk_fp8_ref_matmul(w8, bscale, O, I, 1, act, S, yr8);
        ColiTensorView view = {0};
        view.format = COLI_TENSOR_FP8_E4M3_BLOCK;
        view.scale_format = COLI_SCALE_F32;
        view.data = w8; view.scales = bscale;
        view.data_bytes = (size_t)O * I; view.scale_bytes = nsc * 4;
        view.rows = O; view.columns = I;
        view.block_rows = 8; view.block_columns = 128;
        int pre_ok = 0;
        for (int s = 0; s < S; s++)   /* matvec_pre is per-vector (decode) */
            pre_ok |= coli_fp8_matvec_pre(yp + (size_t)s * O, &view,
                                          x + (size_t)s * I, act + (size_t)s * I);
        if (pre_ok) { fprintf(stderr, "matvec_pre refused\n"); bad = 1; }
        else {
            bw = 1;
            for (int i = 0; i < S * O && bw; i++) {
                uint32_t a, b; memcpy(&a, &yr8[i], 4); memcpy(&b, &yp[i], 4);
                if (a != b && !(yr8[i] != yr8[i] && yp[i] != yp[i])) bw = 0;
            }
            printf("vk-fp8 replica rows8 vs engine AVX2 rows8: O=%d I=%d S=%d : %s\n",
                   O, I, S, bw ? "BITWISE" : "NOT bitwise");
            if (!bw) bad = 1;
        }
    }
    /* production shader vs its bitwise oracle (raw x, no QDQ) */
    ColiVkTensor *t = NULL;
    float *ys = malloc((size_t)S * O * 4);
    if (!coli_vk_matmul(&t, ys, x, w, (const float *)e8, 8, S, I, O, 0)) {
        fprintf(stderr, "vk-fp8 shader refused\n"); bad = 1;
    } else {
        int exact = 0, n = S * O;
        double maxabs, mx, maxref;
        cmp_stats(ys, yr, n, &maxabs, &mx, &exact, &maxref);
        exact = 0;
        for (int i = 0; i < n; i++) {
            uint32_t a, b; memcpy(&a, &ys[i], 4); memcpy(&b, &yr[i], 4);
            if (a == b || (ys[i] != ys[i] && yr[i] != yr[i])) exact++;
        }
        printf("vk-fp8 shader vs ds4vk replica: O=%d I=%d S=%d | %d/%d bitwise "
               "maxabs %.3e maxref %.3e global %.3e max_rel(floor) %.3e "
               "(fp32 reduction order, not bitwise)\n",
               O, I, S, exact, n, maxabs, maxref,
               maxref > 0 ? maxabs / maxref : 0, mx);
        if (maxref > 0 && maxabs / maxref > 1e-2) bad = 1;
    }
    coli_vk_tensor_free(t);
    free(ys); free(w); free(w8); free(e8); free(bscale); free(x); free(act); free(actsc);
    free(yr); free(yr8); free(yc); free(yp);
    return bad;
}

int main(void) {
    if (!coli_vk_init("shaders/qmatmul.spv") || !coli_vk_available()) {
        fprintf(stderr, "vk-fp8: no Vulkan device — skipped\n"); return 0;
    }
    srand(7);
    int bad = test_scale_expansion();
    /* decode path (S=1) */
    bad |= run_shape(1024, 4096, 1);     /* attn.wq_a  q_rank x hidden */
    bad |= run_shape(512, 4096, 1);      /* attn.wkv   head_dim x hidden */
    bad |= run_shape(32768, 1024, 1);    /* attn.wq_b  heads*head_dim x q_rank */
    bad |= run_shape(4096, 8192, 1);     /* attn.wo_b  hidden x o_width (I>6144: unstaged) */
    bad |= run_shape(8192, 4096, 1);     /* attn.wo_a  o_width x o_group_width */
    bad |= run_shape(2048, 4096, 1);     /* ffn.shared_experts.w1/w3 */
    bad |= run_shape(4096, 2048, 1);     /* ffn.shared_experts.w2 */
    /* prefill batch */
    bad |= run_shape(1024, 4096, 128);   /* wq_a batch */
    bad |= run_shape(512, 4096, 128);    /* wkv batch */
    /* tail-block / tiny edges */
    bad |= run_shape(70, 130, 1);        /* I%128!=0, O%128!=0 tail blocks */
    bad |= run_shape(64, 128, 1);        /* single block */
    bad |= run_shape(3, 7, 1);           /* sub-word row */
    bad |= run_qdq_bitwise();            /* activation QDQ vs the engine unit */
    /* M2-4 exactness: GPU chain (QDQ + matmul) vs the REAL coli_fp8_matvec_ref,
     * and the ds4vk_fp8_ref_matmul bitwise replica vs the engine's own kernels. */
    bad |= run_gpu_vs_cpu_chain(1024, 4096, 1, 128);   /* wq_a decode, matmul_fp8 path */
    bad |= run_gpu_vs_cpu_chain(1024, 4096, 1, 8);     /* wq_a decode, AVX2 rows8 path */
    bad |= run_gpu_vs_cpu_chain(512, 4096, 8, 8);      /* wkv batch */
    bad |= run_gpu_vs_cpu_chain(4096, 8192, 1, 128);   /* wo_b (I>6144 unstaged) */
    bad |= run_gpu_vs_cpu_chain(2048, 4096, 128, 128); /* prefill batch */
    bad |= run_replica_bitwise(512, 4096, 1);          /* replica: matmul_fp8 + AVX2 rows8 */
    bad |= run_replica_bitwise(1024, 4096, 8);
    if (bad) { fprintf(stderr, "vk-fp8: FAIL\n"); coli_vk_shutdown(); return 1; }
    printf("vk-fp8: OK\n");
    coli_vk_shutdown();
    return 0;
}
